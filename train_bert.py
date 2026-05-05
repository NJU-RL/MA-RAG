import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass, field
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.auto_docstring import auto_docstring
from typing import Optional, Union
from utils import copy_files
torch._dynamo.config.disable = True


@dataclass
class DataArguments:
    train_file: str = field(metadata={'help': 'Path to training data'})
    eval_file: str = field(metadata={'help': 'Path to evaluation data'})
    max_length: int = field(default=4096)
    instruction: str = field(default='Given a Query, judge whether the Document is a correct answer to the Query.')


class WeightedModernBertForSequenceClassification(ModernBertForSequenceClassification):
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # if self.config.problem_type is None:
            #     if self.num_labels == 1:
            #         self.config.problem_type = "regression"
            #     elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            #         self.config.problem_type = "single_label_classification"
            #     else:
            #         self.config.problem_type = "multi_label_classification"

            # if self.config.problem_type == "regression":
            #     loss_fct = MSELoss()
            #     if self.num_labels == 1:
            #         loss = loss_fct(logits.squeeze(), labels.squeeze())
            #     else:
            #         loss = loss_fct(logits, labels)
            # elif self.config.problem_type == "single_label_classification":
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # elif self.config.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(logits, labels)
            assert logits.shape[-1] == 1
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).float())

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class ModelArguments:
    model_name: str = field(default='answerdotai/ModernBERT-base', metadata={'help': 'Path to tokenizer'})


def stable_sigmoid(x: np.ndarray):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def softmax(x: np.ndarray, axis: int = -1):
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = labels.reshape(-1)
    if predictions.shape[-1] > 1:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        preds = np.argmax(predictions, axis=-1)
        scores = softmax(predictions)[:, 1]
    else:
        predictions = predictions.reshape(-1)
        preds = predictions > 0
        scores = stable_sigmoid(predictions)
    accuracy = (preds == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0.0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'prediction': preds.mean(), 'mean_pos_score': scores[labels == 1].mean(), 'mean_pos_std': scores[labels == 1].std(), 'mean_neg_score': scores[labels == 0].mean(), 'mean_neg_std': scores[labels == 0].std(), 'pred_mean_pos_score': scores[preds == True].mean(), 'pred_mean_neg_score': scores[preds == False].mean()}

def build_training_args():
    model_args = ModelArguments('answerdotai/ModernBERT-base')
    bert_config = ModernBertConfig.from_pretrained(
        model_args.model_name,
        classifier_pooling='cls',
        num_labels=1,
    )
    data_args = DataArguments(
        train_file='./datasets/train_sigmoid.csv',
        eval_file='./datasets/eval_sigmoid.csv',
        max_length=2048,
    )
    training_args = TrainingArguments(
        output_dir='./runs/training/ModernBERT-base/test',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=1e-4,
        warmup_ratio=0.3,
        # lr_scheduler_type='constant_with_warmup',
        # lr_scheduler_kwargs={'num_warmup_steps': 100},
        lr_scheduler_type='cosine_warmup_with_min_lr',
        lr_scheduler_kwargs={'min_lr': 1e-7},
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        fp16=True,
        group_by_length=False,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        logging_strategy='steps',
        logging_steps=1,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=2,
        save_only_model=True,
        save_safetensors=True,
        ddp_timeout=3600,
        load_best_model_at_end=False,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        report_to='tensorboard',
        seed=42,
        data_seed=42,
        ddp_find_unused_parameters=False,
        eval_on_start=True,
    )
    return model_args, bert_config, data_args, training_args

def main():
    def preprocess_function(examples):
        # fmt_texts = [f"Given a Question, judge whether the Answer is correct.\n\nQuestion:\n{q}\n\nAnswer:\n{a}" for q, a in zip(examples['question'], examples['response'])]
        toks = tokenizer(
            examples['question'],
            examples['response'],
            padding=False,
            truncation=True,
            max_length=data_args.max_length,
            return_attention_mask=False,
        )
        assert len(toks['input_ids']) == len(examples['label'])

        return {
            'input_ids': toks['input_ids'],
            'labels': [int(label) for label in examples['label']]
        }

    model_args, bert_config, data_args, training_args = build_training_args()
    copy_files(training_args.output_dir, files=['train_bert.py'])

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    model = WeightedModernBertForSequenceClassification.from_pretrained(model_args.model_name, config=bert_config).train()
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    dataset = load_dataset('csv', data_files={'train': data_args.train_file, 'eval': data_args.eval_file})
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    main()