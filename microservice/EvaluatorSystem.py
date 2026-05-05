import argparse
import asyncio
import torch
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM, ModernBertForSequenceClassification
from typing import List, Optional


class Evaluator:
    def __init__(self, model_name_or_path: str, instruction: str, peft_path: Optional[str] = None, max_length: int = 2048):
        self.model_name = model_name_or_path
        self.instruction = instruction
        self.max_length = max_length
        if 'qwen3-reranker' in self.model_name.lower():
            self.reranker_tok = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
            base = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
            self.reranker = PeftModel.from_pretrained(base, peft_path).eval()
            self.token_false_id = self.reranker_tok.convert_tokens_to_ids("no")
            self.token_true_id = self.reranker_tok.convert_tokens_to_ids("yes")

        elif 'modernbert' in self.model_name.lower():
            assert peft_path == None
            self.reranker_tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.reranker = ModernBertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1, device_map='auto', trust_remote_code=True)

        else:
            raise NotImplementedError

    def rerank(self, query: str, docs: List[str]):
        if 'qwen3-reranker' in self.model_name.lower():
            def format_instruction(instruction: str, query: str, doc: str) -> str:
                return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)

            def process_inputs(pairs: List[str]):
                inputs = self.reranker_tok(
                    pairs, padding=False, truncation=True,
                    return_attention_mask=False, max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens)
                )
                for i, ele in enumerate(inputs['input_ids']):
                    inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
                inputs = self.reranker_tok.pad(inputs, padding=True, return_tensors="pt")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.reranker.device)
                return inputs

            @torch.no_grad()
            def compute_logits(inputs, **kwargs) -> List[float]:
                batch_scores = self.reranker(**inputs).logits[:, -1, :]
                true_vector = batch_scores[:, self.token_true_id]
                false_vector = batch_scores[:, self.token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
                return scores

            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = self.reranker_tok.encode(prefix, add_special_tokens=False)
            suffix_tokens = self.reranker_tok.encode(suffix, add_special_tokens=False)
            queries = [query] * len(docs)
            pairs = [format_instruction(self.instruction, query, doc) for query, doc in zip(queries, docs)]

            # Tokenize the input texts
            inputs = process_inputs(pairs)
            scores = compute_logits(inputs)

            return scores

        elif 'modernbert' in self.model_name.lower():
            inputs = self.reranker_tok(
                [query] * len(docs),
                docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.reranker.device) for k, v in inputs.items()}
            with torch.no_grad():
                scores = torch.nn.functional.sigmoid(self.reranker(**inputs).logits / 4).cpu().numpy().reshape(-1).tolist()

            return scores

        else:
            raise NotImplementedError


class EvaluatorRequest(BaseModel):
    query: str
    docs: List[str]


evaluator = None
def init_service():
    global evaluator
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default='answerdotai/ModernBERT-base', help='The model name for reranking, e.g., "Qwen3-Reranker-0.6B" or "ModernBERT-base", or the path to the local fine-tuned checkpoint.')
    parser.add_argument('--instruction', type=str, default='Given a Query, judge whether the Document is a correct answer to the Query.')
    parser.add_argument('--peft-path', type=str, default=None)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--port', type=int, default=8880)
    local_args, _ = parser.parse_known_args()
    if evaluator:
        return
    evaluator = Evaluator(local_args.model_name_or_path, instruction=local_args.instruction, peft_path=local_args.peft_path, max_length=local_args.max_length)
    print('Initialize evaluator.')
    return local_args

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_service()
    yield

app = FastAPI(lifespan=lifespan)
global_lock = asyncio.Lock()

@app.post('/reranker')
async def reranker_documents(request: EvaluatorRequest):
    start_time = time()
    async with global_lock:
        scores = evaluator.rerank(request.query, request.docs)
    stop_time = time()
    print(scores)
    print(f"Time cost: {stop_time - start_time:.2f} s")
    return {'scores': scores}


if __name__ == '__main__':
    local_args = init_service()
    uvicorn.run('microservice.EvaluatorSystem:app', host='0.0.0.0', port=local_args.port, log_level='info', loop='asyncio', http='h11')
