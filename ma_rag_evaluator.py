import argparse
import logging
from pathlib import Path


################################## parameters ##################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', '-dp', type=str, default='./datasets/MedMCQA.json')
parser.add_argument('--model-name', '-mn', type=str, default='qwen3-8b')
parser.add_argument('--base-url', type=str, default=None)
parser.add_argument('--exp', '-e', type=str, default='test', help='Experiment name.')
parser.add_argument('--num-workers', '-nw', type=int, default=8, help='Number of inferences per round.')
parser.add_argument('--num-round', '-nr', type=int, default=8, help='Number of round.')
parser.add_argument('--start-id', '-si', type=int, default=0, help='Evaluate the dataset from the start-id to end-id, both closed.')
parser.add_argument('--end-id', '-ei', type=int, default=-1)
local_args = parser.parse_args()
local_args.dataset_path = Path(local_args.dataset_path)
################################################################################

#################################### logger ####################################
log_dir = Path(f"runs/MA-RAG/{local_args.dataset_path.stem}/{local_args.model_name}/exp_{local_args.exp}".lower())
log_dir.mkdir(parents=True, exist_ok=True)
# logger = get_logger(log_dir)
logging.basicConfig(level=logging.WARNING, filename=log_dir / 'logging.log', filemode='a', format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(vars(local_args))

from utils import copy_files
copy_files(log_dir, folders=['microservice'], files=['ma_rag_evaluator.py'])
################################################################################

################################## evaluation ##################################
import json
import re
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from liquid import Template
from tqdm import tqdm
from microservice import CustomLanguageModel
from utils import get_query, judger, inference, combine_docs, calculate_accuracy, RetrievalService, RerankerSyetem

# load dataset
with open(local_args.dataset_path, 'r', encoding='utf-8') as fp:
    datasets = json.load(fp)
start_id = local_args.start_id if local_args.start_id >= 0 else 0
end_id = local_args.end_id if local_args.end_id >= 0 else 1000000

# filter exist experiments
if (log_dir / 'evaluations').exists():
    exist_question_ids = [int(path.stem.split('_')[-1]) for path in (log_dir / 'evaluations').glob('question_*')]
else:
    exist_question_ids = []
datasets = [dataset for dataset in datasets if (start_id <= dataset['id'] <= end_id) and (dataset['id'] not in exist_question_ids)]
print(f"Exist {len(exist_question_ids)}, Left {len(datasets)}")

# prompt
system_prompt = '''You are a medical assistant, please answer my medical questions. Give your final choice (capital option) closed in tag `<answer>X</answer>` after your analysis.
Your response should be as detailed as possible, but please do not use any subheadings.'''
user_prompt = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

Please analyze the question and give your answer. Give your analysis and final choice.''')
user_prompt_round = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

### Documents
{{documents}}

---

I will provide several assistant's previous answers and their scores (0-10) for your reference (higher scores indicate higher quality). But these previous answers may be incorrect, please analyze and then re-answer.

{{answers}}

---

The above are the question along with the assistant's previous answers, and some relevant documents. Please analyze and then re-answer. Please give your response towards **higher score**. Give your analysis and final answer.''')

system_prompt_query = '''\
### Role & Goal:
You are a medical expert. I will provide multiple different answers to the same question, which may be incorrect. Your task is to identify contradictions, ambiguities, and core dispute points among the answers, and extract key concept queries for further retrieval and verification.

### Processing Steps:
1. Analyze different answers and summarize the differences between them.
2. Extract keywords from these differences for retrieval.
3. Generate 1-4 precise queries for further search using **BM25** retriever.

### Output format:
ONLY give the queries:
[Query 1] xxx
[Query 2] xxx
(more queries)'''
user_prompt_query = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

---

I will provide several assistant's previous answers, which may be incorrect:

{{answers}}

---

Please analysis the differences among the answers and generate 4 queries for further search. Give your queries in the given format.''')

# evaluation
model = CustomLanguageModel(local_args.model_name, logger, base_url=local_args.base_url)
retriever = RetrievalService()
reranker = RerankerSyetem()
for dataset in tqdm(datasets, ncols=100):
    dataset['option_str'] = get_query(dataset)['option_str']
    for round_id in range(1, local_args.num_round + 1):
        if round_id == 1:
            prompt = user_prompt.render(question=dataset['question'], options=dataset['option_str'])
        else:
            prompt = user_prompt_round.render(question=dataset['question'], options=dataset['option_str'], answers='\n\n'.join(previous_answers), documents=documents)

        results = inference(system_prompt, prompt, model, n=local_args.num_workers)

        log_path = log_dir / f"evaluations/question_{dataset['id']}"
        log_path.mkdir(parents=True, exist_ok=True)
        round_results = []
        previous_answer_contents = []
        for answer_id, result in enumerate(results, start=1):
            previous_answer_contents.append(result[1])
            ans, is_correct = judger(result[1], dataset['answer'], single_answer=True)

            temp_result = OrderedDict()
            temp_result['round'] = round_id
            temp_result['answer_id'] = answer_id
            temp_result['answer'] = dataset['answer']
            temp_result['prediction'] = ans
            temp_result['is_correct'] = is_correct
            temp_result['prompt'] = prompt
            temp_result['reasoning'] = result[0]
            temp_result['response'] = result[1]
            temp_result['logprobs'] = result[2]['logprobs']
            temp_result['token_entropies'] = result[2]['token_entropies']
            # temp_result['documents'] = documents
            round_results.append(temp_result)

        previous_answers = [f"{i}. The assistant's previous answer is:\n" + answer for i, answer in enumerate(previous_answer_contents, start=1)]

        with open(log_path / f"round_{round_id}.jsonl", 'w', encoding='utf-8') as fp:
            for round_result in round_results:
                fp.write(json.dumps(round_result, ensure_ascii=False) + '\n')

        # If all answers are the same, then break the loop.
        if len(set([round_result['prediction'] for round_result in round_results])) == 1 or round_id == local_args.num_round:
            break

        previous_answers_query = previous_answers

        reason_content, query_content, _ = inference(system_prompt_query, user_prompt_query.render(question=dataset['question'], options=dataset['option_str'], answers='\n\n'.join(previous_answers_query)), model, enable_thinking=True, temperature=0.)[0]
        queries = re.findall(r"\[Query .*?\](.*?)$", query_content, re.MULTILINE)
        queries = list(set(map(lambda x: x.strip(), queries)))
        logger.info(f"Retrieve {len(queries)} queries.")
        if queries:
            documents = []
            total_doc_ids = []
            retrieve_func = partial(retriever.retrieve, total_k=32, top_k=2, combine_docs=False, use_reranker=True)
            with ThreadPoolExecutor(max_workers=len(queries)) as excutor:
                results = excutor.map(retrieve_func, queries)
            for result in results:
                docs, scores = result
                docs = [doc for doc in docs if doc['id'] not in total_doc_ids]
                documents.extend(docs)
                total_doc_ids.extend([doc['id'] for doc in docs])
            documents = combine_docs(documents, combine_sep='\n')
        else:
            documents = 'null'

        # rerank previous answers by their scores
        scores = reranker.rerank(query=dataset['question'] + '\nAnswer Choices: ' + dataset['option_str'].replace('\n', '   '), docs=[answer.split('<answer>')[0].replace('\n\n', ' ').replace('\n', ' ').strip() for answer in previous_answer_contents])
        arg_scores = np.argsort(scores)
        previous_answer_contents = [previous_answer_contents[i] for i in arg_scores]
        previous_answers = [f"{i}. The assistant's previous answer is (Score: {int(round(10 * score, 0))}):\n" + answer for i, (answer, score) in enumerate(zip(previous_answer_contents, sorted(scores)), start=1)]

# calculate accuracy
calculate_accuracy(log_dir / 'evaluations')
################################################################################