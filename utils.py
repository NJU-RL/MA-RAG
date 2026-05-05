from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

import json
import os
import random
import re
import requests
import shutil
import numpy as np
from collections import Counter
from functools import lru_cache
from pathlib import Path
from openai.types.chat.chat_completion import ChatCompletion
from scipy.stats import entropy
from time import sleep
from typing import Dict, Any, List, Tuple, Optional
from microservice import CustomLanguageModel


def copy_files(to_path: str, folders: List = [], files: List = [], parent: bool = True) -> None:
    path = Path(to_path) / 'Codes' if parent else Path(to_path)
    path.mkdir(parents=True, exist_ok=True)

    # copy files
    for folder in folders:
        destiantion = path / folder
        if destiantion.exists():
            shutil.rmtree(destiantion)
        shutil.copytree(folder, destiantion, ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))

    for file in files:
        shutil.copy(file, path)

def get_query(data: Dict[str, Any]):
    temp_data = {}
    # data["exam_type"] = data["exam_type"]
    # data["exam_class"] = data["exam_class"]
    # data["exam_subject"] = data["exam_subject"]
    temp_data['id'] = data['id']
    # temp_data["question_type"] = data["question_type"]
    temp_data["question"] = data["question"]
    temp_data['option'] = data['option']
    temp_data["option_str"] = "\n".join(
        [f"{k}. {v}" for k, v in data["option"].items() if len(v) > 0 and v!=" "]
    )
    return temp_data

def judger(answer: str, target: str, single_answer: bool = True) -> Tuple[str, bool]:
    try:
        if '<answer>' in answer and '</answer>' in answer:
            ans = re.findall('([A-Z])', re.findall(r'<answer>(.*?)</answer>', answer)[-1])
        else:
            answer = re.sub(r'[（）。、，：！？\[\]]', ' ', answer)
            for split_str in ['最终答案', '答案是', '答案为', '答案选', 'Final Answer', 'Answer:']:
                if split_str in answer:
                    ans = re.findall(r'\b([A-Z])\b', answer.split(split_str)[-1])[::-1]
                    break
            else:
                ans = re.findall(r'\b([A-Z])\b', answer)

        ans = sorted(ans)
        if single_answer:
            ans = ans[-1]
        else:
            ans = ''.join(ans)
    except:
        ans = ''

    return ans, ans == target

def process_response(response: ChatCompletion) -> List[Dict[str, Any]]:
    responses = []
    for choice in response.choices:
        tokens = [cont.token for cont in choice.logprobs.content]
        if 'token_id:' in tokens[0]:
            try:
                think_rindex = len(tokens) - tokens[::-1].index('token_id:151668')  # </think> token
            except ValueError:
                think_rindex = 0
        else:
            try:
                think_rindex = len(tokens) - tokens[::-1].index('</think>')  # </think> token
            except ValueError:
                think_rindex = 0
        text = choice.message.content
        token_entropies = [entropy(np.exp([top_logprobs.logprob for top_logprobs in cont.top_logprobs])) for cont in choice.logprobs.content[think_rindex:]]
        logprobs = [cont.logprob for cont in choice.logprobs.content[think_rindex:]]
        # entropies = [-cont.logprob * math.exp(cont.logprob) for cont in choice.logprobs.content]
        # top_logprobs = [[top_logprobs.logprob for top_logprobs in cont.top_logprobs] for cont in choice.logprobs.content]
        responses.append({
            'text': text,
            # 'entropies': entropies,
            'token_entropies': token_entropies,
            'logprobs': logprobs,
            # 'top_logprobs': top_logprobs,
            # 'tokens': tokens,
        })

    return responses

def format_check(content: str) -> bool:
    try:
        content = json.loads(content)
        keys = ['analysis', 'core', 'answer']
        for key in keys:
            if key not in content:
                return False
        return True
    except:
        return False

def inference(system: str, prompt: str, model: CustomLanguageModel, enable_thinking: bool = False, temperature: float = 0.7, check_format: bool = False, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': prompt},
    ]
    max_new_tokens = 8192 if enable_thinking else 2048
    return_infos = []
    while True:
        responses = process_response(model.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, top_logprobs=20, **kwargs))
        try:
            for response in responses:
                if enable_thinking:
                    reason_content, content = response['text'].rsplit('</think>', maxsplit=1)
                    if not check_format or format_check(content.strip()):
                        return_infos.append([reason_content.split('<think>', maxsplit=1)[1].strip(), content.strip(), response])
                else:
                    content = response['text']
                    if not check_format or format_check(content.strip()):
                        return_infos.append(['', content.strip(), response])
            return return_infos
        except Exception as e:
            print(f"Error processing response: {e}. Retrying...")

def combine_docs(docs: List[Dict], combine_sep: str = '\n'):
    return combine_sep.join([f'{doc["content"]}' for i, doc in enumerate(docs)])

def calculate_accuracy(result_dir: Path):
    result_dirs = sorted(list(result_dir.glob('*')), key=lambda x: int(str(x).split('_')[-1]))
    print(f"Total {len(result_dirs)} questions.")

    total_accuracy = 0
    for question_result_dir in result_dirs:
        round_result = sorted(list(question_result_dir.glob('*.jsonl')), key=lambda x: int(x.stem.split('_')[-1]))[-1]
        with open(round_result, 'r', encoding='utf-8') as fp:
            results = [json.loads(line) for line in fp]

        most_common_preds = Counter([result['prediction'] for result in results]).most_common(2)
        if most_common_preds[0][0] == results[0]['answer']:
            if len(most_common_preds) == 1 or most_common_preds[0][1] > most_common_preds[1][1]:
                total_accuracy += 1

    print(f"Accuracy: {total_accuracy / len(result_dirs):.4f}")

class RetrievalService:
    def __init__(self):
        self.host = os.getenv('RETRIEVER_HOST')

    @lru_cache(maxsize=8192)
    def retrieve(self, query: str, total_k: int, top_k: Optional[int] = None, num_retrieved_docs: int = 0, combine_docs: bool = False, combine_sep: str = '   ', question: Optional[str] = None, use_reranker: bool = False, min_score: Optional[float] = None):
        request = {'query': query, 'total_k': total_k, 'top_k': top_k, 'combine_docs': combine_docs, 'combine_sep': combine_sep, 'num_retrieved_docs': num_retrieved_docs, 'question': question, 'use_reranker': use_reranker, 'min_score': min_score}
        for _ in range(10):
            try:
                response = requests.post(f'{self.host}/retriever', json=request, timeout=200).json()
                break
            except Exception as e:
                print(e)
                sleep(random.uniform(0.0, 2.0))
        return response['documents'], response['scores']

    def combine_docs(self, docs: List[Dict], combine_sep: str = '\n', num_retrieved_docs: int = 0):
        return combine_sep.join([f'Document [{i+num_retrieved_docs}] (Title: {doc["title"]}) {doc["content"]}' for i, doc in enumerate(docs)])


class RerankerSyetem:
    def __init__(self):
        self.host = os.getenv('RERANKER_HOST')

    def rerank(self, query: str, docs: List[str]):
        params = {'query': query, 'docs': docs}
        response = requests.post(f'{self.host}/reranker', json=params).json()
        return response['scores']