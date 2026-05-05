import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
import argparse
import asyncio
import faiss
import json
import torch
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from functools import lru_cache
from logging import Logger
from pathlib import Path
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from time import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

corpus_names = {
    'PubMed': ['pubmed'],
    'Textbooks': ['textbooks'],
    'StatPearls': ['statpearls'],
    'Wikipedia': ['wikipedia'],
    'MedText': ['textbooks', 'statpearls'],
    'MedCorp': ['pubmed', 'textbooks', 'statpearls', 'wikipedia'],
}
retriever_paths = {
    'BM25': ['bm25'],
    'MedCPT': ['ncbi/MedCPT-Query-Encoder'],
}
reranker_paths = {
    'Qwen3-Reranker-0.6B': 'Qwen/Qwen3-Reranker-0.6B',
    'MedCPT-Cross-Encoder': 'ncbi/MedCPT-Cross-Encoder',
}

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + ' ' + content.strip()
    else:
        return title.strip() + '. ' + content.strip()


class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling 'MEAN' to 'CLS'
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        '''
        Creates a simple Transformer + CLS Pooling model and returns the modules
        '''
        print(f'No sentence-transformers model found with name {model_name_or_path}. Creating a new one with CLS pooling.')
        token = kwargs.get('token', None)
        cache_folder = kwargs.get('cache_folder', None)
        revision = kwargs.get('revision', None)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        if 'token' in kwargs or 'cache_folder' in kwargs or 'revision' in kwargs or 'trust_remote_code' in kwargs:
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={'token': token, 'trust_remote_code': trust_remote_code, 'revision': revision},
                tokenizer_args={'token': token, 'trust_remote_code': trust_remote_code, 'revision': revision},
            )
        else:
            transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


def embed(chunk_dir: Path, index_dir: Path, model_path: str, **kwargs):
    save_dir = index_dir / 'embedding'
    save_dir.mkdir(exist_ok=True, parents=True)

    if 'contriever' in model_path.lower() or 'qwen3-embedding' in model_path.lower():
        model = SentenceTransformer(model_path, device=kwargs.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    else:
        model = CustomizeSentenceTransformer(model_path, device=kwargs.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model.eval()

    # create embeddings
    fpaths = sorted([fpath for fpath in chunk_dir.glob('*.jsonl')])
    with torch.no_grad():
        for fpath in tqdm(fpaths, ncols=100):
            save_path = save_dir / fpath.name.replace('.jsonl', '.npy')
            if save_path.exists():
                continue
            if open(fpath).read().strip() == '':
                continue

            texts = [json.loads(item) for item in open(fpath).read().strip().split('\n')]
            if 'contriever' in model_path.lower():
                texts = ['. '.join([item['title'], item['content']]).replace('..', '.').replace('?.', '?') for item in texts]
            elif 'medcpt' in model_path.lower():
                texts = [[item['title'], item['content']] for item in texts]
            else:
                texts = [concat(item['title'], item['content']) for item in texts]

            embed_chunks = model.encode(texts, batch_size=32, **kwargs)
            np.save(save_path, embed_chunks)

    return model.encode([''], **kwargs).shape[-1]

def construct_index(index_dir: Path, model_name: str, h_dim: int = 768):
    with open(index_dir / 'metadatas.jsonl', 'w') as fp:
        fp.write('')
    fp = open(index_dir / 'metadatas.jsonl', 'a+')

    if 'specter' in model_name.lower():
        index = faiss.IndexFlatL2(h_dim)
    else:
        index = faiss.IndexFlatIP(h_dim)

    for fpath in tqdm(sorted((index_dir / 'embedding').glob('*')), ncols=100):
        curr_embed = np.load(fpath)
        index.add(curr_embed)
        fp.write('\n'.join([json.dumps({'index': i, 'source': fpath.name.replace('.npy', '')}) for i in range(len(curr_embed))]) + '\n')

    fp.close()
    faiss.write_index(index, str((index_dir / 'faiss.index').absolute()))
    return index


class Retriever: 
    def __init__(self, logger: Logger, retriever_path: str = 'ncbi/MedCPT-Query-Encoder', corpus_name: str = 'textbooks', db_dir: str = './corpus', cache: bool = False, **kwargs):
        self.logger = logger
        self.cache = cache
        self.retriever_name = retriever_path.split('/')[-1]
        self.retriever_path = retriever_path
        self.corpus_name = corpus_name
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)
        self.chunk_dir = self.db_dir / f'{self.corpus_name}/chunk'  # directory of the corpus chunked dataset
        self.index_dir = self.db_dir / f'{self.corpus_name}/index/{self.retriever_name}'.replace('Query-Encoder', 'Article-Encoder')  # directory of embeddings and index
        # self.cache_path = self.db_dir / f'{self.corpus_name}/cache.jsonl'
        # self.cache_writer = open(self.cache_path, 'a+', encoding='utf-8')

        if self.cache:
            if (self.db_dir / f"{self.corpus_name}/id2text.json").exists():
                with open(self.db_dir / f"{self.corpus_name}/id2text.json", 'r', encoding='utf-8') as fp:
                    self.dict = json.load(fp)
            else:
                self.dict = {}
                for fname in tqdm(sorted(self.chunk_dir.glob('*.jsonl'))):
                    if open(fname).read().strip() == "":
                        continue
                    for i, line in enumerate(open(fname).read().strip().split('\n')):
                        item = json.loads(line)
                        _ = item.pop('contents', None)
                        # assert item['id'] not in self.dict
                        self.dict[item['id']] = item
                with open(self.db_dir / f"{self.corpus_name}/id2text.json", 'w', encoding='utf-8') as fp:
                    json.dump(self.dict, fp)
        else:
            self.dict = None

        if "bm25" in self.retriever_name.lower():
            from pyserini.search.lucene import LuceneSearcher
            self.metadatas = None
            self.embedding_function = None
            if self.index_dir.exists():
                cache_index_dir = '/cache/corpus/' + str(self.index_dir.absolute()).split('corpus/')[-1]
                self.index = LuceneSearcher(cache_index_dir)
            else:
                import os
                os.system("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(str(self.chunk_dir.absolute()), str(self.index_dir.absolute())))
                self.index = LuceneSearcher(str(self.index_dir.absolute()))
            print(f"Successful loading index {str(self.index_dir.absolute())}.")
            self.index.set_bm25(k1=0.9, b=0.4)
        else:
            # read index and metadata
            if (self.index_dir / 'faiss.index').exists():
                self.index = faiss.read_index(str((self.index_dir / 'faiss.index').absolute()))
                if 'collected' in self.corpus_name.lower():
                    self.corpus = {}
                    for corpus in self.chunk_dir.glob('*.jsonl'):
                        with open(corpus, 'r', encoding='utf-8') as fp:
                            self.corpus[corpus.name.rsplit('.', maxsplit=1)[0]] = fp.read().strip().split('\n')
                #     gpu_resources = faiss.StandardGpuResources()
                #     # gpu_resources.setDefaultNullStreamAllDevices()
                #     # gpu_resources.setDefaultStream(0, True)
                #     self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                #     self.logger.info('Load index from cpu to gpu')
                self.metadatas = [json.loads(line) for line in open(self.index_dir / 'metadatas.jsonl').read().strip().split('\n')]
            # create index and metadata
            else:
                self.logger.info(f'[In progress] Embedding the {self.corpus_name} corpus with the {self.retriever_name.replace("Query-Encoder", "Article-Encoder")} retriever...')
                h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_path=self.retriever_path.replace('Query-Encoder', 'Article-Encoder'), **kwargs)
                self.logger.info(f'[In progress] Embedding finished! The dimension of the embeddings is {h_dim}.')

                self.logger.info('[In progress] Corpus indexing started!')
                self.index = construct_index(index_dir=self.index_dir, model_name=self.retriever_name.replace('Query-Encoder', 'Article-Encoder'), h_dim=h_dim)
                self.logger.info('[In progress] Corpus indexing finished!')
                self.metadatas = [json.loads(line) for line in open(self.index_dir / 'metadatas.jsonl').read().strip().split('\n')]

            if 'qwen3-embedding' in self.retriever_name.lower() or 'contriever' in self.retriever_name.lower():
                self.embedding_function = SentenceTransformer(self.retriever_path, device=kwargs.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
            else:
                self.embedding_function = CustomizeSentenceTransformer(self.retriever_path, device=kwargs.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

            self.embedding_function.eval()

    def invoke(self, questions: List[str], k: int = 32, **kwargs) -> Tuple[List[Dict], List[float]]:
        '''
        Return the top k documents and their scores for the given question.
        '''
        if 'qwen3-embedding' in self.retriever_name.lower():
            kwargs['prompt_name'] = 'query'

        if k == 0:
            return [], []

        if "bm25" in self.retriever_name.lower():
            docs = []
            scores = []
            query_ids = list(map(str, range(len(questions))))
            hits = self.index.batch_search(questions, query_ids, k=k, threads=16)
            for query_id in query_ids:
                indices = [{'source': '_'.join(h.docid.split('_')[:-1]), 'index': eval(h.docid.split('_')[-1])} for h in hits[query_id]]
                docs.append(self._idx2json(indices))
                scores.append([h.score for h in hits[query_id]])

        else:
            # start = time()
            with torch.no_grad():
                query_embed = self.embedding_function.encode(questions, show_progress_bar=False, **kwargs)
            # end = time()
            # self.logger.info(f"Embed time cost: {end - start:.2f}")

            # start = time()
            scores, indices = self.index.search(query_embed, k=k)  # [scores, indices]
            scores = [score.tolist() for score in scores]
            # end = time()
            # self.logger.info(f"Faiss search time cost: {end - start:.2f}")

            indices = [self.metadatas[i] for i in np.array(indices).flatten()]
            docs = self._idx2json(indices)
            # for doc in docs:
            #     self.cache_writer.write(json.dumps(doc, ensure_ascii=False) + '\n')
            #     self.cache_writer.flush()
            docs = np.array(docs).reshape(-1, k)

        return docs, scores

    def _idx2json(self, indices: List[Dict]) -> List[Dict]:  # return List of Dict of str
        '''
        Input: List of Dict( {'source': str, 'index': int} )
        Output: List of Dict of str
        '''
        if self.cache:
            ids = [f"{i['source']}_{i['index']}" for i in indices]
            return [self.dict[id_] for id_ in ids]

        if 'collected' in str(self.chunk_dir).lower():
            return [json.loads(self.corpus[i['source']][i['index']]) for i in indices]

        return [json.loads(open(self.chunk_dir / f'{i["source"]}.jsonl').read().strip().split('\n')[i['index']]) for i in indices]


class RetrievalSystem:
    def __init__(self, logger: Logger, retriever_name: str = 'MedCPT', corpus_name: str = 'Textbooks', reranker_name: Optional[str] = None, db_dir: str = './corpus', cache: bool = False, device: str = 'cuda'):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.reranker_name = reranker_name
        self.logger = logger
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_paths
        assert len(retriever_paths[self.retriever_name]) == 1, 'Only one retriever is supported for now'
        self.retrievers = []
        for retriever_path in retriever_paths[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(logger, retriever_path, corpus, db_dir, cache=cache, device=device))

        self.reranker_tok = None
        self.reranker = None
        if self.reranker_name:
            if 'qwen3-reranker' in self.reranker_name.lower():
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.reranker_tok = AutoTokenizer.from_pretrained(reranker_paths[self.reranker_name], padding_side='left', use_fast=False)
                self.reranker = AutoModelForCausalLM.from_pretrained(reranker_paths[self.reranker_name], device_map='auto', max_memory={0: '10GB', 'cpu': '50GB'}).eval()
                self.token_false_id = self.reranker_tok.convert_tokens_to_ids("no")
                self.token_true_id = self.reranker_tok.convert_tokens_to_ids("yes")
            elif 'medcpt' in self.reranker_name.lower():
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.reranker_tok = AutoTokenizer.from_pretrained(reranker_paths[self.reranker_name])
                self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_paths[self.reranker_name], device_map='auto')
            else:
                raise NotImplementedError

    # @lru_cache(maxsize=1024)
    def retrieve(self, queries: List[str], total_k: int = 32, top_ks: Optional[List[int]] = None, questions: List = None, use_reranker: bool = False, min_score: Optional[float] = None):
        '''
        Given query, return the relevant snippets from the corpus.
        If use top_k, more documents can be retrieved and only the top_k documents with the highest scores will be returned.
        `question` is the original question, which can use to rerank documents, if given.
        '''
        if top_ks is None:
            top_ks = [total_k] * len(queries)
        else:
            assert len(top_ks) == len(queries)

        texts = []  # num_retrievers, num_corpus, num_queries, num_documents
        scores = []

        for i in range(len(retriever_paths[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].invoke(queries, k=total_k)
                texts[-1].append(t)
                scores[-1].append(s)
        texts, scores = self._merge(texts, scores)

        # rerank
        if self.reranker_name and use_reranker:
            origin_num_docs = len(texts[0])
            start_time = time()
            if questions:
                questions = [question or query for question, query in zip(questions, queries)]
            else:
                questions = queries
            reranked_texts, reranked_scores = [], []
            for query_id in range(len(texts)):
                texts_, scores_ = self._rerank(questions[query_id], texts[query_id], top_ks[query_id])
                reranked_texts.append(texts_)
                reranked_scores.append(scores_)
            texts, scores = reranked_texts, reranked_scores
            end_time = time()
            self.logger.info(f"Rerank from {origin_num_docs} documents, cost time: {end_time - start_time:.2f}")

        if min_score:
            texts = [[texts[i][j] for j in range(len(texts[i])) if scores[i][j] >= min_score] for i in range(len(texts))]
            scores = [[scores[i][j] for j in range(len(scores[i])) if scores[i][j] >= min_score] for i in range(len(scores))]

        for text, query in zip(texts, queries):
            self.logger.info(f'Retrieved {len(text)} documents for query: {query}')
        return texts, scores

    @torch.no_grad()
    def _rerank(self, question: str, docs: List[str], top_k: int):
        torch.cuda.empty_cache()
        if len(docs) == 0:
            return [], []

        if 'qwen3-reranker' in self.reranker_name.lower():
            def format_instruction(instruction: str, query: str, doc: str) -> str:
                return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)

            def process_inputs(pairs: List[str]):
                inputs = self.reranker_tok(
                    pairs, padding=False, truncation='longest_first',
                    return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
                )
                for i, ele in enumerate(inputs['input_ids']):
                    inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
                inputs = self.reranker_tok.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
                for key in inputs:
                    inputs[key] = inputs[key].to(self.reranker.device)
                return inputs

            def compute_logits(inputs, **kwargs) -> List[float]:
                batch_scores = self.reranker(**inputs).logits[:, -1, :]
                true_vector = batch_scores[:, self.token_true_id]
                false_vector = batch_scores[:, self.token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
                return scores

            max_length = 4096
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = self.reranker_tok.encode(prefix, add_special_tokens=False)
            suffix_tokens = self.reranker_tok.encode(suffix, add_special_tokens=False)
            task = 'Given a document search query, retrieve relevant passages that answer the query'
            queries = [question] * len(docs)
            pairs = [format_instruction(task, query, doc['content']) for query, doc in zip(queries, docs)]

            # Tokenize the input texts
            processed = 0
            batch_size = 4
            total_scores = []
            while processed < len(pairs):
                inputs = process_inputs(pairs[processed:processed+batch_size])
                scores = compute_logits(inputs)
                total_scores.extend(scores)
                processed += batch_size

            arg_scores = np.argsort(total_scores)[::-1][:top_k]
            docs = [docs[i] for i in arg_scores]
            total_scores = [total_scores[i] for i in arg_scores]
            return docs, total_scores

        elif 'medcpt' in self.reranker_name.lower():
            pairs = [[question, doc['content']] for doc in docs]
            encoded = self.reranker_tok(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512)
            for key in encoded:
                encoded[key] = encoded[key].to(self.reranker.device)
            logits = self.reranker(**encoded).logits.squeeze(dim=1).cpu().numpy().tolist()
            arg_sort = np.argsort(logits)[::-1][:top_k]
            docs = [docs[i] for i in arg_sort]
            scores = [logits[i] for i in arg_sort]
            return docs, scores

        else:
            raise NotImplementedError

    def _merge(self, texts: List[List[List[Dict]]], scores: List[List[List[float]]]) -> Tuple[List[List[Dict]], List[List[float]]]:
        '''
        Merge the texts and scores from different retrievers.
        '''
        text_retrievers = [[] for _ in range(len(texts[0][0]))]
        score_retrievers = [[] for _ in range(len(texts[0][0]))]
        for i in range(len(retriever_paths[self.retriever_name])):
            texts_all, scores_all = None, None
            for j in range(len(corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    for doc_id in range(len(texts[i][j])):
                        texts_all[doc_id] = texts_all[doc_id] + texts[i][j][doc_id]
                        scores_all[doc_id] = scores_all[doc_id] + scores[i][j][doc_id]

            sorted_index = []
            if 'specter' in retriever_paths[self.retriever_name][i].lower():
                for scores in scores_all:
                    sorted_index.append(np.array(scores).argsort(axis=-1))
            else:
                for scores in scores_all:
                    sorted_index.append(np.array(scores).argsort(axis=-1)[..., ::-1])  # IP index need to sort in descending order

            for query_id in range(len(text_retrievers)):
                text_retrievers[query_id].extend([texts_all[query_id][doc_id] for doc_id in sorted_index[query_id]])
                score_retrievers[query_id].extend([scores_all[query_id][doc_id] for doc_id in sorted_index[query_id]])

        return text_retrievers, score_retrievers

    def combine_docs(self, docs: List[Dict], combine_sep: str, num_retrieved_docs: int = 0) -> str:
        return combine_sep.join([f'Document [{i+num_retrieved_docs}] (Title: {doc["title"]}) {doc["content"]}' for i, doc in enumerate(docs)])


class QueryRequest(BaseModel):
    query: str
    top_k: int = 1
    total_k: Optional[int] = 1
    combine_docs: bool = False
    combine_sep: str = '   '
    num_retrieved_docs: int = 0
    # reranker
    question: Optional[str] = None
    use_reranker: bool = False
    min_score: Optional[float] = None


retriever = None
def init_retriever():
    global retriever, logger
    if retriever is not None:
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever', type=str, default='BM25')
    parser.add_argument('--reranker', type=str, default='MedCPT-Cross-Encoder')
    parser.add_argument('--corpus-name', type=str, default='Textbooks')
    parser.add_argument('--cuda', '-c', type=int, default=0, help='Use CUDA device (default: 0).')
    parser.add_argument('--port', '-p', type=int, default=8990, help='Port')
    parser.add_argument('--num-workers', '-n', type=int, default=None, help='Number of workers for data loading (default: None).')
    local_args, _ = parser.parse_known_args()
    device = f'cuda:{local_args.cuda}' if torch.cuda.is_available() else 'cpu'

    logger = logging.getLogger(__name__)
    retriever = RetrievalSystem(logger, local_args.retriever, local_args.corpus_name, local_args.reranker, cache=True, device=device)
    logger.info(f"Retriever initialized. Using {local_args.retriever} retriever, {local_args.reranker} reranker, {local_args.corpus_name} corpus.")
    return local_args


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_retriever()
    yield

app = FastAPI(lifespan=lifespan)

pending_queries: List[Dict[str, Any]] = []
pending_futures: List[asyncio.Future] = []

@app.post('/retriever')
async def get_documents(request: QueryRequest):
    future = asyncio.Future()
    pending_queries.append({'query': request.query, 'top_k': request.top_k, 'question': request.question, 'num_retrieved_docs': request.num_retrieved_docs, 'combine_docs': request.combine_docs, 'combine_sep': request.combine_sep})
    pending_futures.append(future)

    if len(pending_queries) == 1:
        asyncio.create_task(process_batch(request.total_k, request.use_reranker, request.min_score))

    return await future

async def process_batch(total_k: int, use_reranker: bool, min_score: float):
    global logger
    await asyncio.sleep(0.5)

    if not pending_queries:
        return

    current_queries = pending_queries[:]
    current_futures = pending_futures[:]
    assert len(current_queries) == len(current_futures)

    pending_queries.clear()
    pending_futures.clear()

    start_time = time()
    docs, scores = await asyncio.get_running_loop().run_in_executor(None, retriever.retrieve, [query['query'] for query in current_queries], total_k, [query['top_k'] for query in current_queries], [query['question'] for query in current_queries], use_reranker, min_score)

    docs = [retriever.combine_docs(doc, query['combine_sep'], query['num_retrieved_docs']) if query['combine_docs'] else doc for doc, query in zip(docs, current_queries)]
    stop_time = time()
    logger.info(f"Retrieval time cost: {stop_time - start_time:.2f}")

    for future, doc, score in zip(current_futures, docs, scores):
        if not future.cancelled():
            future.set_result({'documents': doc, 'scores': score})

if __name__ == '__main__':
    local_args = init_retriever()
    uvicorn.run('microservice.RetrievalSystem:app', host='0.0.0.0', port=local_args.port, log_level='info', loop='uvloop', workers=local_args.num_workers)