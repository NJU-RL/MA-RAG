# From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG

<p align="center">
<a href="https://arxiv.org/abs/2603.03292"><img src="https://img.shields.io/badge/ArXiv-2603.03292-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
<a href="https://github.com/NJU-RL/MA-RAG/stargazers"><img src="https://img.shields.io/github/stars/NJU-RL/MA-RAG" alt="GitHub stars"></a>
</p>

**Wenhao Wu<sup>1,2</sup>, Zhentao Tang<sup>2</sup>, Yafu Li<sup>3</sup>, Shixiong Kai<sup>2</sup>, Mingxuan Yuan<sup>2</sup>, Zhenhong Sun<sup>4</sup>, Chunlin Chen<sup>1</sup>, Zhi Wang<sup>1</sup>**

<sup>1</sup>Nanjing University &nbsp; <sup>2</sup>Huawei Noah's Ark Lab &nbsp; <sup>3</sup>The Chinese University of Hong Kong &nbsp; <sup>4</sup>Australian National University

## Overview
Official codebase for *From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG*

![MA-RAG](./figs/method.png)

## Start retrieval service
We borrow the retrieval service and use the `MedCorp` corpus provided by [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG). We sincerely appreciate their impressive work!

Download the corpus following [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) and place it under `YOUR_PROJECT_PATH/corpus`.

To start retrieval service, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python microservice/RetrievalSystem.py --retriever BM25 --reranker MedCPT-Cross-Encoder --corpus-name MedCorp --cuda 0 --port 8990
```

## Start evaluator service [Optional]
To start extrinsic evaluator service, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python microservice/EvaluatorSystem.py --model-name-or-path YOUR_CHECKPOINT_PATH --port 8880
```

## Run evaluation
Add the `API_KEY` and `BASE_URL` in `.env` file, and then run the following command:
```bash
# MA-RAG (intrinsic entropy)
python ma_rag_entropy.py --dataset-path ./datasets/MedMCQA.json --model-name Qwen3-8B --exp 0 --num-workers 8 --num-round 8
# MA-RAG (extrinsic evaluator)
python ma_rag_evaluator.py --dataset-path ./datasets/MedMCQA.json --model-name Qwen3-8B --exp 0 --num-workers 8 --num-round 8
```

## Main results: Comparison to different RAG frameworks and test-time scaling methods
![Main results](./figs/main_results.png)

## Citation
If you find our paper useful, please consider starring this repository and cite it:
```bibtex
@inproceedings{wu2026marag,
    title={From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic {RAG}},
    author={Wenhao Wu and Zhentao Tang and Yafu Li and Shixiong Kai and Mingxuan Yuan and Zhenhong Sun and Chunlin Chen and Zhi Wang},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=S7lpdz7NAW}
}
```