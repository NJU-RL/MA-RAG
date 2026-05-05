export CUDA_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --api-key "dummy" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --port 8000 \
    --host 0.0.0.0 \