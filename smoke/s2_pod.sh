#!/usr/bin/env bash
# S2 on the GPU pod: install vLLM in its own venv and serve Qwen3.5-9B with the S1 adapter.
# Listens on 127.0.0.1 only; reach it from the laptop through an SSH tunnel.
# Usage: bash /root/smoke/s2_pod.sh [nightly]
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /root

if [ ! -d vllm-env ]; then
  uv venv vllm-env --python 3.12
  if [ "${1:-}" = "nightly" ]; then
    VIRTUAL_ENV=/root/vllm-env uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
  else
    VIRTUAL_ENV=/root/vllm-env uv pip install -U vllm --torch-backend=auto
  fi
fi
/root/vllm-env/bin/python -c "import vllm; print('vLLM', vllm.__version__)"

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec /root/vllm-env/bin/vllm serve Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 8000 \
  --served-model-name qwen-base \
  --dtype bfloat16 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 10}' \
  --enable-lora --max-lora-rank 16 --max-loras 4 \
  --lora-modules smoke-s1=/root/runs/s1/adapter \
  2>&1 | tee /root/s2-serve.log
