#!/usr/bin/env bash
# Start vLLM (base model, optionally with LoRA adapters) on 127.0.0.1:8000 and wait until it answers.
# Usage: pod/serve.sh [name=adapter_dir ...]     e.g. pod/serve.sh qwen-v1=$HOME/runs/v1/adapter
# Stop with: pod/serve.sh --stop
set -euo pipefail
LOG="$HOME/vllm.log"
if [ "${1:-}" = "--stop" ]; then
  pkill -f "vllm-env/bin/vllm serve" || true
  for _ in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null || exit 0; sleep 2; done
  exit 0
fi
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_USE_FLASHINFER_SAMPLER=0   # its JIT needs nvcc, absent on the pod image; we decode greedily
export VLLM_NO_USAGE_STATS=1
LORA_ARGS=()
if [ "$#" -gt 0 ]; then LORA_ARGS=(--enable-lora --max-lora-rank 16 --max-loras 4 --lora-modules "$@"); fi
nohup "$HOME/vllm-env/bin/vllm" serve Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 8000 --served-model-name qwen-base \
  --dtype bfloat16 --max-model-len 16384 --limit-mm-per-prompt '{"image": 10}' \
  "${LORA_ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
for _ in $(seq 1 180); do
  if curl -s -m 3 http://127.0.0.1:8000/v1/models | grep -q qwen-base; then echo "vLLM ready"; exit 0; fi
  if grep -a -q -E "Engine core initialization failed|Traceback" "$LOG"; then echo "vLLM failed; see $LOG"; tail -20 "$LOG"; exit 1; fi
  sleep 5
done
echo "vLLM not ready after 15 min; see $LOG"; exit 1
