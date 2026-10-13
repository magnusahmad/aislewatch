#!/usr/bin/env bash
# Start vLLM on 127.0.0.1:8000, wait until it answers, and VERIFY every expected model name is served.
# Usage: pod/serve.sh <hf_model> <served_name> [lora_name=adapter_dir ...]
#   e.g. pod/serve.sh Qwen/Qwen3.5-4B qwen4b-base qwen4b-v1=$HOME/runs/v1-4b/adapter
# Stop:  pod/serve.sh --stop   (waits until the port is free)
set -euo pipefail
LOG="$HOME/vllm.log"
if [ "${1:-}" = "--stop" ]; then
  pkill -f "vllm-env/bin/vllm serve" || true
  for _ in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null || exit 0; sleep 2; done
  echo "vLLM did not stop"; exit 1
fi
MODEL="$1"; SERVED="$2"; shift 2
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_USE_FLASHINFER_SAMPLER=0   # its JIT needs nvcc, absent on the pod image; we decode greedily
export VLLM_NO_USAGE_STATS=1
LORA_ARGS=(); EXPECT=("$SERVED")
if [ "$#" -gt 0 ]; then
  LORA_ARGS=(--enable-lora --max-lora-rank 16 --max-loras 4 --lora-modules "$@")
  for spec in "$@"; do EXPECT+=("${spec%%=*}"); done
fi
nohup "$HOME/vllm-env/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8000 --served-model-name "$SERVED" \
  --dtype bfloat16 --max-model-len 16384 --limit-mm-per-prompt '{"image": 10}' \
  "${LORA_ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
for _ in $(seq 1 180); do
  if MODELS=$(curl -s -m 3 http://127.0.0.1:8000/v1/models) && echo "$MODELS" | grep -q "\"$SERVED\""; then
    for name in "${EXPECT[@]}"; do
      echo "$MODELS" | grep -q "\"$name\"" || { echo "CHECK FAILED: vLLM up but '$name' not served"; exit 1; }
    done
    echo "vLLM ready: ${EXPECT[*]}"; exit 0
  fi
  if grep -a -q -E "Engine core initialization failed|Traceback" "$LOG"; then echo "vLLM failed; see $LOG"; tail -20 "$LOG"; exit 1; fi
  sleep 5
done
echo "vLLM not ready after 15 min; see $LOG"; exit 1
