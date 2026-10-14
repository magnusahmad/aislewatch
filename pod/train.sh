#!/usr/bin/env bash
# LoRA SFT with prime-rl, with checks that the intended run actually happened:
#   resolved config matches what we asked for; LoRA found target modules; loss decreased; adapter written.
# Usage: pod/train.sh <hf_model> <dataset> <run_dir_name> [epochs=2] [lr=1e-4] [batch=8]
# Writes ~/runs/<run_dir_name>/adapter and ~/runs/<run_dir_name>.train_report.json
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="$1"; DATASET="$2"; RUN="$3"; EPOCHS="${4:-2}"; LR="${5:-1e-4}"; BATCH="${6:-8}"
DATA_DIR="$HOME/aislewatch/data/datasets/$DATASET"
mkdir -p "$DATA_DIR/sft-train" && ln -sf "$DATA_DIR/train.parquet" "$DATA_DIR/sft-train/train.parquet"
ROWS=$(cd "$HOME/aislewatch" && uv run python -c "import pyarrow.parquet as pq; print(pq.read_metadata('$DATA_DIR/train.parquet').num_rows)")
STEPS=$(( (ROWS * EPOCHS + BATCH - 1) / BATCH ))
WARMUP=$(( STEPS / 15 > 1 ? STEPS / 15 : 1 ))
CFG="$HOME/runs/$RUN.toml"; mkdir -p "$HOME/runs"
# Tied-embedding models (Qwen3.5-4B) train with a zero LM head in prime-rl (loss = ln(vocab), grad norm 0);
# train on an untied copy instead. The adapter serves unchanged on the original model.
HF_MODEL="$MODEL"
MODEL=$(cd "$HOME/prime-rl" && uv run python "$HOME/aislewatch/pod/untie.py" "$HF_MODEL" "$HOME/models/untied-${HF_MODEL##*/}" | tail -1)
[ -n "$MODEL" ] || { echo "CHECK FAILED: untie.py produced no model path"; exit 1; }
cat > "$CFG" <<EOF
max_steps = $STEPS

[model]
name = "$MODEL"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[model.vlm]
vision_encoder_attr = "model.visual"
language_model_attr = "model.language_model"
freeze_vision_encoder = true

[model.lora]
rank = 16
alpha = 32
dropout = 0.0

[model.ac]

[renderer]
name = "qwen3.5"
enable_thinking = false

[data]
type = "sft"
name = "$DATA_DIR/sft-train"
batch_size = $BATCH
micro_batch_size = 1
seq_len = 16384

[data.loss_mask]
assistant = true

[optim]
lr = $LR

[scheduler]
type = "cosine"
warmup_steps = $WARMUP

[ckpt]
EOF
echo "train $RUN: model=$MODEL rows=$ROWS epochs=$EPOCHS batch=$BATCH steps=$STEPS lr=$LR warmup=$WARMUP"
cd "$HOME/prime-rl"

# 1. Dry run: the resolved config must match what we asked for.
uv run sft @ "$CFG" --output-dir "$HOME/runs" --run.name "$RUN-dry" --clean --dry-run > "$HOME/runs/$RUN.dryrun.log" 2>&1
RESOLVED=$(find "$HOME/runs/$RUN-dry" -name sft.json | head -1)
[ -n "$RESOLVED" ] || { echo "CHECK FAILED: dry run wrote no sft.json"; exit 1; }
uv run python - "$RESOLVED" "$MODEL" "$STEPS" "$LR" "$BATCH" <<'PY'
import json, sys
path, model, steps, lr, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
c = json.load(open(path))
checks = {"model": (c.get("model", {}).get("name"), model), "max_steps": (c.get("max_steps"), steps),
          "lr": (c.get("optim", {}).get("lr"), lr), "batch_size": (c.get("data", {}).get("batch_size"), batch),
          "lora_rank": ((c.get("model", {}).get("lora") or {}).get("rank"), 16),
          "renderer": ((c.get("renderer") or {}).get("name"), "qwen3.5"),
          "enable_thinking": ((c.get("renderer") or {}).get("enable_thinking"), False),
          "scheduler": ((c.get("scheduler") or {}).get("type"), "cosine")}
bad = {k: v for k, v in checks.items() if v[0] is not None and v[0] != v[1]}
missing = [k for k, v in checks.items() if v[0] is None]
if checks["model"][0] is None:
    bad["model"] = checks["model"]  # the model name must be findable
print("resolved config check:", "OK" if not bad else f"MISMATCH {bad}", f"(not found, unchecked: {missing})" if missing else "")
sys.exit(1 if bad else 0)
PY

# 2. Wait for an empty GPU: vLLM's engine can still hold memory for a while after its port closes, and a
#    trainer started into that dies at model load (Oct 14, v2: exit 1 after 28 s right after the base eval).
for _ in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-99999}" -lt 2000 ] && ! pgrep -f "vllm serve|EngineCore" > /dev/null && break
  sleep 5
done
echo "GPU before training: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
[ "${USED:-99999}" -lt 2000 ] || { echo "CHECK FAILED: GPU still holds ${USED} MiB after 5 min"; nvidia-smi; exit 1; }

# 3. Train, with an early guard: after 3 logged steps, a loss pinned at ln(vocab) (all-zero logits) or an
#    all-zero grad norm means nothing is learning -> kill it now instead of paying for the remaining steps.
uv run sft @ "$CFG" --output-dir "$HOME/runs" --run.name "$RUN" --clean > "$HOME/runs/$RUN.log" 2>&1 &
SFT_PID=$!
GUARD_OK=0
while kill -0 "$SFT_PID" 2>/dev/null; do
  TL=$(ls "$HOME/runs/$RUN/logs/"*/trainer.log 2>/dev/null | head -1 || true)   # absent until the trainer starts
  if [ "$GUARD_OK" = 0 ] && [ -n "$TL" ]; then
    VERDICT=$(python3 - "$TL" <<'PY'
import math, re, sys
t = open(sys.argv[1], errors="ignore").read()
s = re.findall(r"Step (\d+) \|[^|]*\| Loss ([0-9.]+) \| Grad\. Norm ([0-9.]+)", t)
if len(s) < 3: print("wait"); sys.exit()
loss = [float(x[1]) for x in s[:3]]; gn = [float(x[2]) for x in s[:3]]
uniform = math.log(248320)
if all(abs(l - uniform) < 0.02 for l in loss): print(f"DEAD loss pinned at ln(vocab)={uniform:.4f}: {loss}")
elif all(g == 0 for g in gn): print(f"DEAD grad norm 0 for 3 steps: {gn}")
else: print(f"ok loss {loss} grad norm {gn}")
PY
)
    case "$VERDICT" in
      DEAD*) echo "CHECK FAILED: not learning: ${VERDICT#DEAD }"; pkill -P "$SFT_PID"; kill "$SFT_PID"; pkill -f "prime_rl.trainer" || true; exit 1 ;;
      ok*) echo "learning check: $VERDICT"; GUARD_OK=1 ;;
    esac
  fi
  sleep 15
done
wait "$SFT_PID" || {
  echo "CHECK FAILED: sft exited non-zero"; tail -30 "$HOME/runs/$RUN.log"
  for f in $(ls "$HOME/runs/$RUN/logs/"*/trainer.log "$HOME/runs/$RUN/logs/"*/trainer/torchrun/*/*/*/stderr.log 2>/dev/null); do
    echo "---- $f (last 40 lines)"; grep -v -E "SyntaxWarning|register_constant|make_block_ptr|warn\(" "$f" | tail -40
  done
  exit 1; }

# 4. Post-checks: targets found, loss decreased, adapter present and non-trivial.
TRAINER_LOG=$(ls "$HOME/runs/$RUN/logs/"*/trainer.log | head -1)
uv run python - "$TRAINER_LOG" "$HOME/runs/$RUN/adapter" "$HOME/runs/$RUN.train_report.json" "$STEPS" <<'PY'
import json, re, sys, os
log, adapter, report_path, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
text = open(log, errors="ignore").read()
targets = re.search(r"Found (\d+) target modules", text) or re.search(r"LoRA enabled: ([\d,]+) adapter params", text)
losses = [float(x) for x in re.findall(r"Step \d+ \|[^|]*\| Loss ([0-9.]+)", text)]
peak = re.findall(r"Peak Mem\. ([0-9.]+)/([0-9.]+) GiB", text)
from safetensors import safe_open
size = os.path.getsize(f"{adapter}/adapter_model.safetensors") if os.path.exists(f"{adapter}/adapter_model.safetensors") else 0
n_tensors = 0
if size:
    with safe_open(f"{adapter}/adapter_model.safetensors", "pt") as f:
        n_tensors = len(list(f.keys()))
k = max(1, len(losses) // 10)
first, last = (sum(losses[:k]) / k, sum(losses[-k:]) / k) if losses else (None, None)
report = {"target_modules": int(targets.group(1).replace(",", "")) if targets else None,  # module count, or adapter param count on newer prime-rl "steps_logged": len(losses), "steps_expected": steps,
          "loss_first": first, "loss_last": last, "peak_mem_gib": peak[-1] if peak else None,
          "adapter_bytes": size, "adapter_tensors": n_tensors}
problems = []
if not report["target_modules"]: problems.append("LoRA found no target modules")
if len(losses) < steps: problems.append(f"only {len(losses)}/{steps} steps logged")
if first is not None and last >= first * 0.9: problems.append(f"loss did not decrease ({first:.4f} -> {last:.4f})")
if size < 1_000_000 or n_tensors == 0: problems.append("adapter missing or empty")
report["problems"] = problems
json.dump(report, open(report_path, "w"), indent=2)
print("train report:", json.dumps(report))
sys.exit(1 if [p for p in problems if "loss" not in p] else 0)  # a flat loss is reported, not fatal: eval decides
PY
echo "adapter: $HOME/runs/$RUN/adapter"
