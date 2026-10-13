#!/usr/bin/env bash
# LoRA SFT of Qwen3.5-9B on a built dataset with prime-rl. Writes the adapter to ~/runs/<run>/adapter.
# Usage: pod/train.sh <dataset_name> <run_name> [epochs=2] [lr=1e-4] [batch=8]
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASET="$1"; RUN="$2"; EPOCHS="${3:-2}"; LR="${4:-1e-4}"; BATCH="${5:-8}"
DATA_DIR="$HOME/aislewatch/data/datasets/$DATASET"
mkdir -p "$DATA_DIR/sft-train" && ln -sf "$DATA_DIR/train.parquet" "$DATA_DIR/sft-train/train.parquet"
ROWS=$(cd "$HOME/aislewatch" && uv run python -c "import pyarrow.parquet as pq; print(pq.read_metadata('$DATA_DIR/train.parquet').num_rows)")
STEPS=$(( (ROWS * EPOCHS + BATCH - 1) / BATCH ))
WARMUP=$(( STEPS / 15 > 1 ? STEPS / 15 : 1 ))
CFG="$HOME/runs/$RUN.toml"; mkdir -p "$HOME/runs"
cat > "$CFG" <<EOF
max_steps = $STEPS

[model]
name = "Qwen/Qwen3.5-9B"
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
echo "train: rows=$ROWS epochs=$EPOCHS batch=$BATCH steps=$STEPS lr=$LR warmup=$WARMUP"
cd "$HOME/prime-rl"
uv run sft @ "$CFG" --output-dir "$HOME/runs" --run.name "$RUN" --clean --dry-run > "$HOME/runs/$RUN.dryrun.log" 2>&1
uv run sft @ "$CFG" --output-dir "$HOME/runs" --run.name "$RUN" --clean 2>&1 | tee "$HOME/runs/$RUN.log"
test -f "$HOME/runs/$RUN/adapter/adapter_model.safetensors"
echo "adapter: $HOME/runs/$RUN/adapter"
