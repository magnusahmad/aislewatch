#!/usr/bin/env bash
# S1 on the GPU pod: install prime-rl (pinned), patch adapter export, dry-run, train, inspect adapter.
# Expects $HOME/smoke/ to contain this folder and $HOME/smoke/sft/train.parquet (copied by the laptop).
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # S1 OOMed in fla Triton autotune without it
PRIME_RL_COMMIT=7c9d4744cd1d892eb6524191d869d6994c802252
cd "$HOME"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh || command -v uv; fi

if [ ! -d prime-rl ]; then git clone https://github.com/PrimeIntellect-ai/prime-rl.git; fi
cd prime-rl
git config --global url."https://github.com/".insteadOf git@github.com:   # prime-envs submodule uses an SSH URL
git fetch -q origin && git checkout -q "$PRIME_RL_COMMIT"
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --all-extras
uv run python -V
python3 $HOME/smoke/patch_export_adapter.py

echo "== Dry run (resolved config) =="
uv run sft @ $HOME/smoke/s1_sft.toml --data.name $HOME/smoke/sft --dry-run --output-dir $HOME/runs --run.name s1-dry --clean
find $HOME/runs/s1-dry -name '*.json' -path '*resolved*' | head -5

echo "== Train =="
uv run sft @ $HOME/smoke/s1_sft.toml --data.name $HOME/smoke/sft --output-dir $HOME/runs --run.name s1 --clean 2>&1 | tee $HOME/s1.log

echo "== LoRA targets found =="
grep -m2 "target modules" $HOME/s1.log || true

echo "== Adapter =="
ls -la $HOME/runs/s1/adapter
uv run python - <<'EOF'
import json
from safetensors import safe_open
import os; p = os.path.expanduser("~/runs/s1/adapter")
cfg = json.load(open(f"{p}/adapter_config.json"))
print("target_modules:", cfg.get("target_modules"))
with safe_open(f"{p}/adapter_model.safetensors", "pt") as f:
    keys = list(f.keys())
print(len(keys), "tensors; sample:", keys[:4])
print("linear-attention layers adapted:", any(("linear_attn" in k) or ("in_proj" in k) for k in keys))
print("vision tower adapted:", any("visual" in k for k in keys))
EOF
