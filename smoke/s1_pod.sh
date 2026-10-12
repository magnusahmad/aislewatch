#!/usr/bin/env bash
# S1 on the GPU pod: install prime-rl (pinned), patch adapter export, dry-run, train, inspect adapter.
# Expects /root/smoke/ to contain this folder and /root/smoke/sft/train.parquet (copied by the laptop).
set -euo pipefail

PRIME_RL_COMMIT=7c9d4744cd1d892eb6524191d869d6994c802252
cd /root

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d prime-rl ]; then git clone https://github.com/PrimeIntellect-ai/prime-rl.git; fi
cd prime-rl
git fetch -q origin && git checkout -q "$PRIME_RL_COMMIT"
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --all-extras
uv run python -V
python3 /root/smoke/patch_export_adapter.py

echo "== Dry run (resolved config) =="
uv run sft @ /root/smoke/s1_sft.toml --dry-run --output-dir /root/runs --run.name s1-dry --clean
find /root/runs/s1-dry -name '*.json' -path '*resolved*' | head -5

echo "== Train =="
uv run sft @ /root/smoke/s1_sft.toml --output-dir /root/runs --run.name s1 --clean 2>&1 | tee /root/s1.log

echo "== LoRA targets found =="
grep -m2 "target modules" /root/s1.log || true

echo "== Adapter =="
ls -la /root/runs/s1/adapter
uv run python - <<'EOF'
import json
from safetensors import safe_open
p = "/root/runs/s1/adapter"
cfg = json.load(open(f"{p}/adapter_config.json"))
print("target_modules:", cfg.get("target_modules"))
with safe_open(f"{p}/adapter_model.safetensors", "pt") as f:
    keys = list(f.keys())
print(len(keys), "tensors; sample:", keys[:4])
print("linear-attention layers adapted:", any(("linear_attn" in k) or ("in_proj" in k) for k in keys))
print("vision tower adapted:", any("visual" in k for k in keys))
EOF
