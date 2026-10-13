#!/usr/bin/env bash
# One-time pod setup: uv, prime-rl (pinned, adapter-export patch), vLLM venv, aislewatch package.
# Expects the repo synced to ~/aislewatch. Idempotent. Fixes from the Oct 12 smoke tests are baked in.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PRIME_RL_COMMIT=7c9d4744cd1d892eb6524191d869d6994c802252
VLLM_VERSION=0.31.0

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh || command -v uv; fi
git config --global url."https://github.com/".insteadOf git@github.com:   # prime-envs submodule uses an SSH URL

cd "$HOME"
[ -d prime-rl ] || git clone -q https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git fetch -q origin && git checkout -q "$PRIME_RL_COMMIT"
git submodule update --init --force -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync -q --all-extras
python3 "$HOME/aislewatch/pod/patch_export_adapter.py"

cd "$HOME"
if [ ! -x vllm-env/bin/vllm ]; then
  uv venv -q vllm-env --python 3.12
  VIRTUAL_ENV="$HOME/vllm-env" uv pip install -q "vllm==$VLLM_VERSION" --torch-backend=auto
fi

cd "$HOME/aislewatch" && uv sync -q
echo "setup done: prime-rl $PRIME_RL_COMMIT, vLLM $("$HOME/vllm-env/bin/python" -c 'import vllm; print(vllm.__version__)')"
