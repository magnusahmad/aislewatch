#!/usr/bin/env bash
# One GPU session: evaluate base -> train -> evaluate adapter. Run on the pod under nohup.
# Usage: pod/pipeline.sh <dataset> <run_name> [epochs] [lr]
# Progress: ~/pipeline.log; done marker: ~/pipeline.<run>.done (or .failed)
set -uo pipefail
DATASET="$1"; RUN="$2"; EPOCHS="${3:-2}"; LR="${4:-1e-4}"
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/aislewatch"
step() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { step "FAILED: $*"; touch "$HOME/pipeline.$RUN.failed"; bash pod/serve.sh --stop; exit 1; }
rm -f "$HOME/pipeline.$RUN.done" "$HOME/pipeline.$RUN.failed"

step "setup"; bash pod/setup.sh || fail setup

if [ ! -f "data/results/$DATASET/test/qwen-base-structured.summary.json" ]; then
  step "serve base"; bash pod/serve.sh || fail "serve base"
  for split in test val; do
    step "eval base $split"
    uv run aw-eval --dataset "$DATASET" --split "$split" --model qwen-base || fail "eval base $split"
  done
  uv run aw-eval --dataset "$DATASET" --split test --model qwen-base --prompt generic || true
  uv run aw-eval --dataset "$DATASET" --split test --model qwen-base --shuffle-frames || true
  bash pod/serve.sh --stop
fi

step "train $RUN"; bash pod/train.sh "$DATASET" "$RUN" "$EPOCHS" "$LR" || fail "train"

step "serve base + $RUN"
python3 - "$RUN" <<'EOF'
import json, sys, pathlib
run = sys.argv[1]; p = pathlib.Path("config/models.json"); m = json.loads(p.read_text())
m.setdefault(f"qwen-{run}", {**m["qwen-v1"], "model": f"qwen-{run}"}); p.write_text(json.dumps(m, indent=2))
EOF
bash pod/serve.sh "qwen-$RUN=$HOME/runs/$RUN/adapter" || fail "serve adapter"
for split in test val; do
  step "eval $RUN $split"
  uv run aw-eval --dataset "$DATASET" --split "$split" --model "qwen-$RUN" || fail "eval $RUN $split"
done
uv run aw-eval --dataset "$DATASET" --split test --model "qwen-$RUN" --shuffle-frames || true
bash pod/serve.sh --stop
uv run aw-scoreboard --dataset "$DATASET" --split test
uv run aw-scoreboard --dataset "$DATASET" --split val
step "done"; touch "$HOME/pipeline.$RUN.done"
