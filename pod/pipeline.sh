#!/usr/bin/env bash
# One GPU session: for each model size, evaluate base -> train -> evaluate adapter. Run on the pod under nohup.
# Usage: pod/pipeline.sh <dataset> <run> [sizes="4b 9b"] [epochs=2] [lr=1e-4]
# Progress: ~/pipeline.log (one "[HH:MM:SS] ..." line per step); status: ~/pipeline.status
# Done marker: ~/pipeline.<run>.done, or ~/pipeline.<run>.failed with the reason.
# Every step is checked; any failed check stops the run so the launcher terminates the pod.
set -uo pipefail
DATASET="$1"; RUN="$2"; SIZES="${3:-4b 9b}"; EPOCHS="${4:-2}"; LR="${5:-1e-4}"
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/aislewatch"
status() { echo "$*" > "$HOME/pipeline.status"; echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { status "FAILED: $*"; echo "$*" > "$HOME/pipeline.$RUN.failed"; bash pod/serve.sh --stop >/dev/null 2>&1; exit 1; }
rm -f "$HOME/pipeline.$RUN.done" "$HOME/pipeline.$RUN.failed"
hf_model() { case "$1" in 4b) echo "Qwen/Qwen3.5-4B" ;; 9b) echo "Qwen/Qwen3.5-9B" ;; *) echo "unknown size $1" >&2; return 1 ;; esac; }

# Register served model names (qwen4b-base, qwen4b-<run>, ...) in config/models.json so aw-eval can address them.
register() {
  uv run python - "$1" <<'EOF'
import json, sys, pathlib
name = sys.argv[1]; p = pathlib.Path("config/models.json"); m = json.loads(p.read_text())
m.setdefault(name, {**m["qwen-base"], "model": name}); p.write_text(json.dumps(m, indent=2))
EOF
}

# Evaluate one served model on a split and check the run is complete and mostly error-free.
evaluate() {  # evaluate <model_name> <split> [extra aw-eval args]
  local name="$1" split="$2"; shift 2
  uv run aw-eval --dataset "$DATASET" --split "$split" --model "$name" "$@" > /dev/null || return 1
  local tag="$name-structured"; [[ " $* " == *" --shuffle-frames "* ]] && tag="$tag-shuffled"
  [[ " $* " == *" --prompt generic "* ]] && tag="$name-generic"
  uv run python - "data/results/$DATASET/$split/$tag.summary.json" "data/datasets/$DATASET" "$split" <<'EOF'
import json, sys, pathlib
s = json.load(open(sys.argv[1])); root = pathlib.Path(sys.argv[2]); split = sys.argv[3]
expected = sum(1 for _ in open(root / "test" / "test.jsonl")) if split == "test" else \
    __import__("pyarrow.parquet", fromlist=["x"]).read_metadata(root / f"{split}.parquet").num_rows
problems = []
if s["n"] != expected: problems.append(f"evaluated {s['n']} of {expected} windows")
if s.get("errors", 0) > 0.05 * s["n"]: problems.append(f"{s['errors']} request errors")
if "valid_json_rate" in s and s["valid_json_rate"] < 0.5: problems.append(f"valid JSON only {s['valid_json_rate']}")
print(f"  {s['model']} {split}{' shuffled' if s.get('shuffled') else ''}: n={s['n']}",
      f"F1={s.get('macro_f1')} acc={s.get('accuracy')} json={s.get('valid_json_rate')} errors={s.get('errors')}" if "macro_f1" in s
      else f"door-mention rate={s.get('mentions_door_state_rate')}", "| CHECK FAILED: " + "; ".join(problems) if problems else "| ok")
sys.exit(1 if problems else 0)
EOF
}

status "setup"; bash pod/setup.sh || fail "setup"

for size in $SIZES; do
  MODEL=$(hf_model "$size") || fail "unknown size $size"
  BASE="qwen$size-base"; TUNED="qwen$size-$RUN"; RUNDIR="$RUN-$size"
  register "$BASE"; register "$TUNED"

  if [ ! -f "data/results/$DATASET/test/$BASE-structured.summary.json" ]; then
    status "$size: serve base"; bash pod/serve.sh "$MODEL" "$BASE" || fail "$size: serve base"
    status "$size: eval base"
    evaluate "$BASE" test || fail "$size: eval base test"
    evaluate "$BASE" val || fail "$size: eval base val"
    evaluate "$BASE" test --prompt generic || status "warn: $size base generic eval incomplete (side check, continuing)"
    evaluate "$BASE" test --shuffle-frames || status "warn: $size base shuffled eval incomplete (side check, continuing)"
    bash pod/serve.sh --stop || fail "$size: stop vLLM"
  fi

  if [ -s "$HOME/runs/$RUNDIR/adapter/adapter_model.safetensors" ]; then
    status "$size: adapter exists, skipping training"   # pushed by run_remote.sh from models/adapters/
  else
    status "$size: train"
    bash pod/train.sh "$MODEL" "$DATASET" "$RUNDIR" "$EPOCHS" "$LR" || fail "$size: train (see ~/runs/$RUNDIR.log)"
  fi

  status "$size: serve adapter"
  bash pod/serve.sh "$MODEL" "$BASE" "$TUNED=$HOME/runs/$RUNDIR/adapter" || fail "$size: serve adapter"
  status "$size: eval adapter"
  evaluate "$TUNED" test || fail "$size: eval $TUNED test"
  evaluate "$TUNED" val || fail "$size: eval $TUNED val"
  evaluate "$TUNED" test --shuffle-frames || status "warn: $size $TUNED shuffled eval incomplete (side check, continuing)"
  bash pod/serve.sh --stop || fail "$size: stop vLLM"
  uv run aw-scoreboard --dataset "$DATASET" --split test > /dev/null
  status "$size: done"
done

uv run aw-scoreboard --dataset "$DATASET" --split val > /dev/null
status "done"; touch "$HOME/pipeline.$RUN.done"
