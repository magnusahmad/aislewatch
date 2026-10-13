#!/usr/bin/env bash
# Laptop side: sync code + one dataset to a pod, run pod/pipeline.sh there, wait, pull results back.
# Pod creation and termination stay explicit (cost decisions), see MORNING.md.
# Usage: pod/run_remote.sh <ssh-host e.g. ubuntu@1.2.3.4> <dataset> <run_name> [epochs] [lr]
set -euo pipefail
HOST="$1"; DATASET="$2"; RUN="$3"; EPOCHS="${4:-2}"; LR="${5:-1e-4}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SSH=(ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$HOST")

echo "sync code and dataset $DATASET"
rsync -az --delete --exclude '.venv' --exclude 'data' --exclude 'models' --exclude '.git' "$REPO/" "$HOST:aislewatch/"
"${SSH[@]}" "mkdir -p aislewatch/data/datasets" < /dev/null
rsync -az "$REPO/data/datasets/$DATASET" "$HOST:aislewatch/data/datasets/"
if [ -d "$REPO/data/results/$DATASET" ]; then
  "${SSH[@]}" "mkdir -p aislewatch/data/results" < /dev/null
  rsync -az "$REPO/data/results/$DATASET" "$HOST:aislewatch/data/results/"   # reuse a finished base eval
fi

echo "start pipeline on pod"
"${SSH[@]}" "cd aislewatch && nohup bash pod/pipeline.sh $DATASET $RUN $EPOCHS $LR > ~/pipeline.log 2>&1 < /dev/null &" < /dev/null

last=""
while true; do
  state=$("${SSH[@]}" "if [ -f ~/pipeline.$RUN.done ]; then echo DONE; elif [ -f ~/pipeline.$RUN.failed ]; then echo FAILED; else echo RUNNING; fi; grep -a '^\[' ~/pipeline.log | tail -1; grep -a -o 'Step [0-9]* |[^|]*| Loss [0-9.]*' ~/runs/$RUN/logs/*/trainer.log 2>/dev/null | tail -1" < /dev/null 2>/dev/null || echo "SSH-ERROR")
  if [ "$state" != "$last" ]; then echo "$(date +%H:%M:%S) $state" | tr '\n' ' '; echo; last="$state"; fi
  case "$state" in DONE*|FAILED*) break ;; esac
  sleep 30
done

echo "pull results, adapter and logs"
mkdir -p "$REPO/data/results" "$REPO/models/adapters/$RUN" "$REPO/data/logs/$RUN"
rsync -az "$HOST:aislewatch/data/results/$DATASET" "$REPO/data/results/" || true
rsync -az "$HOST:runs/$RUN/adapter/" "$REPO/models/adapters/$RUN/" || true
rsync -az "$HOST:runs/$RUN.toml" "$HOST:runs/$RUN.log" "$HOST:pipeline.log" "$REPO/data/logs/$RUN/" || true
cat "$REPO/data/results/$DATASET/test/SCOREBOARD.md" 2>/dev/null || true
case "$state" in DONE*) exit 0 ;; *) exit 1 ;; esac
