#!/usr/bin/env bash
# Laptop side: sync code + one dataset to a pod, run pod/pipeline.sh there, watch it, pull results back.
# Exits non-zero on pipeline failure, SSH loss, or a STALL (no progress for STALL_MIN minutes), so the
# launcher (pod/launch.sh) terminates the pod instead of paying for an idle or stuck GPU.
# Usage: pod/run_remote.sh <ssh-host> <dataset> <run> [sizes="4b 9b"] [epochs] [lr]
set -uo pipefail
HOST="$1"; DATASET="$2"; RUN="$3"; SIZES="${4:-4b 9b}"; EPOCHS="${5:-2}"; LR="${6:-1e-4}"
STALL_MIN="${STALL_MIN:-20}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new "$HOST")
log() { echo "[$(date +%H:%M:%S)] $*"; }

pull() {
  mkdir -p "$REPO/data/results" "$REPO/data/logs/$RUN"
  rsync -az "$HOST:aislewatch/data/results/$DATASET" "$REPO/data/results/" 2>/dev/null || true
  for size in $SIZES; do
    mkdir -p "$REPO/models/adapters/$RUN-$size"
    rsync -az "$HOST:runs/$RUN-$size/adapter/" "$REPO/models/adapters/$RUN-$size/" 2>/dev/null || true
    rsync -az "$HOST:runs/$RUN-$size/logs" "$REPO/data/logs/$RUN/$RUN-$size-trainer/" 2>/dev/null || true
    rsync -az "$HOST:runs/$RUN-$size.toml" "$HOST:runs/$RUN-$size.log" "$HOST:runs/$RUN-$size.train_report.json" \
      "$REPO/data/logs/$RUN/" 2>/dev/null || true
  done
  rsync -az "$HOST:pipeline.log" "$HOST:vllm.log" "$REPO/data/logs/$RUN/" 2>/dev/null || true
}

log "sync code and dataset $DATASET"
rsync -az --delete --exclude '.venv' --exclude 'data' --exclude 'models' --exclude '.git' "$REPO/" "$HOST:aislewatch/" || exit 1
"${SSH[@]}" "mkdir -p aislewatch/data/datasets aislewatch/data/results" < /dev/null || exit 1
rsync -az "$REPO/data/datasets/$DATASET" "$HOST:aislewatch/data/datasets/" || exit 1
[ -d "$REPO/data/results/$DATASET" ] && rsync -az "$REPO/data/results/$DATASET" "$HOST:aislewatch/data/results/"
for size in $SIZES; do   # reuse adapters already trained (pipeline.sh then skips training for that size)
  [ -s "$REPO/models/adapters/$RUN-$size/adapter_model.safetensors" ] && \
    { "${SSH[@]}" "mkdir -p runs/$RUN-$size/adapter" < /dev/null && rsync -az "$REPO/models/adapters/$RUN-$size/" "$HOST:runs/$RUN-$size/adapter/"; }
done

log "start pipeline (sizes: $SIZES)"
# "cd X || exit; nohup ... &" (not "cd X && nohup ... &"): with &&, the & backgrounds the whole list as a
# subshell that keeps SSH's output channel open, so this call blocked for the entire run and the watch loop
# below (stall / SSH-loss detection) never ran; a network blip then killed a healthy run (Oct 13, v2).
"${SSH[@]}" "cd aislewatch || exit 1; nohup bash pod/pipeline.sh $DATASET $RUN '$SIZES' $EPOCHS $LR > ~/pipeline.log 2>&1 < /dev/null &" < /dev/null || exit 1

last=""; last_change=$(date +%s); ssh_fail=0; result=1
while true; do
  # Progress fingerprint: status line + last training step + size of the logs. Any change = progress.
  probe=$("${SSH[@]}" "if [ -f ~/pipeline.$RUN.done ]; then echo DONE; elif [ -f ~/pipeline.$RUN.failed ]; then echo FAILED: \$(cat ~/pipeline.$RUN.failed); else echo RUNNING; fi;
          cat ~/pipeline.status 2>/dev/null; grep -a -o 'Step [0-9]* |[^|]*| Loss [0-9.]*' ~/runs/*/logs/*/trainer.log 2>/dev/null | tail -1;
          stat -c %s ~/pipeline.log ~/vllm.log 2>/dev/null | awk '{s+=\$1} END {print s}'" < /dev/null 2>/dev/null)
  if [ -z "$probe" ]; then
    ssh_fail=$((ssh_fail + 1)); log "ssh probe failed ($ssh_fail/10)"
    [ "$ssh_fail" -ge 10 ] && { log "STOP: lost SSH to the pod"; break; }
    sleep 30; continue
  fi
  ssh_fail=0
  if [ "$probe" != "$last" ]; then
    log "$(echo "$probe" | head -3 | tr '\n' ' ')"; last="$probe"; last_change=$(date +%s)
  elif [ $(( $(date +%s) - last_change )) -ge $(( STALL_MIN * 60 )) ]; then
    log "STOP: no progress for $STALL_MIN min (stalled)"; break
  fi
  case "$probe" in DONE*) result=0; break ;; FAILED*) log "pipeline failed"; break ;; esac
  sleep 30
done

log "pull results, adapters and logs"
pull
cat "$REPO/data/results/$DATASET/test/SCOREBOARD.md" 2>/dev/null || true
exit $result
