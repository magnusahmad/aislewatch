#!/usr/bin/env bash
# Wait for GPU capacity, then run pod/launch.sh once. Retries ONLY on capacity errors (launch.sh exit 75:
# no offer listed, or the provider answered "not available right now"); any other outcome is final.
# Nothing is spent while waiting: no pod exists until creation succeeds.
# Usage: pod/wait_launch.sh <dataset> <run> [sizes] ; Env: GPU_TYPES="L40S_48GB RTX6000Ada_48GB" WAIT_HOURS=3 RETRY_MIN=5
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU_TYPES="${GPU_TYPES:-L40S_48GB RTX6000Ada_48GB}"; WAIT_HOURS="${WAIT_HOURS:-3}"; RETRY_MIN="${RETRY_MIN:-5}"
DEADLINE=$(( $(date +%s) + $(python3 -c "print(int($WAIT_HOURS*3600))") ))
attempt=0
while :; do
  for g in $GPU_TYPES; do
    attempt=$((attempt+1)); echo "[$(date +%T)] wait_launch: attempt $attempt on $g"
    GPU_TYPE="$g" bash "$REPO/pod/launch.sh" "$@"; rc=$?
    [ $rc -eq 75 ] || { echo "[$(date +%T)] wait_launch: launch.sh finished with exit $rc"; exit $rc; }
  done
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "[$(date +%T)] wait_launch: GAVE UP after $WAIT_HOURS h, no capacity"; exit 75; }
  sleep $(( RETRY_MIN * 60 ))
done
