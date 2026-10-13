#!/usr/bin/env bash
# Spend-safe GPU session: create a pod, run the pipeline, ALWAYS terminate the pod.
#
#  - refuses to start if (hourly price x MAX_HOURS) > MAX_USD, or the wallet can't cover it
#  - independent watchdog terminates the pod after MAX_HOURS even if this script dies
#  - terminates on success, failure, stall (run_remote.sh), Ctrl-C or any error (trap)
#  - appends every pod to data/gpu/ledger.jsonl with times and estimated cost
#
# Usage: pod/launch.sh <dataset> <run> [sizes="4b 9b"]
# Env:   GPU_TYPE=L40S_48GB MAX_HOURS=3.5 MAX_USD=4 STALL_MIN=20 EPOCHS=2 LR=1e-4
set -uo pipefail
DATASET="$1"; RUN="$2"; SIZES="${3:-4b 9b}"
GPU_TYPE="${GPU_TYPE:-L40S_48GB}"; MAX_HOURS="${MAX_HOURS:-3.5}"; MAX_USD="${MAX_USD:-4}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; LEDGER="$REPO/data/gpu/ledger.jsonl"; mkdir -p "$(dirname "$LEDGER")"
log() { echo "[$(date +%H:%M:%S)] $*"; }
POD=""; WATCHDOG=""; START=0; PRICE=0

terminate() {
  if [ -n "$POD" ]; then
    log "terminating pod $POD"
    for _ in 1 2 3; do prime --plain pods terminate "$POD" -y < /dev/null >/dev/null 2>&1 && break; sleep 10; done
    local mins=$(( ($(date +%s) - START) / 60 ))
    python3 - "$LEDGER" "$POD" "$GPU_TYPE" "$PRICE" "$START" "$mins" "$RUN" "${1:-}" <<'EOF'
import json, sys, time
ledger, pod, gpu, price, start, mins, run, outcome = sys.argv[1:9]
rec = {"pod": pod, "gpu": gpu, "usd_per_hour": float(price), "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(int(start))),
       "ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "minutes": int(mins), "est_usd": round(float(price) * int(mins) / 60, 2),
       "run": run, "outcome": outcome or "unknown"}
open(ledger, "a").write(json.dumps(rec) + "\n"); print("ledger:", json.dumps(rec))
EOF
    POD=""
  fi
  if [ -n "$WATCHDOG" ]; then pkill -P "$WATCHDOG" 2>/dev/null; kill "$WATCHDOG" 2>/dev/null; fi; WATCHDOG=""
  log "pods still running: $(prime --plain pods list --output json 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); d=d.get("pods",d) if isinstance(d,dict) else d; print(len(d))' 2>/dev/null)"
}
trap 'terminate interrupted; exit 130' INT TERM
trap 'terminate "${OUTCOME:-error}"' EXIT

# 1. Price and budget checks.
OFFER=$(prime --plain availability list --gpu-type "$GPU_TYPE" --gpu-count 1 --output json 2>/dev/null | python3 -c '
import sys, json
r = [x for x in json.load(sys.stdin)["gpu_resources"] if not x.get("is_spot") and x.get("stock_status") != "Unavailable"]
r.sort(key=lambda x: x["price_value"])
print(" ".join(str(r[0][k]) for k in ("id", "price_value", "provider")) if r else "")')
[ -n "$OFFER" ] || { log "no $GPU_TYPE available"; OUTCOME=no-gpu; exit 75; }   # 75 = capacity, retryable (pod/wait_launch.sh)
read -r OFFER_ID PRICE PROVIDER <<< "$OFFER"
WORST=$(python3 -c "print(round($PRICE * $MAX_HOURS, 2))")
BALANCE=$(prime --plain wallet 2>/dev/null | python3 -c 'import sys,re; m=re.search(r"Balance:\s*\$([0-9.]+)", sys.stdin.read()); print(m.group(1) if m else 0)')
log "offer $OFFER_ID: $GPU_TYPE at \$$PRICE/h ($PROVIDER); worst case \$$WORST over $MAX_HOURS h; budget \$$MAX_USD; wallet \$$BALANCE"
python3 -c "import sys; sys.exit(0 if $WORST <= $MAX_USD and $BALANCE >= $WORST else 1)" || { log "REFUSED: worst case exceeds budget or wallet"; OUTCOME=refused; exit 1; }

# 2. Create the pod and arm the watchdog immediately.
START=$(date +%s)
CREATE_OUT=$(prime --plain pods create --id "$OFFER_ID" --name "aislewatch-$RUN" --disk-size 200 --image "${IMAGE:-ubuntu_22_cuda_12}" -y < /dev/null 2>&1)
POD=$(printf '%s\n' "$CREATE_OUT" | grep -oi 'created pod [0-9a-f]*' | awk '{print $3}')
if [ -z "$POD" ]; then
  log "pod creation failed; prime said:"; printf '%s\n' "$CREATE_OUT" | tail -20; OUTCOME=create-failed
  printf '%s' "$CREATE_OUT" | grep -qiE "HTTP 503|available right now|try again later" && exit 75   # out of stock: retryable
  exit 1
fi
log "pod $POD created"
WATCH_S=$(python3 -c "print(int($MAX_HOURS * 3600))")
nohup bash -c "sleep $WATCH_S; prime --plain pods terminate $POD -y; echo '{\"pod\": \"$POD\", \"watchdog_fired\": \"'\$(date +%FT%T)'\"}' >> '$LEDGER'" \
  > "$REPO/data/gpu/watchdog-$POD.log" 2>&1 < /dev/null &
WATCHDOG=$!
log "watchdog armed: terminates $POD after $MAX_HOURS h (pid $WATCHDOG)"

# 3. Wait for SSH (max 20 min).
HOST=""
for _ in $(seq 1 60); do
  HOST=$(prime --plain pods status "$POD" 2>/dev/null | awk '/^SSH/ {print $2}' | grep '@' || true)
  [ -n "$HOST" ] && ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$HOST" true < /dev/null 2>/dev/null && break
  HOST=""; sleep 20
done
[ -n "$HOST" ] || { log "pod never became reachable"; OUTCOME=unreachable; exit 1; }
log "pod reachable at $HOST"

# 4. Run and watch. run_remote.sh exits non-zero on failure or stall.
if bash "$REPO/pod/run_remote.sh" "$HOST" "$DATASET" "$RUN" "$SIZES" "${EPOCHS:-2}" "${LR:-1e-4}"; then OUTCOME=done; else OUTCOME=failed-or-stalled; fi
log "outcome: $OUTCOME"
exit 0   # EXIT trap terminates the pod and writes the ledger
