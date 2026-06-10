#!/usr/bin/env bash
# vim: set ft=sh:
# reflect_parallel.sh
# Launch multiple reflect.py workers, each handling a partition of sources.
#
# Usage:
#   ./reflect_parallel.sh                        # 3 processes × 2 threads
#   ./reflect_parallel.sh --workers 4 --threads 3
#   ./reflect_parallel.sh --limit 100
#   ./reflect_parallel.sh --model google/gemini-3.1-flash-lite
#   ./reflect_parallel.sh --throttle-check 60   # poll optimizer every N seconds (default: 30)
#   ./reflect_parallel.sh --throttle-sleep 10   # sleep N seconds when yellow (default: 8)
#   ./reflect_parallel.sh --no-throttle         # disable adaptive throttling entirely
#
# Logs: logs/reflect_worker_N.log
# Ctrl-C kills all workers cleanly.

set -uo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
WORKERS=3
THREADS=2
LIMIT=0
MODEL="google/gemini-3.1-flash-lite"
LOG_DIR="logs"
THROTTLE_CHECK=30      # seconds between optimizer status polls
THROTTLE_SLEEP=8       # seconds to pause all workers when optimizer is busy
THROTTLE_ENABLED=true  # set false via --no-throttle

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)         WORKERS="$2";        shift 2 ;;
        --threads)         THREADS="$2";        shift 2 ;;
        --limit)           LIMIT="$2";          shift 2 ;;
        --model)           MODEL="$2";          shift 2 ;;
        --log-dir)         LOG_DIR="$2";        shift 2 ;;
        --throttle-check)  THROTTLE_CHECK="$2"; shift 2 ;;
        --throttle-sleep)  THROTTLE_SLEEP="$2"; shift 2 ;;
        --no-throttle)     THROTTLE_ENABLED=false; shift ;;
        *) echo "[!] unknown arg: $1"; exit 1 ;;
    esac
done

# ── qdrant optimizer health check ─────────────────────────────────────────────
# Returns "green", "yellow", or "unknown" for meta_reflections.
qdrant_optimizer_status() {
    python3 - <<'PYEOF'
import os, sys, requests

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

load_env()
url  = os.environ.get("QDRANT_URL", "http://localhost:6333")
key  = os.environ.get("QDRANT_API_KEY", "").strip()
hdrs = {"api-key": key} if key else {}

try:
    r = requests.get(f"{url}/collections/meta_reflections", headers=hdrs, timeout=10)
    if r.status_code != 200:
        print("unknown"); sys.exit(0)
    data   = r.json()
    status = data.get("result", {}).get("status", "unknown")
    opt    = data.get("result", {}).get("optimizer_status", "unknown")
    if status == "yellow" or (isinstance(opt, dict) and opt.get("ok") is False):
        print("yellow")
    elif status == "green":
        print("green")
    else:
        print("unknown")
except Exception:
    print("unknown")
PYEOF
}

# ── adaptive throttle monitor (background) ────────────────────────────────────
# Polls Qdrant every THROTTLE_CHECK seconds. When meta_reflections goes yellow,
# sends SIGSTOP to all worker PIDs to freeze them, waits THROTTLE_SLEEP seconds,
# then sends SIGCONT to resume. Workers wake up exactly where they left off.
throttle_monitor() {
    local -n _pids=$1
    while true; do
        sleep "$THROTTLE_CHECK"

        local alive=0
        for pid in "${_pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && ((alive++)) || true
        done
        [[ $alive -eq 0 ]] && break

        local status
        status=$(qdrant_optimizer_status)

        if [[ "$status" == "yellow" ]]; then
            echo "[throttle] meta_reflections is yellow — pausing $alive workers for ${THROTTLE_SLEEP}s"
            for pid in "${_pids[@]}"; do
                kill -STOP "$pid" 2>/dev/null || true
            done
            sleep "$THROTTLE_SLEEP"
            for pid in "${_pids[@]}"; do
                kill -CONT "$pid" 2>/dev/null || true
            done
            echo "[throttle] workers resumed (optimizer caught up)"
        else
            echo "[throttle] meta_reflections=${status} — all clear"
        fi
    done
}

mkdir -p "$LOG_DIR"

# ── discover sources ──────────────────────────────────────────────────────────
echo "[reflect_parallel] discovering sources from qdrant..."

SOURCES_JSON=$(python3 - <<'PYEOF'
import os, json, requests

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

load_env()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
headers = {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else None

sources = set()
offset = None
while True:
    body = {"limit": 250, "with_payload": {"include": ["source_id", "source_file"]}, "with_vector": False}
    if offset:
        body["offset"] = offset
    r = requests.post(f"{QDRANT_URL}/collections/mb_chunks/points/scroll",
                      json=body, headers=headers, timeout=30).json()
    result = r.get("result", {})
    for pt in result.get("points", []):
        pl = pt.get("payload", {})
        sid = (pl.get("source_id") or pl.get("source_file") or "").strip()
        if sid:
            sources.add(sid)
    offset = result.get("next_page_offset")
    if not offset:
        break

print(json.dumps(sorted(sources)))
PYEOF
)

# parse into array
readarray -t ALL_SOURCES < <(echo "$SOURCES_JSON" | python3 -c "
import json, sys
for s in json.load(sys.stdin):
    print(s)
")

TOTAL=${#ALL_SOURCES[@]}
echo "[reflect_parallel] found $TOTAL sources, splitting across $WORKERS workers"

if [[ $TOTAL -eq 0 ]]; then
    echo "[!] no sources found in mb_chunks"
    exit 1
fi

# ── partition sources round-robin ─────────────────────────────────────────────
declare -a PARTITIONS
for ((i=0; i<WORKERS; i++)); do
    PARTITIONS[$i]=""
done

idx=0
for src in "${ALL_SOURCES[@]}"; do
    worker=$((idx % WORKERS))
    if [[ -z "${PARTITIONS[$worker]}" ]]; then
        PARTITIONS[$worker]="$src"
    else
        PARTITIONS[$worker]="${PARTITIONS[$worker]},$src"
    fi
    ((idx++))
done

# ── launch workers ────────────────────────────────────────────────────────────
PIDS=()
LOGFILES=()
THROTTLE_PID=""

cleanup() {
    echo ""
    echo "[reflect_parallel] shutting down workers..."
    # resume any frozen workers before killing so nothing gets stuck
    for pid in "${PIDS[@]}"; do
        kill -CONT "$pid" 2>/dev/null || true
        kill       "$pid" 2>/dev/null || true
    done
    [[ -n "${THROTTLE_PID}" ]] && kill "$THROTTLE_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[reflect_parallel] done"
}
trap cleanup INT TERM EXIT

for ((i=0; i<WORKERS; i++)); do
    partition="${PARTITIONS[$i]}"
    if [[ -z "$partition" ]]; then
        echo "[worker $i] no sources — skipping"
        continue
    fi

    source_count=$(echo "$partition" | tr ',' '\n' | wc -l | tr -d ' ')
    logfile="$LOG_DIR/reflect_worker_${i}.log"
    LOGFILES+=("$logfile")

    cmd_args=(--model "$MODEL" --workers "$THREADS" --source-collections "mb_chunks")
    if [[ $LIMIT -gt 0 ]]; then
        cmd_args+=(--limit "$LIMIT")
    fi

    echo "[worker $i] $source_count sources → $logfile"

    # write a header to the log immediately so the file exists
    echo "[worker $i] sources: $partition" > "$logfile"
    echo "[worker $i] cmd: python3 reflect.py ${cmd_args[*]}" >> "$logfile"
    echo "---" >> "$logfile"

    MB_REFLECT_SOURCE_FILTER="$partition" \
        python3 reflect.py "${cmd_args[@]}" >> "$logfile" 2>&1 &

    PIDS+=($!)
    sleep 0.3
done

echo ""
echo "[reflect_parallel] ${#PIDS[@]} workers running (PIDs: ${PIDS[*]})"

# ── start throttle monitor ─────────────────────────────────────────────────────
if [[ "$THROTTLE_ENABLED" == "true" && ${#PIDS[@]} -gt 0 ]]; then
    echo "[reflect_parallel] throttle monitor ON (check every ${THROTTLE_CHECK}s, sleep ${THROTTLE_SLEEP}s on yellow)"
    throttle_monitor PIDS >> "$LOG_DIR/throttle.log" 2>&1 &
    THROTTLE_PID=$!
else
    echo "[reflect_parallel] throttle monitor OFF"
fi

echo ""
echo "Watch all logs:"
echo "  tail -f $LOG_DIR/reflect_worker_*.log"
echo ""
echo "Watch one:"
for ((i=0; i<${#LOGFILES[@]}; i++)); do
    echo "  tail -f ${LOGFILES[$i]}"
done
if [[ "$THROTTLE_ENABLED" == "true" ]]; then
    echo ""
    echo "Watch throttle:"
    echo "  tail -f $LOG_DIR/throttle.log"
fi
echo ""
echo "Ctrl-C to stop all workers"
echo ""

# wait for all workers
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo "[reflect_parallel] all workers finished"
