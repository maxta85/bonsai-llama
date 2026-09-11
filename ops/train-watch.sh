#!/bin/bash
# ops/train-watch.sh v2 — progress state machine for detached Colab training.
#
# States: ACQUIRING → STARTING → RUNNING → COMPLETE (or STALLED/FAILED)
#
# Liveness = Hub checkpoint step ADVANCING (not pgrep, not log tail).
# Polls the HF Hub API every 5 min; if no new checkpoint within 2 checkpoint
# intervals (20 steps ≈ 40 min at observed pace) while RUNNING → STALLED.
#
# On STALLED or session-gone: acquire a T4 via try-t4.sh, upload ALL THREE
# files (train_release.py, train.jsonl, release manifest), relaunch detached,
# verify startup by seeing a fresh STEP line or checkpoint within 15 min.
#
# Single-writer lease: lockfile /tmp/train-watch.lease with PID + timestamp.
# Another watcher instance must exit immediately.
#
# Exit 0 ONLY when: trainer prints the completion receipt AND the Hub has
# releases/ final upload verified.
#
# Security: HF_TOKEN read from environment ONLY. Never hardcoded.
#
# Usage: HF_TOKEN=hf_xxx bash ops/train-watch.sh
set -u

export HOME=/home/coder
export PATH="/home/coder/.local/bin:$PATH"

# --- Configuration -------------------------------------------------------
HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable is required}"
HUB_API="https://huggingface.co/api/models/maxta85/bonsai-checkpoints"
HUB_REPO="maxta85/bonsai-checkpoints"
LOG="/tmp/train_watch.log"
LEASE_FILE="/tmp/train-watch.lease"
SESSION_NAME="bonsai-16b"
POLL_INTERVAL=300          # 5 minutes
STALL_INTERVALS=2          # 2 checkpoint intervals = 20 steps ≈ 40 min
STARTUP_TIMEOUT=900         # 15 minutes to see first STEP or checkpoint
MAX_RETRIES=2              # retry once after FAILED
TOTAL_STEPS_TARGET=2002
CKPT_STEPS_INTERVAL=10      # checkpoints every 10 steps

# Paths to upload to the VM
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_release.py"
DATASET_FILE="${BONSAI_DATASET:-/tmp/bonsai_sft_merged.jsonl}"
MANIFEST_FILE="/tmp/bonsai_release_manifest.json"

# --- State machine ------------------------------------------------------
STATE="ACQUIRING"
LAST_SEEN_STEP=0
LAST_SEEN_STEP_TIME=0
RETRY_COUNT=0

# --- Logging ------------------------------------------------------------

log() {
    echo "$(date -u +%FT%TZ) [$STATE] $*" >> "$LOG"
}

# --- Single-writer lease ------------------------------------------------

acquire_lease() {
    if [ -f "$LEASE_FILE" ]; then
        local pid ts
        pid=$(awk -F: '{print $1}' "$LEASE_FILE" 2>/dev/null || echo "")
        ts=$(awk -F: '{print $2}' "$LEASE_FILE" 2>/dev/null || echo "")
        # Check if the PID is still alive
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "another watcher (PID $pid) is already running; exiting"
            exit 0
        fi
        # Stale lease — take over
        log "stale lease from PID $pid (ts=$ts); taking over"
    fi
    echo "$$:$(date +%s)" > "$LEASE_FILE"
    log "lease acquired (PID $$)"
}

release_lease() {
    if [ -f "$LEASE_FILE" ]; then
        local pid
        pid=$(awk -F: '{print $1}' "$LEASE_FILE" 2>/dev/null || echo "")
        if [ "$pid" = "$$" ]; then
            rm -f "$LEASE_FILE"
            log "lease released"
        fi
    fi
}

trap release_lease EXIT INT TERM

# --- Hub polling --------------------------------------------------------

hub_step() {
    # Returns the highest checkpoint step number on the Hub, or 0 on error.
    curl -s --max-time 20 \
        -H "Authorization: Bearer $HF_TOKEN" \
        "$HUB_API" 2>/dev/null | \
        python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    cks = sorted(set(
        int(s['rfilename'].split('/')[0].split('-')[1])
        for s in d.get('siblings', [])
        if s['rfilename'].startswith('checkpoint-')
        and '/' in s['rfilename']
    ))
    print(cks[-1] if cks else 0)
except Exception:
    print(0)
" 2>/dev/null || echo 0
}

hub_has_release() {
    # Check if a releases/ path exists on the Hub.
    curl -s --max-time 20 \
        -H "Authorization: Bearer $HF_TOKEN" \
        "$HUB_API" 2>/dev/null | \
        python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for s in d.get('siblings', []):
        if s['rfilename'].startswith('releases/'):
            print('YES')
            sys.exit(0)
    print('NO')
except Exception:
    print('NO')
" 2>/dev/null || echo "NO"
}

hub_has_completion_receipt() {
    # Check if the completion receipt file exists on the Hub.
    curl -s --max-time 20 \
        -H "Authorization: Bearer $HF_TOKEN" \
        "$HUB_API" 2>/dev/null | \
        python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for s in d.get('siblings', []):
        if s['rfilename'].startswith('releases/') and s['rfilename'].endswith('completion_receipt.json'):
            print('YES')
            sys.exit(0)
    print('NO')
except Exception:
    print('NO')
" 2>/dev/null || echo "NO"
}

# --- Upload + launch ----------------------------------------------------

sha256sum_local() {
    sha256sum "$1" 2>/dev/null | awk '{print $1}'
}

sha256sum_remote() {
    # $1 = session name, $2 = remote file path
    colab exec --session "$1" --timeout 60 2>/dev/null <<PYEOF
import hashlib
try:
    h = hashlib.sha256()
    with open("$2", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(h.hexdigest())
except Exception:
    print("ERROR")
PYEOF
}

upload_and_verify() {
    # $1 = local file, $2 = remote path
    local local_sha remote_sha
    local_sha=$(sha256sum_local "$1")
    if [ -z "$local_sha" ]; then
        log "ERROR: cannot sha256 local file $1"
        return 1
    fi
    colab upload --session "$SESSION_NAME" "$1" "$2" >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
        log "ERROR: upload failed for $1 -> $2"
        return 1
    fi
    remote_sha=$(sha256sum_remote "$SESSION_NAME" "$2")
    if [ "$remote_sha" = "ERROR" ] || [ -z "$remote_sha" ]; then
        log "ERROR: cannot sha256 remote file $2"
        return 1
    fi
    if [ "$local_sha" != "$remote_sha" ]; then
        log "ERROR: sha256 mismatch for $2: local=$local_sha remote=$remote_sha"
        return 1
    fi
    log "upload verified: $2 (sha256=$local_sha)"
    return 0
}

write_manifest() {
    # Write a release manifest with file hashes
    local script_sha dataset_sha
    script_sha=$(sha256sum_local "$TRAIN_SCRIPT")
    dataset_sha=$(sha256sum_local "$DATASET_FILE")
    python3 -c "
import json
m = {
    'script_sha256': '$script_sha',
    'dataset_sha256': '$dataset_sha',
    'created': $(date +%s),
    'target_steps': $TOTAL_STEPS_TARGET,
}
with open('$MANIFEST_FILE', 'w') as f:
    json.dump(m, f, indent=2)
print('manifest written')
" 2>/dev/null
}

acquire_and_launch() {
    # Acquire a T4 and launch detached training. Returns 0 on success.
    log "acquiring T4 session..."

    # Rotate accounts and get a T4
    if ! /home/coder/.config/colab-cli/try-t4.sh >> "$LOG" 2>&1; then
        log "FAILED: no T4 acquired (all accounts exhausted)"
        return 1
    fi

    if ! colab status --session "$SESSION_NAME" >/dev/null 2>&1; then
        log "FAILED: session not ready after acquisition"
        return 1
    fi

    log "T4 acquired; uploading files..."

    # Upload ALL THREE files: train_release.py, train.jsonl, manifest
    if ! upload_and_verify "$TRAIN_SCRIPT" "/content/train_release.py"; then
        log "FAILED: train_release.py upload/verify failed"
        return 1
    fi
    if ! upload_and_verify "$DATASET_FILE" "/content/train.jsonl"; then
        log "FAILED: dataset upload/verify failed"
        return 1
    fi
    write_manifest
    if ! upload_and_verify "$MANIFEST_FILE" "/content/release_manifest.json"; then
        log "FAILED: manifest upload/verify failed"
        return 1
    fi

    # Launch detached training
    log "launching detached training..."
    colab exec --session "$SESSION_NAME" --timeout 120 2>/dev/null <<'PY'
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Launch detached with setsid + nohup
os.system(
    "cd /content && "
    "HF_TOKEN=" + os.environ.get("HF_TOKEN", "") + " "
    "BONSAI_DATASET=/content/train.jsonl "
    "BONSAI_OUTPUT_DIR=/content/ckpt-trainer "
    "setsid nohup python3 /content/train_release.py "
    "> /content/train.log 2>&1 &"
)
print("launched")
PY

    log "detached training launched; waiting for startup..."
    return 0
}

verify_startup() {
    # Wait up to STARTUP_TIMEOUT for a STEP line or checkpoint in the log.
    local elapsed=0
    while [ "$elapsed" -lt "$STARTUP_TIMEOUT" ]; do
        local step
        step=$(hub_step)
        if [ "$step" -gt "$LAST_SEEN_STEP" ]; then
            log "startup confirmed: hub step $step > $LAST_SEEN_STEP"
            LAST_SEEN_STEP=$step
            LAST_SEEN_STEP_TIME=$(date +%s)
            return 0
        fi
        # Also check the remote log for STEP lines
        local log_tail
        log_tail=$(colab exec --session "$SESSION_NAME" --timeout 60 2>/dev/null <<'PY'
try:
    content = open("/content/train.log").read()
    for line in content.split("\n"):
        if line.startswith("STEP "):
            print(line)
            break
except Exception:
    print("NO_LOG")
PY
)
        if echo "$log_tail" | grep -q "^STEP "; then
            log "startup confirmed: STEP line in remote log"
            return 0
        fi
        sleep 60
        elapsed=$((elapsed + 60))
    done
    log "FAILED: no startup signal within ${STARTUP_TIMEOUT}s"
    return 1
}

# --- Main state machine loop --------------------------------------------

acquire_lease

while true; do
    case "$STATE" in
        ACQUIRING)
            if acquire_and_launch; then
                STATE="STARTING"
                log "transitioned to STARTING"
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
                    STATE="FAILED"
                    log "transitioned to FAILED (retries exhausted: $RETRY_COUNT)"
                    exit 1
                fi
                log "retry $RETRY_COUNT/$MAX_RETRIES in 600s..."
                sleep 600
            fi
            ;;

        STARTING)
            if verify_startup; then
                STATE="RUNNING"
                log "transitioned to RUNNING"
            else
                STATE="ACQUIRING"
                log "startup failed; transitioning to ACQUIRING"
            fi
            ;;

        RUNNING)
            STEP=$(hub_step)
            log "hub step=$STEP (last_seen=$LAST_SEEN_STEP)"

            # Check for completion
            if [ "$STEP" -ge "$TOTAL_STEPS_TARGET" ]; then
                log "step $STEP >= target $TOTAL_STEPS_TARGET"
                if [ "$(hub_has_release)" = "YES" ]; then
                    STATE="COMPLETE"
                    log "transitioned to COMPLETE"
                    continue
                else
                    log "waiting for release upload..."
                fi
            fi

            # Check for completion receipt
            if [ "$(hub_has_completion_receipt)" = "YES" ]; then
                STATE="COMPLETE"
                log "completion receipt found on Hub; transitioned to COMPLETE"
                continue
            fi

            # Check for progress stall
            if [ "$STEP" -gt "$LAST_SEEN_STEP" ]; then
                LAST_SEEN_STEP=$STEP
                LAST_SEEN_STEP_TIME=$(date +%s)
                log "progress: step $STEP"
            else
                # No progress — check if stalled
                local now=$(date +%s)
                local elapsed=$((now - LAST_SEEN_STEP_TIME))
                local stall_limit=$((STALL_INTERVALS * CKPT_STEPS_INTERVAL * 120))
                # 2 intervals * 10 steps * ~120s/step = ~2400s = 40 min
                if [ "$elapsed" -gt "$stall_limit" ]; then
                    STATE="STALLED"
                    log "STALLED: no checkpoint progress in ${elapsed}s (limit ${stall_limit}s)"
                fi
            fi

            sleep "$POLL_INTERVAL"
            ;;

        STALLED)
            log "stalled — killing session and re-acquiring..."
            # Try to kill the existing session
            colab kill --session "$SESSION_NAME" 2>/dev/null || true
            sleep 30
            STATE="ACQUIRING"
            RETRY_COUNT=0
            log "transitioned to ACQUIRING (from STALLED)"
            ;;

        COMPLETE)
            log "TRAINING COMPLETE — verified step >= $TOTAL_STEPS_TARGET and release on Hub"
            release_lease
            exit 0
            ;;

        FAILED)
            log "FAILED — retries exhausted"
            release_lease
            exit 1
            ;;
    esac
done
