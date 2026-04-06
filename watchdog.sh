#!/bin/bash
# Keep training running until killed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/scripts/train_kda_muon.py"
LOG_DIR="$SCRIPT_DIR/watchdog_logs"
mkdir -p "$LOG_DIR"

FIRST_RUN=true
RUN_NUM=0

cleanup() {
    echo "[watchdog] Stopping..."
    [[ -n "${CHILD_PID:-}" ]] && kill "$CHILD_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    RUN_NUM=$((RUN_NUM + 1))
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/run_${RUN_NUM}_${TIMESTAMP}.log"

    RESUME_FLAG=""
    if [[ "$FIRST_RUN" == "false" ]]; then
        RESUME_FLAG="--resume"
    else
        ls "$SCRIPT_DIR/checkpoints/step_"*.pt 2>/dev/null | grep -q . && RESUME_FLAG="--resume"
        FIRST_RUN=false
    fi

    echo "[watchdog] === Run #${RUN_NUM} at $(date) | args: $RESUME_FLAG ==="
    python -u "$TRAIN_SCRIPT" $RESUME_FLAG 2>&1 | tee "$LOG_FILE" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    EXIT_CODE=$?
    CHILD_PID=""

    echo "[watchdog] === Run #${RUN_NUM} exited (code ${EXIT_CODE}) ==="
    [[ $EXIT_CODE -ne 0 ]] && sleep 30 || sleep 10
done
