#!/bin/bash
# watchdog.sh — monitors a FitLLM training log and auto-restarts from the
# latest checkpoint if the log goes silent for STALE_SECS seconds.
#
# Usage:
#   chmod +x watchdog.sh
#   ./watchdog.sh &
#
# The watchdog loops forever. Kill it with: kill <watchdog_pid>

set -euo pipefail

SHARD_DIR="./shards-qwen32b"
DATASET="databricks/databricks-dolly-15k"
CKPT_DIR="./checkpoints-qwen32b"
LOG_DIR="./logs"
VENV="/data/fitllm-venv"

# Kill the training process if the log hasn't grown for this many seconds.
STALE_SECS=300   # 5 minutes — one step is ~88s, so 5 min = over 3 missed steps

POLL_INTERVAL=30  # check log freshness every 30 seconds

find_latest_checkpoint() {
    # Returns the path of the most-recently modified checkpoint file, or empty string.
    latest=$(find "$CKPT_DIR" -name "checkpoint_step_*.pt" -printf "%T@ %p\n" 2>/dev/null \
             | sort -rn | head -1 | awk '{print $2}')
    echo "$latest"
}

find_latest_log() {
    # Returns the most-recently modified log file, or empty string.
    latest=$(find "$LOG_DIR" -name "*.log" -printf "%T@ %p\n" 2>/dev/null \
             | sort -rn | head -1 | awk '{print $2}')
    echo "$latest"
}

start_training() {
    local ckpt="$1"
    local run_num
    run_num=$(date +"%Y%m%d_%H%M%S")
    local logfile="$LOG_DIR/watchdog_restart_${run_num}.log"

    echo "[watchdog] $(date): Starting training. Checkpoint: ${ckpt:-none}. Log: $logfile"

    local resume_flag=""
    if [ -n "$ckpt" ]; then
        resume_flag="--resume $ckpt"
    fi

    source "$VENV/bin/activate"
    # shellcheck disable=SC2086
    nohup python -m fitllm train \
        --shard-dir "$SHARD_DIR" \
        --dataset "$DATASET" \
        $resume_flag \
        > "$logfile" 2>&1 &

    echo "$!"   # return the PID
}

kill_training() {
    local pid="$1"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[watchdog] $(date): Killing frozen training process PID $pid"
        kill -9 "$pid" 2>/dev/null || true
        sleep 3
    fi
}

mkdir -p "$LOG_DIR"

echo "[watchdog] $(date): Watchdog started. Stale threshold=${STALE_SECS}s, poll=${POLL_INTERVAL}s"

TRAIN_PID=""
LOG_FILE=""

# Main loop
while true; do
    # Check if the training process is alive
    if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[watchdog] $(date): Training process $TRAIN_PID exited (deadlock kill or crash)"
        TRAIN_PID=""
    fi

    # Start training if not running
    if [ -z "$TRAIN_PID" ]; then
        ckpt=$(find_latest_checkpoint)
        TRAIN_PID=$(start_training "$ckpt")
        LOG_FILE=$(find_latest_log)
        echo "[watchdog] $(date): Training started with PID $TRAIN_PID"
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Refresh log file pointer (a new run creates a new log file)
    LOG_FILE=$(find_latest_log)

    if [ -z "$LOG_FILE" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Check log freshness
    last_mod=$(stat -c "%Y" "$LOG_FILE" 2>/dev/null || echo "0")
    now=$(date +%s)
    age=$(( now - last_mod ))

    if [ "$age" -gt "$STALE_SECS" ]; then
        echo "[watchdog] $(date): Log '$LOG_FILE' is ${age}s old (>${STALE_SECS}s threshold). Deadlock detected."
        kill_training "$TRAIN_PID"
        TRAIN_PID=""
        # Brief wait to let GPU release memory before restarting
        echo "[watchdog] $(date): Waiting 15s for GPU memory to clear..."
        sleep 15
    else
        sleep "$POLL_INTERVAL"
    fi
done
