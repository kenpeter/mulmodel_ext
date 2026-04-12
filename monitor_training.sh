#!/bin/bash
# Monitor training progress and alert at key milestones

echo "=== Training Monitor ==="
echo "Checking progress every 2 minutes..."
echo ""

PREV_STEP=0
while true; do
    # Get latest step from logs
    LATEST=$(tail -200 /tmp/training.log | grep "^Step" | tail -1)

    if [ -z "$LATEST" ]; then
        echo "[$(date +%H:%M:%S)] Waiting for training logs..."
        sleep 30
        continue
    fi

    STEP=$(echo "$LATEST" | awk -F'/' '{print $1}' | awk '{print $2}')
    LOSS=$(echo "$LATEST" | grep -oE "loss=[0-9.]+" | cut -d= -f2)
    SPEED=$(echo "$LATEST" | grep -oE "[0-9.]+st/s" | cut -d's' -f1)
    ETA=$(echo "$LATEST" | grep -oE "ETA [0-9]+min" | cut -d' ' -f2)

    # Milestone alerts
    if [ "$STEP" -gt "$PREV_STEP" ]; then
        if (( $(echo "$STEP >= 200" | bc) )) && (( $(echo "$PREV_STEP < 200" | bc) )); then
            echo "✅ Step 200 reached! First checkpoint saved."
        fi
        if (( $(echo "$STEP >= 2000" | bc) )) && (( $(echo "$PREV_STEP < 2000" | bc) )); then
            echo "🎯 Step 2000 reached! First evaluation checkpoint."
            echo "   Checkpoint: checkpoints/step_2000.pt (or later)"
            echo "   Evaluation will begin..."
        fi
        if (( $(echo "$STEP >= 10000" | bc) )) && (( $(echo "$PREV_STEP < 10000" | bc) )); then
            echo "📊 Step 10000 reached! 60% of training complete."
        fi

        # Regular progress update every 100 steps or 2 min
        if (( $((STEP % 100)) == 0 )); then
            PERCENT=$((STEP * 100 / 16547))
            echo "[$(date +%H:%M:%S)] Step $STEP/16547 ($PERCENT%) | Loss: $LOSS | Speed: ${SPEED}st/s | ETA: $ETA"
        fi

        PREV_STEP=$STEP
    fi

    sleep 120

    # Check if training finished
    if grep -q "Goal achieved\|accuracy.*100" /tmp/training.log 2>/dev/null; then
        echo "🏆 TRAINING COMPLETED - GOAL ACHIEVED!"
        break
    fi
done
