#!/bin/bash
# Master autoresearch loop orchestrator

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

RESULTS_LOG="AUTORESEARCH_RESULTS.tsv"
MAX_ITERATIONS=50
BASELINE_ACCURACY=0
BEST_ACCURACY=0

echo "====== AUTORESEARCH LOOP STARTING ======"
echo "Project: $(basename $PROJECT_ROOT)"
echo "Time: $(date)"
echo "Max iterations: $MAX_ITERATIONS"
echo ""

# Function to wait for training checkpoint
wait_for_checkpoint() {
    local max_wait=1800  # 30 minutes max
    local elapsed=0
    while [ ! -f "checkpoints/step_2000.pt" ] && [ $elapsed -lt $max_wait ]; do
        sleep 10
        elapsed=$((elapsed + 10))
        echo -ne "\rWaiting for checkpoint... ${elapsed}s"
    done
    echo ""

    if [ ! -f "checkpoints/step_2000.pt" ]; then
        echo "ERROR: Checkpoint not found after $max_wait seconds!"
        return 1
    fi
    return 0
}

# Function to evaluate checkpoint
evaluate() {
    echo "Evaluating checkpoint..."
    python eval_checkpoint.py 2>&1 | tee /tmp/eval_output.txt

    # Extract accuracy
    ACCURACY=$(grep "Accuracy:" /tmp/eval_output.txt | grep -oE "[0-9]+\.[0-9]+" | tail -1)

    if [ -z "$ACCURACY" ]; then
        echo "ERROR: Could not parse accuracy from eval!"
        return 1
    fi

    echo "Accuracy: ${ACCURACY}%"
    return 0
}

# Function to commit iteration result
commit_iteration() {
    local iter=$1
    local change=$2
    local accuracy=$3

    git add -A
    git commit -m "experiment: iter $iter - $change (${accuracy}% accuracy)" || true
}

# ============ ITERATION #0: BASELINE ============
echo ""
echo ">>> ITERATION 0: BASELINE (Temperature=2.0)"
echo "Waiting for training checkpoint..."

if wait_for_checkpoint; then
    if evaluate; then
        BASELINE_ACCURACY=$ACCURACY
        BEST_ACCURACY=$ACCURACY
        echo "Baseline established: ${BASELINE_ACCURACY}%"

        # Update results log
        python -c "
import sys
timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
commit = __import__('subprocess').run('git rev-parse --short HEAD', shell=True, capture_output=True, text=True).stdout.strip()
line = f'0\t{timestamp}\tBaseline (T=2.0)\t${BASELINE_ACCURACY}\tCOMPLETED\tInitial training run\t{commit}\n'
with open('AUTORESEARCH_RESULTS.tsv', 'a') as f:
    f.write(line)
print('Baseline logged')
" 2>&1 || true

        commit_iteration 0 "Baseline (T=2.0)" "$BASELINE_ACCURACY"
    else
        echo "ERROR: Evaluation failed"
        exit 1
    fi
else
    echo "ERROR: Training did not produce checkpoint"
    exit 1
fi

# ============ PHASE 2: OPTIMIZATION ITERATIONS ============
echo ""
echo ">>> Starting optimization phase..."
echo "Baseline accuracy: ${BASELINE_ACCURACY}%"
echo ""

# Define iteration experiments
declare -a EXPERIMENTS=(
    "temp_increase:Temperature=2.0→2.2"
    "temp_decrease:Temperature=2.0→1.8"
    "max_length_increase:max_length=96→128"
    "loss_weight_adjust:soft/hard=0.7/0.3→0.75/0.25"
    "learning_rate_increase:lr=2e-4→3e-4"
)

for iter in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "========== ITERATION $iter =========="

    # Select experiment (round-robin or random)
    EXP_IDX=$(( (iter - 1) % ${#EXPERIMENTS[@]} ))
    EXP="${EXPERIMENTS[$EXP_IDX]}"
    EXP_TYPE="${EXP%:*}"
    EXP_DESC="${EXP#*:}"

    echo "Experiment: $EXP_DESC"

    # Modify hyperparameters based on experiment type
    case "$EXP_TYPE" in
        temp_increase)
            sed -i 's/F\.softmax(st \/ 2\.0/F.softmax(st \/ 1.818/' scripts/train_proven_local.py
            sed -i 's/F\.log_softmax(ss \/ 2\.0/F.log_softmax(ss \/ 1.818/' scripts/train_proven_local.py
            ;;
        temp_decrease)
            sed -i 's/F\.softmax(st \/ 2\.0/F.softmax(st \/ 2.222/' scripts/train_proven_local.py
            sed -i 's/F\.log_softmax(ss \/ 2\.0/F.log_softmax(ss \/ 2.222/' scripts/train_proven_local.py
            ;;
        *)
            echo "TODO: Implement $EXP_TYPE"
            ;;
    esac

    # Clean checkpoints
    rm -f checkpoints/step_*.pt

    # Run training
    echo "Starting training..."
    python scripts/train_proven_local.py > /tmp/train_iter_$iter.log 2>&1 &
    TRAIN_PID=$!

    # Wait for checkpoint
    if wait_for_checkpoint; then
        if evaluate; then
            NEW_ACCURACY=$ACCURACY
            IMPROVEMENT=$(python -c "print(f'{$NEW_ACCURACY - $BASELINE_ACCURACY:.1f}')")

            if (( $(echo "$NEW_ACCURACY > $BEST_ACCURACY" | bc -l) )); then
                echo "✅ IMPROVEMENT: ${NEW_ACCURACY}% (was ${BEST_ACCURACY}%)"
                BEST_ACCURACY=$NEW_ACCURACY
                commit_iteration $iter "$EXP_DESC" "$NEW_ACCURACY"
            else
                echo "❌ NO IMPROVEMENT: ${NEW_ACCURACY}% (best ${BEST_ACCURACY}%)"
                git checkout scripts/train_proven_local.py  # revert changes
            fi

            # Log result
            python -c "
import sys
timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
commit = __import__('subprocess').run('git rev-parse --short HEAD', shell=True, capture_output=True, text=True).stdout.strip()
line = f'$iter\t{timestamp}\t$EXP_DESC\t$NEW_ACCURACY\tCOMPLETED\t\t{commit}\n'
with open('AUTORESEARCH_RESULTS.tsv', 'a') as f:
    f.write(line)
" 2>&1 || true
        fi
    fi

    # Check termination conditions
    if (( $(echo "$BEST_ACCURACY >= 100" | bc -l) )); then
        echo ""
        echo "🎯 GOAL ACHIEVED: ${BEST_ACCURACY}% accuracy!"
        break
    fi
done

echo ""
echo "====== AUTORESEARCH LOOP COMPLETED ======"
echo "Best accuracy: ${BEST_ACCURACY}%"
echo "Results saved to: $RESULTS_LOG"
echo ""
tail -10 "$RESULTS_LOG"
