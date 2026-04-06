# Training on Local 2638 Problems with Rotating Eval

## What Changed
✅ **Before:** Trained on public dataset (20 problems) → memorized them
✅ **After:** Trains on YOUR 2638 high-quality local problems → generalizes across diverse problems

## Script to Use
```bash
# Start training
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_proven_local.py

# Monitor progress
watch "ls -lht checkpoints/step_*.pt | head -5"

# Test the model
python solve.py
```

## Training Status
- **Data:** 2110 train samples | 528 eval pool | 20 rotating eval problems
- **Speed:** ~2-3 minutes per 200 training steps
- **Checkpoints:** Saved every 200 steps
- **Eval:** Every 2000 steps (rotating eval set)
- **Estimated Time:** ~16 hours for full training (15,000+ steps)

## Dataset Split
- **Train:** 80% = 2110 problems from `/home/kenpeter/work/data/high_quality_leetcode/train.jsonl`
- **Eval:** 20% = 528 problems (samples 20 random per checkpoint for rotation)
- **Rotating:** Different 20 problems tested each eval cycle

## Key Features
1. **Large Dataset**: 2638 diverse LeetCode problems
2. **Rotating Eval**: Prevents overfitting to same 20 problems
3. **Memory Optimized**: Uses `PYTORCH_ALLOC_CONF=expandable_segments:True`
4. **Proven Architecture**: Based on successful train_kda_muon.py
5. **Resume Support**: Automatically resumes from latest checkpoint

## Model Configuration
- **Architecture:** StudentModel with Kimi Delta Attention (KDA)
- **Optimizer:** MuonClip with momentum=0.95
- **Learning Rate:** 2e-4 (cosine annealing with warmup)
- **Loss:** 0.7 * KL_divergence(temp=1.5) + 0.3 * CrossEntropy
- **Precision:** bfloat16 mixed precision
- **Gradient Accumulation:** 2 steps
- **Max Sequence Length:** 512 tokens

## Autoresearch Loop (Optional)
To automatically optimize hyperparameters:

1. Run training baseline
2. Extract eval accuracy from first checkpoint eval
3. Try variations (lr, loss weights, temperature)
4. Keep improvements, discard regressions
5. Repeat until max iterations or goal met

See `AUTORESEARCH_SETUP.md` for full autoresearch configuration.

## Files
- `scripts/train_proven_local.py` - Main training script
- `solve.py` - Inference script (unchanged)
- `checkpoints/step_*.pt` - Training checkpoints (every 200 steps)
- `checkpoints/final.pt` - Final model after training completes
- `/home/kenpeter/work/data/high_quality_leetcode/train.jsonl` - Training data (2638 problems)

## Next Steps
1. Let training run to completion (or until interrupted)
2. Use `solve.py` to test the final model on any LeetCode problem
3. Check `checkpoints/eval_step_*.json` for eval metrics
4. (Optional) Use autoresearch to find optimal hyperparameters

## Expected Results
Training on the full diverse dataset should improve:
- **Generalization:** Model solves more types of problems (not just the 20 test ones)
- **Code Quality:** Better solutions to arbitrary problems
- **Robustness:** Works with different problem formats and complexities
