# Autoresearch Loop Setup: Train on Large Dataset with Rotating Eval

## Goal
**Maximize accuracy on diverse eval problems** by training on all 2638 local high-quality LeetCode problems.

## Configuration
- **Metric:** Eval accuracy % (higher is better)
- **Scope:** Training script logic (hyperparameters, loss weighting, optimizer settings)
- **Iterations:** Unlimited (until interrupted by user)
- **Guard:** None (only care about eval accuracy, not build passing)
- **Data:** 2110 train | 528 eval pool | 20 rotating problems per eval checkpoint

## Problem Identified
Old training used **public HuggingFace dataset** (`justindal/leetcode-python-dataset`), not your 2638 local problems. This led to overfitting on 20 specific problems.

## Solution Implemented
1. ✅ Created `scripts/train_proven_local.py` - trains on LOCAL 2638 problems
2. ✅ Implemented 80/20 split: 2110 train, 528 eval pool
3. ✅ Rotating eval sets: samples different 20 problems each checkpoint
4. ✅ Based on proven train_kda_muon.py architecture

## Current Status
- **Training Script:** `/home/kenpeter/work/mulmodel_ext/scripts/train_proven_local.py`
- **Data Source:** `/home/kenpeter/work/data/high_quality_leetcode/train.jsonl` (2638 problems)
- **Test Command:** `PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_proven_local.py`
- **Checkpoint Frequency:** Every 200 steps
- **Eval Frequency:** Every 2000 steps

## Hardware Constraint
RTX 4070 Ti (12GB VRAM) is tight for this setup:
- Teacher model: ~3.4GB
- Student model: ~1.5GB  
- Training activations: ~3-5GB
- Memory fragmentation: requires careful management

### Memory Optimization Applied
- `PYTORCH_ALLOC_CONF=expandable_segments:True` - reduces fragmentation
- `grad_accum=2` - gradient accumulation to reduce batch memory
- `batch_size=1` - minimum batch
- `eval_every=2000` - reduced from 1000 to save memory during training
- `max_length=512` - full sequences (necessary for code quality)

## Next Steps for Autoresearch Loop

### Iteration 1: Establish Baseline
1. Let current training run to completion (or first eval at step 2000)
2. Extract eval accuracy
3. Save as baseline_accuracy

### Iteration 2-N: Optimize Via Hyperparameter Search
Autoresearch should try (one per iteration):

**Loss Weighting:**
- Current: `loss = 0.7 * soft_loss + 0.3 * hard_loss`
- Try: 0.8/0.2, 0.6/0.4, 0.5/0.5

**Temperature:**
- Current: `temperature = 1.5` in KL divergence
- Try: 1.0, 2.0, 2.5, 1.2

**Learning Rate:**
- Current: `lr = 2e-4`
- Try: 1e-4, 3e-4, 5e-4, 1.5e-4

**Optimizer Momentum:**
- Current: `momentum = 0.95`
- Try: 0.9, 0.99, 0.999

**Warmup Steps:**
- Current: `warmup_steps = 50`
- Try: 25, 100, 200

### Decision Rules
- **KEEP:** New accuracy > baseline accuracy
- **DISCARD:** New accuracy <= baseline accuracy
- **MODIFY & RETRY:** If failed (OOM, crash) - try smaller eval_size or reduce max_length temporarily

## Usage

### Run Training Once
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_proven_local.py
```

### Monitor Progress
```bash
watch "ls -lht checkpoints/step_*.pt | head -5"
```

### Test Model After Training
```bash
python solve.py  # Interactive solver
```

## Expected Training Time
- 2000 steps with 2110 samples ≈ 1-2 full epochs
- ~3-4 hours on RTX 4070 Ti (estimated)
- Total training to 15,000+ steps ≈ 12-16 hours

## Files Created
- `scripts/train_proven_local.py` - Main training script
- `scripts/train_local_data.py` - Alternative (had memory issues)
- `TRAINING_FIX.md` - Root cause analysis
- `solve.py` - Inference script (unchanged)
- `checkpoints/step_*.pt` - Checkpoints (saved every 200 steps)
- `checkpoints/final.pt` - Final model at end of training
