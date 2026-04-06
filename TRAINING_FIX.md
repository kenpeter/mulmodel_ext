# Root Cause: Why Model Only Learned 20 Problems

## The Problem
Training script was using **public HuggingFace dataset** (`justindal/leetcode-python-dataset`), NOT your local 2638 high-quality problems.

**Old approach** (`scripts/train_kda_muon.py`):
```python
ds = load_dataset("justindal/leetcode-python-dataset", split=split)  # ❌ PUBLIC DATASET
```

This means:
- ✅ Model trained on public dataset
- ✅ Model evaluated on same public dataset (20 problems)
- ✅ Once it memorized those 20 → early_stop_accuracy triggered → training stopped
- ❌ Never saw your local 2638 high-quality problems
- ❌ No train/test split → overfitting to same 20 eval problems

## The Solution
**New approach** (`scripts/train_local_data.py`):

```python
# 1. Load local 2638 problems
with open(local_data_path, 'r') as f:
    all_items = [json.loads(line) for line in f]

# 2. Split 80/20
train_indices = all_indices[:split_idx]      # 2110 problems
eval_indices = all_indices[split_idx:]        # 528 problems

# 3. Rotate eval set
sample_indices = random.sample(eval_indices, eval_size=20)  # Different 20 each time
```

## Key Improvements
✅ **Uses local 2638 high-quality problems** (not public dataset)
✅ **80/20 train/test split** (2110 train, 528 eval pool)
✅ **Rotating eval sets** — samples 20 random problems each eval (prevents memorization)
✅ **Prevents overfitting** — model can't just memorize same 20 problems
✅ **Resume from checkpoint** — continues from last step
✅ **Diverse evaluation** — tests generalization across many different problems

## How to Run
```bash
# New training with local data + rotation
python scripts/train_local_data.py

# This will:
# 1. Load 2638 local problems
# 2. Split into 2110 train / 528 eval
# 3. Train normally
# 4. Every 1000 steps: evaluate on rotating random 20 problems from eval pool
# 5. Resume from checkpoint if one exists
```

## Expected Behavior
- **Before**: Train to 100% on same 20 problems, then stop (model memorized)
- **After**: Train across diverse 528 eval problems, with rotating sets each checkpoint
- **Result**: Better generalization to arbitrary problems

## Files
- `solve.py` — inference (uses final checkpoint)
- `scripts/train_local_data.py` — NEW: proper training with local data + rotation
- `scripts/train_kda_muon.py` — OLD: used public dataset (keep for reference)
