# Mulmodel Distillation — LeetCode Code Generation

Autonomous model distillation pipeline for training a student model to generate Python solutions to LeetCode problems by learning from a teacher model (Qwen 3.5 8B).

## Status

⚠️ **CHECKPOINT REGRESSION DETECTED (2026-04-12)**
- April 6: Achieved **100% accuracy** on 20-problem eval set (commit b51479f)
- April 12: Autonomous loop restarted with wrong checkpoint baseline
- Current state: `checkpoints/final.pt` = **40% accuracy only**
- **Root cause:** Training resumed from degraded `step_200.pt` instead of April 6 working model
- **Action required:** Restore April 6 checkpoint before continuing autonomous optimization

See `debug/FINDINGS.md` for full investigation.

## Project Structure

```
mulmodel_ext/
├── README.md                          # This file
├── scripts/
│   └── train_proven_local.py          # Main training script (2638 local LeetCode problems)
├── model/
│   ├── config.py                      # StudentModel configuration
│   ├── student.py                     # Student model (320M params, 96-token distillation)
│   └── optimizer.py                   # MuonClip optimizer
├── checkpoints/                       # Model checkpoints (1.5GB each, every 200 steps)
├── eval_checkpoint.py                 # Quick eval script (20-problem rotating set)
├── solve.py                           # Solution generator (main entry point)
├── AUTORESEARCH_*.md                  # Autonomous loop documentation
└── debug/
    └── FINDINGS.md                    # Checkpoint regression investigation
```

## Model Architecture

**Student Model:**
- Base: Qwen 2.5 0.5B (quantized to bfloat16)
- Params: ~320M effective
- Context length: 96 tokens (can increase to 128+ for robustness)
- Training: Distillation loss (70% KL-div from teacher, 30% supervised)

**Teacher Model:**
- Qwen 3.5 8B Claude-4.6 Reasoning Distilled
- Generates target logits for knowledge transfer
- Frozen during student training

## Training

### Data

- **Source:** 2638 high-quality LeetCode problems (local)
- **Split:** 80% train (2110), 20% eval pool (528)
- **Eval method:** Rotating 20-problem samples from eval pool
- **Format:** Problem description + starter code → Python solution

### Configuration

**Hyperparameters** (`scripts/train_proven_local.py`):
```python
max_length = 96              # Token sequence length
batch_size = 1              # Per-step batch
learning_rate = 2e-4        # Cosine schedule with warmup
grad_accum = 1              # No gradient accumulation
max_steps = 50000           # Or stop at 100% accuracy
eval_every = 2000           # Evaluation checkpoint frequency
save_every = 200            # Model checkpoint frequency

# Distillation settings
temperature = 2.0           # Knowledge softness (KL divergence)
soft_weight = 0.7           # Distillation loss weight
hard_weight = 0.3           # Supervised loss weight
early_stop_accuracy = 100.0 # Auto-stop when reached
```

### Running Training

```bash
# Start fresh training
python scripts/train_proven_local.py

# Automatically resumes from latest checkpoint in checkpoints/
# Saves progress every 200 steps
# Evaluates every 2000 steps
```

**Expected timeline:**
- Steps 0-50: Warmup + initialization
- Steps 50-2000: Ramp-up phase
- Steps 2000+: Stable training (loss ~2-5 range)
- Eval checkpoints: Steps 2000, 4000, 6000, 8000, 10000...
- Early stop: When eval accuracy ≥ 100%

**Resource requirements:**
- GPU: RTX 4070 Ti (24GB VRAM, tight)
- Memory: ~12GB VRAM occupied during training
- Time: ~20 minutes per 8K training steps
- Disk: ~1.5GB per checkpoint (saved every 200 steps)

## Evaluation

### Quick Eval (Latest Checkpoint)

```bash
python eval_checkpoint.py
```

Output: Accuracy % on 20 random problems

**Evaluation criteria:** Solution contains `def `, `class `, or `return ` (basic syntax check)

### Batch Checkpoint Testing

```bash
# Test multiple checkpoints to find regression point
python debug_checkpoints.py
```

## Solution Generation

### Generate Solution

```python
from solve import generate_solution

problem = """
You are given a list of numbers. Return the sum.
"""

solution = generate_solution(problem)
print(solution)
# Output:
# def solve(nums):
#     return sum(nums)
```

See `solve.py` for integration details.

## Autonomous Loop (WIP)

**Goal:** Optimize hyperparameters automatically to maximize eval accuracy

**Strategy:**
1. **Phase 1 (Baseline):** Establish current accuracy with Temperature=2.0
2. **Phase 2a (If 100%):** Test robustness improvements:
   - Increase sequence length: 96 → 128
   - Tune temperature: 2.0 → 1.8 or 2.2
   - Adjust loss weights: 0.7:0.3 variants
3. **Phase 3 (Validation):** Extended eval on 50+ problems to confirm generalization

**Status:** Setup complete, but **baseline checkpoint was lost** — see Debug Findings below.

**Configuration files:**
- `AUTORESEARCH_LOOP.md` — Loop documentation
- `LOOP_STRATEGY.md` — Hyperparameter exploration strategy
- `AUTORESEARCH_RESULTS.tsv` — Iteration results log

## Known Issues

### 🔴 Critical: Checkpoint Regression (2026-04-12)

**Problem:**
- April 6 achieved 100% accuracy
- April 12 autonomous loop started training from wrong checkpoint
- Current model: only 40% accuracy

**Root cause:**
- Resumed training from `step_200.pt` (old, weak checkpoint)
- Did not preserve April 6 working model
- 16K+ training steps unable to recover

**Impact:**
- Cannot continue optimization from 100% baseline
- Need to restore April 6 checkpoint or retrain from scratch

**Solution:**
1. Recover April 6 checkpoint from git history
2. OR restart training with fresh teacher initialization
3. Fix training script to properly preserve best checkpoints

### ⚠️ Medium: Eval Checkpoints Not Logging

**Symptom:**
- Training completes but no eval results in `/tmp/training.log`
- Early-stop condition (line 337) may not be executing

**Status:** Investigating — likely eval runs but output not captured to logs

## Development Roadmap

- [ ] **Restore April 6 checkpoint** (critical blocker)
- [ ] Implement proper early-stop with best-checkpoint saving
- [ ] Fix eval logging to `/tmp/training.log`
- [ ] Phase 2a: Sequence length optimization (96 → 128)
- [ ] Phase 2b: Temperature tuning (2.0 → 1.8/2.2)
- [ ] Extended eval: 50+ problem set validation
- [ ] Benchmark on external LeetCode problems
- [ ] Performance optimization: reduce VRAM, faster inference

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python scripts/train_proven_local.py` | Start/resume training |
| `python eval_checkpoint.py` | Quick eval (20 problems) |
| `python debug_checkpoints.py` | Find degradation point (slow) |
| `python solve.py <problem>` | Generate solution |
| `git log --oneline` | View training history |
| `tail -f /tmp/training.log` | Monitor training in real-time |

## Results

### Previous Best (April 6)
- **Accuracy:** 100% (20/20 LeetCode problems)
- **Model:** `checkpoints/final.pt` (1.5GB, 96-token context)
- **Training:** ~8000 steps with Temperature=2.0
- **Commit:** b51479f "Autoresearch success: Temperature=2.0 achieves 100% eval accuracy"

### Current Status (April 12)
- **Accuracy:** 40% (degraded due to checkpoint regression)
- **Model:** `checkpoints/final.pt` (1.5GB, wrong weights)
- **Training:** 16K+ steps from poor baseline
- **Issue:** Lost reference to April 6 model

## Debugging

Full investigation in `debug/FINDINGS.md`:
- Timeline of events
- Evidence analysis
- Root cause explanation
- Recovery options

Run debug with: `/autoresearch:debug --scope checkpoints Symptom: "checkpoint regression from 100% to 40%"`

## Contributing

- Add new hyperparameter experiments to `LOOP_STRATEGY.md`
- Document findings in `AUTORESEARCH_RESULTS.tsv`
- Create git commit for each iteration: `experiment: [change] ([accuracy]%)`

## Contact

Questions about training, eval, or autonomous loop? Check:
1. `STATUS.md` — current run state
2. `AUTORESEARCH_GUIDE.md` — loop mechanics
3. `/tmp/training.log` — live training output
4. `debug/FINDINGS.md` — investigation results
