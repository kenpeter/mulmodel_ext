# Autoresearch Iteration 1 - Hyperparameter Optimization

## Objective  
Improve model generalization from 41% average robustness (baseline) to 70%+

## Problem Statement
- Model achieved 100% accuracy on rotating 20-problem eval set at step 4000
- Phase 3 validation revealed severe overfitting: only 41% average on robustness test (5 different problem rotations)
- Root cause: Model memorized patterns from 528-problem eval pool rather than learning to solve arbitrary LeetCode problems
- GPU memory constraints prevent standard retraining approaches

## Iteration 1 Approach
**Starting:** 2026-04-13 10:52 UTC  
**Status:** Training in progress with teacher distillation

### Hyperparameter Changes
1. **Early-stop threshold**: 100% → 90%
   - Rationale: Allows training to continue beyond 4000 steps, exposing model to more training data
   - Expected benefit: Longer training = more data diversity = better generalization

2. **Loss weight adjustment**: soft_weight 0.7 → 0.6, hard_weight 0.3 → 0.4
   - Rationale: Emphasize supervised learning (hard loss) over teacher imitation (soft loss)
   - Expected benefit: Model learns actual code correctness rather than mimicking teacher outputs

3. **GPU memory optimization**: 
   - Reduced max_length from 96 to 32 (initial attempts)
   - Added aggressive cache clearing
   - Final: Using full distilled training with max_length=32

### Metric
**Robustness test** (eval_robustness.py):
- Evaluates model on 5 different 20-problem rotations (seeds: 42, 123, 456, 789, 999)
- Measures: Accuracy per rotation, average, min, max, range
- Baseline: 41% average (40%, 65%, 55%, 15%, 30%)
- Target: 70%+ average

### Training Details
- Dataset: 2110 training problems, 528 eval pool
- Early-stop: Step 2000 eval (guard: ≥85%), Step N eval (threshold: 90%)
- Max context length: 32 tokens (reduced for memory)
- Teacher model: Qwen 3.5 0.8B (8-bit quantized)
- Student model: Qwen 2.5 0.5B

### Timeline
- 10:52 - Initial hyperparameter changes + lightweight training attempts (hit OOM due to pre-existing GPU state)
- 11:00 - Cleared GPU memory (254 MiB → 11.6 GiB available)
- 11:05 - Restarted distilled training with clean GPU state (in progress)

### Expected Completion
- Training: ~5-6 hours (until 6:00 AM UTC time limit)
- Robustness eval: ~30 minutes (5 rotations × 20 problems)
- Total iteration time: ~6 hours

### Success Criteria
✓ Training completes without OOM
✓ Step 2000 eval ≥ 85% (guard condition)
✓ Final robustness average > 41% (improvement over baseline)
✓ If average ≥ 70%: Keep changes, commit to repo
✓ If average < 41%: Revert changes, try different hyperparameters

