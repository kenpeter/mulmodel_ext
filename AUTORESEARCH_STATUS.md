# Autoresearch Loop Status

## Goal
Improve model generalization from 41% average robustness to 70%+

## Iteration 1 Status
**Started:** 2026-04-13 10:52 UTC  
**Status:** In Progress - Training lightweight model (hard-loss-only due to GPU memory constraints)

### Changes Made
1. Reduced early-stop threshold: 100% → 90%  
2. Adjusted loss weights: soft 0.7→0.6, hard 0.3→0.4
3. Switched to lightweight training (hard-loss-only) to fit in GPU memory

### Rationale
- **Overfitting analysis**: Phase 3 validation showed model memorized eval pool distribution patterns
- **Root cause**: Model saw only 528 eval pool problems repeatedly; needed exposure to full 2110 training set
- **Hyperparameter changes**: 
  - Lower early-stop prevents premature convergence and allows longer training
  - Increased hard-loss weight may emphasize actual code correctness over teacher imitation
- **GPU constraints**: Fresh training with full distillation (teacher + student) exceeded 11.6GB VRAM; lightweight approach uses hard-loss only

### Expected Outcome
- Longer training exposure should improve generalization
- If successful, will measure via eval_robustness.py (5 rotations, 20 problems each)
- Target: Average accuracy > 70% (from baseline 41%)

### Next Steps
1. Monitor lightweight training completion
2. Run eval_robustness.py on final.pt
3. If improved: Keep changes, proceed to Iteration 2
4. If degraded or same: Revert and try different hyperparameters
