# Autoresearch Continuation: Distill Model → Improve Accuracy on Expanded Set

## Phase: Model Quality Refinement (Iterations: Unlimited)

### Configuration
| Setting | Value |
|---------|-------|
| **Goal** | Improve accuracy further on expanded LeetCode eval set |
| **Scope** | `scripts/train_proven_local.py` hyperparameters |
| **Metric** | Eval accuracy % from `python eval_checkpoint.py` |
| **Metric Direction** | Higher is better (100% = perfect) |
| **Guard** | Model loads + eval succeeds + accuracy >= 50% |
| **Verify Command** | `bash get_accuracy.sh` |

### Baseline (Iteration #0)
- **Checkpoint**: `checkpoints/final.pt` (temperature=2.0, step 2000)
- **Previous Best**: 100% on 20 LeetCode problems
- **Current Baseline**: (measuring...)
- **Status**: Initial eval running...

## Iteration Log

| Iter | Type | Change | Accuracy | Status | Notes |
|------|------|--------|----------|--------|-------|
| 0 | baseline | — | (measuring) | PENDING | Establish current performance on fresh eval |
| 1 | exp | (TBD) | | PENDING | |

## Hyperparameter Exploration Strategy

### Phase 1: Validate Baseline (1 iteration)
- Run eval on current best model
- If accuracy = 100% → Phase 2a (push accuracy higher)
- If accuracy < 100% → Phase 2b (recover accuracy)

### Phase 2a: If Baseline = 100%
Primary goals: Improve model robustness/generalization
1. **Increase sequence length** (96 → 128 or 192)
   - May improve code generation quality
   - Memory tradeoff: grad_accum or batch_size reduction
2. **Adjust loss weights** (soft:hard 0.7:0.3)
   - Try 0.75:0.25 or 0.6:0.4
3. **Tune temperature** (2.0 → 2.2 or 1.8)
   - Test knowledge transfer boundaries

### Phase 2b: If Baseline < 100%
Primary goals: Recover accuracy to 100%
1. **Reduce learning rate** if unstable
2. **Increase eval frequency** for faster feedback
3. **Adjust temperature** to optimize knowledge transfer
4. **Increase max_steps** if underfitting

## Key Hyperparameters (train_proven_local.py)

```python
max_length = 96              # Line 70: token sequence length
batch_size = 1              # Line 71: training batch size
lr = 2e-4                   # Line 72: learning rate
grad_accum = 1              # Line 73: gradient accumulation
max_steps = 50000           # Line 74: training steps
eval_every = 2000           # Line 77: eval checkpoint frequency
# In KL divergence (lines 247-248):
temperature = 2.0           # Knowledge transfer softness
# Loss weighting (line 252):
soft_weight, hard_weight = 0.7, 0.3  # Distillation vs correctness
```

## Success Criteria
- **Primary**: Accuracy >= 100% on baseline eval set (20 problems)
- **Secondary**: Maintain accuracy while increasing sequence length or model robustness
- **Stretch**: Test on 50-100 problem set shows consistent performance

## Notes
- Each full training run: ~20 min (8K-10K steps on RTX 4070 Ti)
- Eval per problem: ~3-4 sec (includes generate + decode)
- Memory: 12GB VRAM (tight fit, no headroom)
