# Autoresearch Loop: Improve Accuracy on Expanded Eval Set

## Configuration
- **Goal:** Improve accuracy further on expanded eval set (50 problems)
- **Scope:** `scripts/train_proven_local.py` hyperparameters
- **Metric:** Eval accuracy % on 50 problems (higher = better)
- **Verify:** `python eval_checkpoint.py 2>&1 | grep "Accuracy"`
- **Guard:** Model loads without crash + accuracy >= 50%
- **Iterations:** Unlimited

## Baseline (Iteration #0)
- **Checkpoint:** `checkpoints/final.pt` (trained with Temperature=2.0)
- **Eval set size:** 50 random problems (seed=42)
- **Status:** Running initial evaluation...
- **Expected:** ~100% on expanded set (or lower if 20→50 problem diversity matters)

## Iteration Log

### Iteration 1: (pending)

## Tunable Hyperparameters
| Parameter | Current | Exploration Range | Impact |
|-----------|---------|------------------|--------|
| `max_length` | 96 | 96-256 | Code quality vs memory |
| `batch_size` | 1 | 1-4 | Gradient stability vs speed |
| `lr` | 2e-4 | 1e-4 to 5e-4 | Learning speed vs stability |
| `grad_accum` | 1 | 1-4 | Effective batch size |
| `temperature` | 2.0 | 1.5-2.5 | Knowledge transfer softness |
| `soft_loss_weight` | 0.7 | 0.6-0.8 | Distillation emphasis |
| `eval_every` | 2000 | 1000-2000 | Feedback frequency |

## Strategy
1. Test baseline on 50-problem expanded set
2. If accuracy < 100%, tune hyperparameters to improve
3. If accuracy = 100%, explore increasing sequence length or other optimizations
4. Iterate until reaching maximum achievable accuracy
