# Autoresearch Autonomous Loop Strategy

## Baseline (Iteration #0) - NOW RUNNING
- **Training**: Fresh start with Temperature=2.0 (proven from previous)
- **Expected time**: ~20 minutes
- **ETA completion**: ~15:50 UTC
- **After training**: Auto-evaluate on 20 LeetCode problems

## Loop Phases (After Baseline Established)

### Phase 1: Validate Baseline Performance (Iter 1-3)
**Goal**: Confirm model trains correctly and reaches high accuracy

**Metrics to track**:
- Training loss progression
- Eval accuracy (target: >= 90%)
- Generation quality (no token loops)

**If baseline accuracy < 80%**: Increase `max_steps` or adjust `lr`
**If baseline accuracy >= 90%**: Proceed to Phase 2

### Phase 2: Improve via Hyperparameter Tuning (Iter 3+)

#### 2a) Sequence Length Exploration
| Change | Rationale | Risk |
|--------|-----------|------|
| max_length: 96 → 128 | Capture longer code sequences | Memory OOM |
| max_length: 96 → 64 | Faster training, less memory | Reduced code quality |

#### 2b) Temperature Optimization  
| Change | Rationale | Risk |
|--------|-----------|------|
| temperature: 2.0 → 2.2 | Softer targets, better transfer | Slower convergence |
| temperature: 2.0 → 1.8 | Sharper targets, faster learning | May overfit |

#### 2c) Loss Weight Tuning
| Change | Rationale | Risk |
|--------|-----------|------|
| soft:hard 0.7:0.3 → 0.75:0.25 | More distillation emphasis | Less direct loss supervision |
| soft:hard 0.7:0.3 → 0.6:0.4 | More supervised learning | Less knowledge transfer |

#### 2d) Learning Rate Sweep
| Change | Rationale | Risk |
|--------|-----------|------|
| lr: 2e-4 → 3e-4 | Faster convergence | Instability, divergence |
| lr: 2e-4 → 1e-4 | Safer, slower | May undershoot capacity |

### Phase 3: Consolidate & Validate (Final Iterations)
- Lock in best hyperparameters
- Extended eval on 50+ problems to confirm generalization
- Check for overfitting (train accuracy vs eval accuracy gap)

## Auto-Decision Rules for Loop

```
For each iteration:
  1. Train with modified hyperparameter
  2. Evaluate on 20 problems
  3. Compare accuracy to baseline
  
  IF accuracy > previous_best:
    → KEEP (commit as "experiment: <change> → +X% accuracy")
    → Try next experiment based on trend
  
  IF accuracy = previous_best ± 2%:
    → DISCARD (no significant change)
    → Try different hyperparameter
  
  IF accuracy < previous_best - 5%:
    → REVERT immediately
    → Try opposite direction next
  
  IF accuracy drops to 0 or training fails:
    → REVERT and investigate
    → May indicate memory issue or bad config
```

## Expected Behavior

**Scenario A: Baseline = 100%**
- Iterations 1-3: Test boundary conditions (increase seq_length)
- Iterations 4-6: Fine-tune secondary parameters
- Goal: Maintain 100% while improving generalization

**Scenario B: Baseline = 80-99%**
- Iterations 1-2: Increase max_steps or reduce learning rate
- Iterations 3-5: Temperature tuning
- Goal: Push to 100%

**Scenario C: Baseline < 80%**
- Iterations 1-3: Emergency debug (wrong checkpoint? config issue?)
- May need to extend max_steps significantly
- Consider model architecture vs just hyperparams

## Log Format

```
AUTORESEARCH_RESULTS.tsv:
iteration | timestamp | change | accuracy | status | notes | git_commit
```

Each successful iteration → git commit with `experiment:` prefix
Each failed iteration → git revert
All attempts logged in TSV regardless of outcome
