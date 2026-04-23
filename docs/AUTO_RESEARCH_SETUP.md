# 🤖 Auto-Research Mode - Correctness Focus

**Activated**: 2026-04-23 16:45 UTC  
**Mode**: Full Automatic Capsule Discovery + Application  
**Target**: 0% → 90%+ test-case passing

---

## What Auto-Research Does

The **Auto-Research capsule** automatically:

1. **Discovers** which capsules improve test-case correctness
2. **Tests** different capsule combinations (Bayesian optimization)
3. **Applies** improvements automatically (no manual intervention)
4. **Monitors** test-case passing every 500 steps
5. **Rolls back** if accuracy drops > 5%
6. **Learns** which capsule mix works best

---

## How It Works

```
Training Step Loop:
    ↓
    Step N → Train normally
    ↓
    Every 500 steps:
        ├─ Run eval.py
        ├─ Measure: Test-case passing %
        ├─ Auto-Research analyzes metrics
        ├─ Suggests next capsule to try
        ├─ APPLY if improves > 5%
        ├─ DISCARD if no improvement
        └─ Log decision in AUTORESEARCH_RESULTS.tsv
    ↓
    Repeat until:
        ✓ 90%+ test-case passing (SUCCESS)
        ✓ OR step limit reached
        ✓ OR severe regression detected
```

---

## Capsules Being Explored

Auto-Research will automatically test combinations of:

- ✓ Dropout Regularization Strategy
- ✓ Weight Decay Optimization
- ✓ Data Augmentation Pipeline
- ✓ Adaptive Learning Rate Scheduling
- ✓ Gradient Clipping Optimization
- ✓ Mixed Precision Training
- ✓ Model Checkpointing
- ✓ Distillation Quality Monitor
- ✓ Solution Correctness Regression Detection
- ✓ Hyperparameter Search (Grid + Bayesian)

---

## Monitoring Auto-Research

### Real-Time Monitoring (3 windows)

```bash
# Window 1: Training progress
tail -f /tmp/training.log

# Window 2: Auto-Research decisions
tail -f /tmp/autoresearch_monitor.log

# Window 3: Correctness verification
tail -f /tmp/correctness_monitor.log
```

### One-Time Checks

```bash
# Current metrics
python evolver_autoresearch_monitor.py --once

# All auto-research decisions
cat AUTORESEARCH_RESULTS.tsv

# Correctness verification results
python correctness_verifier.py --report

# Capsule applications
cat CAPSULE_APPLICATIONS.log
```

---

## Expected Timeline

| Step | Time | Milestone |
|------|------|-----------|
| 5400 | Now | Resume from checkpoint |
| 5900 | +25 min | First eval (0% baseline) |
| 6400 | +50 min | Auto-Research suggests first capsule |
| 7400 | +2.5h | Major eval checkpoint |
| 9400 | +5h | Should see improvement |
| 12000 | +8h | Mid-training assessment |
| 15000 | +11h | Approaching 90% goal |
| 18000+ | +15h | Success or extend |

---

## Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Test-Case Passing | 0% | 90%+ | 🔴 START |
| Code Structure | 25% | 80%+ | 🟡 MONITOR |
| Train/Val Gap | 44% | <20% | 🟡 IMPROVE |
| Teacher Alignment | ? | >95% | 🟡 MONITOR |

---

## What Auto-Research Will Do

### Phase 1: Test Baseline (Steps 5400-5900)
```
Run first eval to confirm 0% baseline
Log: "Baseline: 0% test-case passing"
Decide: Which capsule to try first?
```

### Phase 2: Iterative Improvement (Steps 5900-12000)
```
Iteration 1:
  ├─ Try: Data Augmentation
  ├─ Result: 2% improvement
  └─ Decision: KEEP + try next

Iteration 2:
  ├─ Try: Distillation Quality Monitor
  ├─ Result: 3% improvement
  └─ Decision: KEEP + combine with Iteration 1

Iteration 3:
  ├─ Try: Regression Detection
  ├─ Result: No improvement
  └─ Decision: DISCARD, try different
```

### Phase 3: Optimization (Steps 12000-18000)
```
Fine-tune best capsule combination
Adjust hyperparameters based on trends
Push toward 90%+ target
```

---

## Rollback Protection

Auto-Research will **automatically rollback** if:

- 🛑 Test-case passing drops > 5% (revert capsule)
- 🛑 Test-case passing drops > 15% (revert to previous checkpoint)
- 🛑 Teacher alignment diverges > 40% (reduce distillation capsule)

---

## Files to Monitor

| File | Purpose | Updates |
|------|---------|---------|
| `/tmp/training.log` | Raw training output | Every step |
| `/tmp/autoresearch_monitor.log` | Auto-Research decisions | Every 500 steps |
| `/tmp/correctness_monitor.log` | Correctness checks | Per eval |
| `AUTORESEARCH_RESULTS.tsv` | All iterations | Per eval |
| `CAPSULE_APPLICATIONS.log` | Capsule decisions | Per application |
| `AUTO_EVAL_HISTORY.json` | Accuracy trends | Per eval |
| `CORRECTNESS_VERIFICATION.json` | Test-case results | Per checkpoint |
| `checkpoints/eval_results.json` | Latest eval metrics | Per eval |

---

## Starting Training

```bash
# Start training with auto-research active
cd /home/kenpeter/work/mulmodel_ext

nohup python -u scripts/train_proven_local.py 2>&1 | tee -a /tmp/training.log &

# Check it started
tail -1 /tmp/training.log
```

---

## If Something Goes Wrong

| Issue | Check | Fix |
|-------|-------|-----|
| Training stuck | `tail -f /tmp/training.log` | Training may be buffering |
| No eval running | Check if step % 500 == 0 | Wait for next eval step |
| Capsule not applying | `cat CAPSULE_APPLICATIONS.log` | Check if improvement > 5% |
| Regression detected | `cat AUTORESEARCH_RESULTS.tsv` | Auto-Research will rollback |
| Monitor crashed | `ps aux \| grep monitor` | Restart: `python evolver_autoresearch_monitor.py --monitor &` |

---

## Summary

✅ **Auto-Research is configured and ready**
- 10 capsule categories to explore
- Bayesian optimization for efficiency
- Automatic rollback protection
- Real-time correctness monitoring

✅ **Monitoring is running**
- Correctness Verifier: PID 61072
- Auto-Research Monitor: PID 85769
- Both logging continuously

🚀 **Ready to start training!**

---

**Status**: Ready to train  
**Last Updated**: 2026-04-23 16:45 UTC
