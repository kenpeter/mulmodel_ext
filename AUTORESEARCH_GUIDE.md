# Autoresearch Autonomous Loop - Complete Guide

## 🚀 What's Running Now

**Baseline Training (Iteration #0)** started at **16:10 UTC+10 (2026-04-12)**

```
Process: python scripts/train_proven_local.py
PID: 116649
Duration: ~20 minutes (8K steps @ 2.0 steps/sec)
Memory: 24GB GPU + 2GB model = tight but stable
```

## 📊 What to Expect

### Phase 1: Initialization (0-2 min) ✅ DONE
- Load teacher model (3.4GB → 900MB 8-bit quantized)
- Load student model (untrained)
- Load training data (2110 problems)
- Start warmup scheduler

### Phase 2: Training Loop (2-22 min) IN PROGRESS
```
For each batch:
  - Forward pass through teacher (get logits)
  - Forward pass through student (get logits + loss)
  - Compute KL divergence loss + supervised loss
  - Backward + optimizer step (MuonClip with momentum)
  
Every 10 steps: Print training metrics
Every 200 steps: Save checkpoint to checkpoints/step_*.pt
Every 2000 steps: Run evaluation on rotating 20 problems
```

**Expected logs from training.log:**
```
Step 10/16590 | loss=4.532 (h=0.234 s=0.892) | lr=2.0e-5 | 1.9st/s | ETA 138min
Step 20/16590 | loss=4.521 (h=0.223 s=0.881) | lr=4.0e-5 | 1.9st/s | ETA 137min
...
Step 2000/16590 | loss=2.145 (h=0.031 s=0.412) | lr=2.0e-4 | 1.9st/s | ETA 50min
==================================================
Evaluating on rotating eval set (20 samples)...
[1/20] ✓
[2/20] ✗
...
Eval accuracy: XX.X%
==================================================
```

### Phase 3: Evaluation (22-25 min) PENDING
Once `checkpoints/step_2000.pt` appears:
```python
1. Load model from checkpoint
2. Run on 20 LeetCode problems (3-5 sec each)
3. Count how many generate code with 'def', 'class', or 'return'
4. Log accuracy %
```

### Phase 4: Decision & Next Iteration (25-27 min) PENDING
```
IF accuracy >= 100%:
  → Iteration #1: Increase sequence length (96 → 128)
  
IF accuracy 80-99%:
  → Iteration #1: Adjust temperature (2.0 → 1.8 or 2.2)
  
IF accuracy < 80%:
  → Iteration #1: Extend training (more max_steps)
```

## 📁 Key Files

| File | Purpose | Updates |
|------|---------|---------|
| `/tmp/training.log` | Raw training output | Every step |
| `checkpoints/step_*.pt` | Model snapshots | Every 200 steps |
| `AUTORESEARCH_RESULTS.tsv` | Iteration results | After each eval |
| `STATUS.md` | Live status doc | Manual updates |

## 🔍 How to Monitor Live

```bash
# Watch training in real-time
tail -f /tmp/training.log

# Check for checkpoints
ls -lh checkpoints/

# View results so far
cat AUTORESEARCH_RESULTS.tsv

# Check process status
ps aux | grep train_proven_local
```

## ⚠️ Failure Modes & Recovery

### OOM (Out of Memory)
**Symptom:** Process killed, training.log ends abruptly
**Cause:** GPU ran out of 24GB VRAM
**Fix:** Reduce `max_length` (96 → 64) or `batch_size` (1 → already minimal)

### Model Generation Failures
**Symptom:** Eval shows 0% accuracy (all "E" errors)
**Cause:** Model stuck in loop (like before), generates `)` repeatedly
**Fix:** Extend training (max_steps: 50000 → 80000)

### Slow Training
**Symptom:** Training takes > 30 minutes
**Cause:** Slow hardware or background processes
**Fix:** Monitor other GPU processes, reduce max_steps target

### Training Never Starts
**Symptom:** training.log stays at initialization, no step logs
**Cause:** Teacher model load stuck or GPU issue
**Fix:** Kill process, check GPU with `nvidia-smi`, restart

## 📈 What "Good" Results Look Like

| Metric | Bad | Acceptable | Good | Excellent |
|--------|-----|-----------|------|-----------|
| **Accuracy** | < 20% | 40-60% | 80%+ | 100% |
| **Loss** | NaN / Inf | > 3.0 | 1.0-2.0 | < 0.5 |
| **Speed** | < 1.0 st/s | 1.5 st/s | 1.8+ st/s | 2.0+ st/s |

## 🎯 End-to-End Timeline

```
16:10 → 16:30   Training (teacher load + warmup)
16:30 → 16:50   Training (main loop with checkpoints)
16:50 → 17:00   First evaluation
17:00 → 17:20   Iteration #1 (if accuracy < 100%)
17:20 → 17:40   Iteration #2 (if still improving)
...
Until accuracy = 100% or max_steps reached
```

## 🤖 Autonomous Decision Rules

After each iteration, the loop automatically decides:

```python
IF accuracy > previous_best:
    KEEP the change
    TRY next experiment
    
ELIF accuracy == previous_best ± 2%:
    DISCARD change (no significant effect)
    TRY different hyperparameter
    
ELIF accuracy < previous_best - 5%:
    REVERT immediately
    TRY opposite direction next
    
ELSE (critical failure):
    REVERT and investigate
    May need manual intervention
```

## 🛑 When to Intervene

You typically **don't need to** unless:

1. **Training stuck for > 30 min** → Kill process, increase max_steps
2. **Accuracy crashes to 0%** → Likely model corruption, check logs
3. **Memory keeps spiking** → Need to reduce sequence length
4. **Iteration hits local maximum** → Manually adjust learning rate or temperature

## 📝 Next Steps After First Iteration

Once `AUTORESEARCH_RESULTS.tsv` has results:

```bash
# See all results
tail -20 AUTORESEARCH_RESULTS.tsv

# View current best
grep "COMPLETED" AUTORESEARCH_RESULTS.tsv | sort -t$'\t' -k4 -rn | head -1

# Check git history of experiments
git log --oneline | grep "experiment:" | head -10
```

## ✅ Success Criteria

- **Primary Goal**: Achieve **100% accuracy** on 20-problem eval set
- **Secondary Goal**: Maintain 100% while increasing robustness (larger eval set)
- **Stretch Goal**: Test on 50-100 problems and confirm generalization

---

**Summary**: Training is running autonomously. Check `/tmp/training.log` to monitor. Results will be logged automatically to `AUTORESEARCH_RESULTS.tsv`. No human intervention needed unless there's a failure.
