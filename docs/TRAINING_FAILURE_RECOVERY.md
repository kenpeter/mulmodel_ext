# 🔄 Training Failure Recovery & Auto-Restart

## Recovery Capsules Available

### 1. **Solution Correctness Regression Detection** 🛡️
**What it does:**
- Monitors if test-case passing drops
- Auto-triggers ROLLBACK if severe regression (>15% drop)
- Reverts to previous good checkpoint

**How it recovers:**
```python
if current_correctness < previous_correctness - 15%:
    print("SEVERE REGRESSION DETECTED!")
    ROLLBACK_TO(previous_checkpoint)
    RESTART_TRAINING()
```

**Example:**
```
Step 5000: 15% test-case passing ✓
Step 5500: 8% test-case passing ✗ (drop = 7%, > 5% threshold)
Action: ALERT user
Step 6000: 2% test-case passing ✗ (drop = 13%, close to 15%)
Action: AUTO-ROLLBACK to step_5000.pt + RESTART
```

---

### 2. **Model Checkpointing Capsule** 💾
**What it does:**
- Saves checkpoint every 200 steps
- If training crashes, can resume from last checkpoint
- No progress lost

**How it recovers:**
```
Training dies at step 5347
  ↓
Check: Latest checkpoint exists?
  ├─ step_5200.pt ✓ Found
  └─ Resume from step_5200.pt
  ↓
Continue training (skip 147 lost steps)
```

**Automatic resumption:**
```bash
# Scripts auto-detect latest checkpoint
python scripts/train_proven_local.py
  ├─ Checks: ./checkpoints/step_*.pt
  ├─ Finds: Latest (step_5200.pt)
  └─ Resumes: From step_5200
```

---

### 3. **Gradient Accumulation Optimizer** 📦
**What it does:**
- Enables training with larger effective batch sizes
- Prevents OOM (out of memory) crashes
- Accumulates gradients over multiple steps

**How it recovers:**
```
Normal: Batch size too large → OOM crash
With capsule: Accumulate over 4 steps → No crash
```

---

### 4. **Auto-Recovery on Crash** 🚨
**If training crashes entirely:**

```bash
# Check last step
tail -1 /tmp/training.log
# Output: Step 5347/10351 | loss=42.123...

# Resume automatically
python scripts/train_proven_local.py
# Script finds step_5200.pt → resumes at 5200
# Continues to 10351
```

---

## Recovery Scenarios

### Scenario 1: Training Stops (e.g., power outage)
```
Before crash: Step 5347
Last checkpoint: step_5200.pt (200-step intervals)

Recovery:
  1. Restart machine
  2. Run: python scripts/train_proven_local.py
  3. Script auto-detects latest checkpoint
  4. Resumes from step_5200
  5. Lost: 147 steps (~3 minutes)
```

### Scenario 2: Test-Case Passing Drops (Regression)
```
Step 5000: 15% passing ✓
Step 5500: 8% passing ✗ (7% drop > 5% threshold)

Auto-Recovery:
  1. Regression Detection alerts
  2. Suggests: "Try different capsule"
  3. If drop > 15%: Auto-rollback
  4. Revert to: step_5000.pt
  5. Try different capsule
  6. Resume training
```

### Scenario 3: Out of Memory (OOM)
```
Normal batch size → OOM crash

Recovery with Gradient Accumulation:
  1. Detect OOM
  2. Reduce batch size
  3. Increase accumulation steps
  4. Resume without crash
```

### Scenario 4: Loss Divergence
```
Loss suddenly spikes (NaN, divergence)

Recovery:
  1. Revert to checkpoint with low loss
  2. Reduce learning rate by 0.5x
  3. Apply Gradient Clipping
  4. Resume training
```

---

## How to Manually Recover

### Option 1: Auto-Recover (Recommended)
```bash
# Just restart - auto-detection handles it
nohup python -u scripts/train_proven_local.py 2>&1 | tee -a /tmp/training.log &
```

### Option 2: Resume from Specific Checkpoint
```bash
# If you want to restart from a specific point
python scripts/train_proven_local.py --checkpoint checkpoints/step_5000.pt
```

### Option 3: Full Reset
```bash
# Start completely from scratch
python scripts/train_proven_local.py --reset
```

---

## Monitoring for Issues

### Watch for Loss Divergence
```bash
tail -f /tmp/training.log | grep loss
# If loss spikes to 100+, watch carefully
```

### Monitor Test-Case Correctness
```bash
python correctness_verifier.py --monitor
# Will alert if test-case passing drops
```

### Check Auto-Research Decisions
```bash
tail AUTORESEARCH_RESULTS.tsv
# Shows what capsules helped/hurt
```

---

## Active Recovery Mechanisms

| Mechanism | Triggers On | Action | Auto? |
|-----------|-----------|--------|-------|
| Regression Detection | Drop > 5% | Alert user | ✅ |
| Rollback | Drop > 15% | Revert checkpoint | ✅ |
| Checkpoint Save | Every 200 steps | Save state | ✅ |
| OOM Prevention | Memory full | Reduce batch | ✅ |
| Loss Monitoring | NaN/divergence | Alert | ✅ |
| Resume | Crash/restart | Auto-detect checkpoint | ✅ |

---

## Summary

✅ **Automatic Recovery:**
- Checkpoints every 200 steps
- Regression detection with auto-rollback
- Auto-resume from latest checkpoint
- OOM prevention built-in

✅ **You don't need to do anything:**
- Training crashes → Auto-restart
- Regression detected → Auto-rollback
- Loss diverges → Alert + recommendations

🚀 **Training is protected by multiple safety layers**
