# Debug Findings: Checkpoint Regression (2026-04-12)

## Primary Issue: Lost Working Checkpoint

**Status:** CONFIRMED HIGH SEVERITY

The autonomous loop setup (Apr 12) restarted training from the wrong checkpoint baseline.

### Timeline
- **Apr 6 23:38** — Commit b51479f: 100% accuracy achieved on 20 LeetCode problems
  - Checkpoint: `checkpoints/final.pt` (from that session)
  - Status: Working model, ready for optimization

- **Apr 12 16:13** — Commit 5cc6ca7: Autonomous loop setup
  - **Action:** Started fresh training from `step_200.pt` (OLD checkpoint from 6+ days before)
  - **Impact:** Lost reference to April 6 working model
  - Training ran 16,387 steps but never recovered

- **Apr 12 17:37** — Final checkpoint saved
  - `checkpoints/final.pt` = corrupted weights (40% accuracy only)
  - All intermediate `step_*.pt` files are from degraded training baseline

### Evidence
1. **No eval completions in current logs:** `/tmp/training.log` has NO "Accuracy:" lines from Apr 12 session
2. **Eval never ran successfully:** Training reached step 13936 but eval checkpoint logic never printed results
3. **Final checkpoint shows 40%:** Current baseline is regression, not improvement

### Fix Required
**Restore April 6 checkpoint or restart with proper initial weights**
