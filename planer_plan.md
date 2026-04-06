# Planner — Accuracy Drop Investigation Plan

## Goal
Investigate and fix the accuracy drop issue: 60% (20 problems) → 22-26% (50 problems)

## Definition of Done
- [ ] Fix eval_student.py to use correct checkpoint path (final.pt)
- [ ] Modify leetcode_eval.py to save detailed per-problem results
- [ ] Run evaluation on same checkpoint: first 20, next 20, last 10 problems
- [ ] Compare results to determine if drop is real or eval bug
- [ ] (Optional) Fix correctness validation - run actual test cases

---

## Steps

### Step 1: Fix checkpoint path in eval_student.py
- **Files**: `/home/kenpeter/work/mulmodel_ext/eval_student.py`
- **Change**: Line 20: `checkpoint_path = "/home/kenpeter/work/mulmodel_ext/checkpoints/final.pt"` (was student_final.pt)
- **Verify**: Check file exists with `ls -la checkpoints/final.pt`
- **Risk**: None - just path fix, easily reversible

### Step 2: Save detailed per-problem results in training eval
- **Files**: `/home/kenpeter/work/mulmodel_ext/scripts/train_kda_muon.py`
- **Change**: Line 298-299 - remove the filter that excludes "results":
  ```python
  # Before (excludes detailed results):
  save_results = {k: v for k, v in eval_results.items() if k != "results"}
  
  # After (includes detailed results):
  save_results = eval_results.copy()
  save_results["detailed_results"] = eval_results.get("results", [])
  ```
- **Verify**: After next training eval, check eval_step_*.json has "detailed_results" array
- **Risk**: Increases file size, but valuable for debugging

### Step 3: Add problem index tracking to leetcode_eval.py
- **Files**: `/home/kenpeter/work/mulmodel_ext/eval/leetcode_eval.py`
- **Change**: Add problem difficulty and title to result dict:
  - Extract difficulty from problem metadata if available
  - Add problem index to result (already has "index": i)
- **Verify**: Check saved results include problem identifiers
- **Risk**: None - adds more info, doesn't change logic

### Step 4: Run segmented evaluation on final.pt
- **Files**: Create temporary eval script or use existing eval_student.py with modifications
- **Change**: Evaluate same checkpoint (final.pt) on three segments:
  1. Problems 0-19 (first 20)
  2. Problems 20-39 (next 20)
  3. Problems 40-49 (last 10)
- **Verify**: Compare accuracy across three segments
- **Risk**: None - just running evaluation

### Step 5: Analyze results and determine root cause
- **Expected outcomes**:
  - If all segments show similar (lower) accuracy → real degradation, not eval bug
  - If first 20 still ~60% but others lower → something else (difficulty, dataset order)
  - If all segments show ~60% → previous 50-problem eval was wrong

---

## Test Plan

1. **Unit test**: Verify checkpoint path fix works
   ```bash
   ls -la checkpoints/final.pt
   python eval_student.py 2>&1 | head -20
   ```

2. **Manual verification**: Check segmented results
   - First 20: ~60%? 
   - Next 20: ~?%
   - Last 10: ~?%

3. **Integration test**: Compare with previous eval results
   - Load previous eval_results.json
   - Compare problem-by-problem if possible

---

## Out of Scope
- Training new models
- Fixing the model architecture
- Adding test case execution (would require significant changes to leetcode_eval.py)
- Investigating why accuracy dropped during training (only investigating eval)

---

## Risk Assessment
- **Low risk**: Path fix is trivial
- **Medium risk**: Segmented eval requires running 3 separate evals (time-consuming but safe)
- **Rollback**: Can always revert code changes with git
