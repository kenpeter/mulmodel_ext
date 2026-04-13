# OFFICIAL Evaluation Protocol (IMMUTABLE)

**Status**: LOCKED - This protocol CANNOT CHANGE

## Two-Metric Evaluation System

### 1. Real Metric (PRIMARY) 🔴
**Name**: Test Case Passing  
**Script**: `eval.py` (runs both metrics)  
**Measurement**: Execute generated code, run all test assertions from data  
**Range**: 0-100%  
**What It Measures**: Does the generated solution actually solve the problem?  

**How It Works**:
1. Generate solution code from model
2. Execute the generated code in Python
3. Run the test function from `data['test']` field (50-100+ assertions per problem)
4. If ALL assertions pass → 1 point
5. If ANY assertion fails OR code errors → 0 points

**Why This Metric**:
- Measures what we actually care about: solving LeetCode problems
- Objective and mechanical: no interpretation needed
- Aligns with human evaluation: "passing all test cases"
- Prevents optimizing for fake metrics (code keywords, syntax, etc.)
- Uses REAL test data from dataset

### 2. Guard Metric (SECONDARY) 🟡
**Name**: Code Structure  
**Script**: `eval.py` (included)  
**Measurement**: Does output contain keywords: `def`, `class`, `return`  
**Range**: 0-100%  
**What It Measures**: Does the model generate code-like output?  

**Why This Metric**:
- Sanity check that model still generates code
- Prevents catastrophic regression (model outputting random text)
- Early warning sign of training instability
- Guard: Real metric can improve ONLY if Code metric doesn't degrade

---

## Ground Truth Results

**Current Model State** (step_4000.pt + SDPA attention):
- Real Metric: **0%** (0/20 problems solved)
- Code Metric: **25%** (5/20 have keywords)

**Interpretation**: Model generates code-like text 25% of the time, but NONE of it is correct.

---

## Running Evaluation

### Run Official Eval (Both Metrics)
```bash
python eval.py
```

### Output Interpretation
```
📊 OFFICIAL EVALUATION RESULTS:

   🔴 REAL METRIC (Test Case Passing):
      0/20 = 0.0%
      ↳ Does generated solution actually solve the problem?

   🟡 GUARD METRIC (Code Structure):
      5/20 = 25.0%
      ↳ Does output contain code keywords (def, class, return)?
```

---

## Success Criteria

For training to be considered successful:
- **Real Metric must IMPROVE**: 0% → target%
- **Guard Metric must NOT degrade**: keep >20%
- Both metrics reported together, always

---

## Why This Matters

The previous evaluation (40% keyword-checking) was **misleading**:
- Model had 40% accuracy at generating keywords
- But 0% accuracy at actually solving problems
- Training looked successful when model was actually failing

The new protocol fixes this deception and measures real performance.

---

## Immutability Clause

This protocol is locked and CANNOT be changed because:
1. Prevents reverting to fake metrics
2. Ensures consistent baseline for future iterations
3. Maintains alignment with human evaluation
4. Guards against metric gaming

To change this evaluation, create a separate NEW protocol file (e.g., `EVALUATION_PROTOCOL_V2.md`) instead of modifying this one.

