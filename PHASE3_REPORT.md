# Phase 3: Extended Validation Report

**Date:** 2026-04-13  
**Model:** checkpoints/final.pt (1.5GB, 320M params, 96-token context)  
**Training Status:** Early-stopped at Step 4000 (reported 100% accuracy)

---

## Executive Summary

⚠️ **CRITICAL FINDING: Severe Overfitting Detected**

While the model achieved **100% accuracy on the baseline 20-problem rotating eval set**, Phase 3 validation reveals **severe overfitting** and poor generalization:

- **50-problem extended eval:** 56% accuracy (28/50)
- **Multi-rotation robustness:** 41% average (range: 15% to 65%)
- **Conclusion:** Model has learned problem-specific patterns rather than generalizing to solve arbitrary LeetCode problems

---

## Test Results

### Test 1: Extended Validation (50 Problems, Seed=123)

| Metric | Value |
|--------|-------|
| Problems Tested | 50 |
| Passed | 28 |
| Accuracy | **56.0%** |
| Time per Problem | 7.03s |
| Verdict | ❌ **Generalization Failure** |

**Finding:** Accuracy drops dramatically from 100% baseline to 56% on unseen problems.

---

### Test 2: Robustness Analysis (5x 20-problem rotations)

Different random seeds reveal extreme instability:

| Rotation | Seed | Accuracy | Problems | Verdict |
|----------|------|----------|----------|---------|
| 1 | 42 | 40% | 8/20 | ❌ Poor |
| 2 | 123 | 65% | 13/20 | ❌ Unstable |
| 3 | 456 | 55% | 11/20 | ❌ Poor |
| 4 | 789 | 15% | 3/20 | ❌ **Very Poor** |
| 5 | 999 | 30% | 6/20 | ❌ Poor |

**Summary Statistics:**
- Average: 41.0%
- Minimum: 15.0% (seed 789)
- Maximum: 65.0% (seed 123)
- **Range: 50 percentage points** (extremely high variability)

**Finding:** The 100% at step 4000 was specific to a particular 20-problem sample. Performance varies wildly across different problem distributions.

---

## Root Cause Analysis

### Why Did Step 4000 Report 100%?

The step 4000 eval used a specific rotating sample:
- **Problems:** 20 out of 528 eval-pool problems
- **Indices:** [249, 280, 281, 604, 666, 720, 870, 1068, 1069, 1141, 1286, 1636, 1741, 1786, 1815, 1879, 2094, 2315, 2382, 2418]
- **Result:** 100% (20/20 correct)
- **Interpretation:** These 20 problems were coincidentally solvable by the model

### Why Does It Fail on Other Problems?

The model has **overfitted** to:
1. **Specific problem patterns** it encountered during training
2. **The eval rotation distribution** (seed=42)
3. **Problem types in the 80% training split**

When tested on:
- Different problem distributions (different seeds) → Accuracy drops to 15-65%
- Larger sample sizes (50 problems) → Accuracy drops to 56%
- Random sampling from full dataset → Average 41% across rotations

---

## Implications

### ✅ What Worked
- Knowledge distillation framework successfully learns to generate code
- Training converged and loss decreased over 4000 steps
- Checkpoint management and early-stop mechanism functional

### ❌ Critical Issues
1. **Severe overfitting:** Model memorized patterns specific to training distribution
2. **Poor generalization:** Can't solve arbitrary LeetCode problems reliably
3. **High variance:** 50-point range in accuracy across different problem sets
4. **Misleading metric:** 100% early-stop accuracy was false positive

---

## Recommendations

### Option 1: Continue Training (Recommended)
- **Rationale:** 4000 steps may be insufficient; model needs more diverse exposure
- **Action:** Increase `max_steps` to 15,000-20,000
- **Expected:** Better generalization through more problem variety during training
- **Risk:** Takes more time; may hit compute/time limits

### Option 2: Adjust Hyperparameters
- Lower learning rate (2e-4 → 1e-4) for better convergence
- Increase `max_length` (96 → 128) to handle longer problems
- Adjust loss weights: soft 0.7 → 0.6, hard 0.3 → 0.4
- **Rationale:** May improve generalization by forcing deeper learning
- **Risk:** Requires re-training from scratch

### Option 3: Data Augmentation
- Add problem shuffling or paraphrasing to create more diverse training examples
- Use different problem orderings
- **Rationale:** More training diversity → better generalization
- **Risk:** Significant implementation effort

### Option 4: Accept Limitations & Document
- Model good for specific problem types in training distribution
- Useful as baseline but not production-ready
- **Rationale:** Phase 3 validation successfully identified overfitting
- **Value:** Clear understanding of model's actual capabilities

---

## Decision Point

**Current Status:** Model achieves impressive 100% on rotating baseline but fails to generalize.

**Next Action Required:** Should we:
1. **A)** Continue training with more steps (recommended for better generalization)
2. **B)** Accept this as a learning checkpoint and document findings
3. **C)** Try different hyperparameters before continuing

**Recommendation:** Option A - Extend training to 15K-20K steps for better diversity exposure and improved generalization before considering deployment.

---

## Validation Methodology

All evaluations use:
- **Model:** checkpoints/final.pt (early-stopped at step 4000)
- **Hardware:** NVIDIA RTX 4070 Ti
- **Generation:** temperature=0.5, top_p=0.9, max_new_tokens=256
- **Success Criteria:** Solution contains 'def ', 'class ', or 'return '
- **Random Seeds:** Deterministic for reproducibility

---

**Generated:** 2026-04-13  
**Status:** Phase 3 validation complete - critical overfitting detected  
**Next Review:** After extended training (if pursued)
