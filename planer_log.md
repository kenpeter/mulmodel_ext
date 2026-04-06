# Planner Log

## Current Task: Accuracy Drop Investigation

### Problem
- 60% accuracy on 20 problems → 22-26% on 50 problems
- Fact Seeker identified 4 issues:
  1. Difficulty NOT cause (later problems have more Easy)
  2. Detailed results NOT saved (train_kda_muon.py excludes "results")
  3. Wrong checkpoint path (student_final.pt vs final.pt)
  4. No correctness validation (only checks if code runs, not if correct)

### Plan Created
- Fix eval_student.py checkpoint path
- Save detailed per-problem results in training eval
- Run segmented evaluation (first 20, next 20, last 10) on same checkpoint
- Determine if drop is real or eval bug

---

## Decision: Same-Size Self-Attn Distillation

### Why match teacher architecture exactly?
1. **Direct weight init**: Teacher has 6 self_attn layers (3,7,11,15,19,23) with identical shapes. Student uses all 24 layers as self_attn. This means 6/24 layers start with teacher knowledge — massive head start.
2. **Same tokenizer**: Teacher vocab (248320) used directly. No vocab projection layer needed. Logit distillation is direct: teacher_logits[B,S,248320] → KL(student_logits, teacher_logits).
3. **Same hidden size (1024)**: Enables feature-level distillation via direct hidden state comparison (optional, adds another signal).

### Architecture differences (student vs teacher)
- Teacher: 18 linear_attn (GatedDeltaNet) + 6 self_attn
- Student: 24 self_attn (SageAttention + attn_residual)
- The 18 random-initialized layers are distilled purely via logit matching
- The 6 matching layers get both weight init AND logit distillation

### VRAM budget (RTX 4070 Ti, 12GB)
- Teacher (frozen, bf16): ~1.5GB
- Student (bf16): ~1.5GB
- Gradients: ~1.5GB
- Optimizer states (8-bit Adam): ~1.5GB
- Activations (gradient checkpointing): ~2GB
- Buffer/headroom: ~4.5GB
- Total: fits within 12GB

### Training data choice
- justindal/leetcode-python-dataset: 2856 train / 228 test
- Chat format with system prompt, problem statement, and reference solution
- Real LeetCode problems — directly evaluable
