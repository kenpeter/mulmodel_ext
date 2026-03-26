# Planner Log

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
