# Coder Log

## Files Created
1. `model/config.py` — StudentConfig dataclass (hidden=1024, 24 layers, 8Q/2KV heads, head_dim=256)
2. `model/attention.py` — SageAttention (with Q/K norm, RoPE, gate from q_proj, sageattn kernel with SDPA fallback) + AttentionResidual (learnable sigmoid gate)
3. `model/mlp.py` — SwiGLU MLP (gate/up/down projections)
4. `model/layer.py` — TransformerDecoderLayer (pre-norm, attn + attn_residual + mlp)
5. `model/student.py` — StudentModel (embed, 24 layers, norm, lm_head, init_from_teacher, generate)
6. `train/data.py` — LeetCodeDistillDataset (justindal/leetcode-python-dataset, chat format, labels masking)
7. `train/distill.py` — DistillationTrainer (KL soft loss + CE hard loss, 8-bit Adam, grad checkpointing, bf16 AMP)
8. `eval/leetcode_eval.py` — LeetCode eval (code extraction, sandbox execution, syntax check)
9. `scripts/run_distill.py` — Entry point (load teacher, build student, init from teacher, train, eval, run-until timer)

## Key Design Decisions
- Teacher's q_proj outputs 4096 = Q(2048) + gate(2048) split — matched exactly in student
- Teacher's o_proj is [1024, 2048] not [1024, 4096] because GQA: 8 Q heads * 256 head_dim = 2048
- AttentionResidual adds extra gate BEYOND the teacher's internal gate (sigmoid from q_proj)
- Vocabulary matched directly (248320) — no projection layer needed
- Training data: 2856 LeetCode problems, 2048 max length
- Run until 6am timer built into script
