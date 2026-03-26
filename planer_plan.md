# Planner — Distillation Pipeline Plan

## Goal
Distill Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled (teacher) into a custom coding model (student) with SageAttention + attention residual, evaluated on real LeetCode problems.

## Strategy: Same-Size Self-Attn Distillation
- Student matches teacher's self-attention architecture (hidden=1024, 24 layers, 8Q/2KV heads)
- All 24 student layers are self-attention (no linear_attn / GatedDeltaNet)
- Direct weight initialization from teacher's 6 self-attention layers (3,7,11,15,19,23)
- Remaining 18 layers initialized randomly, distilled via logit matching
- Use teacher's tokenizer — no vocabulary projection needed
- Add SageAttention + attention residual on top

## File Structure
```
/home/kenpeter/work/mulmodel_ext/
├── model/
│   ├── __init__.py
│   ├── config.py          # StudentModelConfig
│   ├── attention.py       # SageAttention + AttentionResidual
│   ├── mlp.py             # SwiGLU MLP
│   ├── layer.py           # TransformerDecoderLayer
│   └── student.py         # StudentModel (full model)
├── train/
│   ├── __init__.py
│   ├── distill.py         # DistillationTrainer (KL + CE loss)
│   ├── data.py            # LeetCode dataset loader
│   └── config.py          # Training hyperparams
├── eval/
│   ├── __init__.py
│   └── leetcode_eval.py   # Run model on LeetCode problems, sandbox execute
├── scripts/
│   └── run_distill.sh     # Entry point
└── requirements.txt
```

## Implementation Steps

### Step 1: model/config.py
- Define `StudentConfig` dataclass matching teacher's self-attention architecture
- hidden_size=1024, num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=2
- head_dim=256, intermediate_size=3584, vocab_size=248320
- rope_theta=10000000, partial_rotary_factor=0.25, rms_norm_eps=1e-6
- Add sage_attention=True, attn_residual=True flags

### Step 2: model/attention.py
- `SageAttention(nn.Module)`:
  - Standard Q/K/V projections + o_proj (same shapes as teacher)
  - Q/K norm (RMSNorm on head dim)
  - Rotary position embeddings (RoPE with partial rotary factor=0.25)
  - SageAttention kernel: use `sageattn` for attention computation (FP8 Q/K)
  - Fallback to standard SDPA if sageattn fails
- `AttentionResidual(nn.Module)`:
  - Learnable gate: sigmoid(gate_param)
  - output = gate * attn_output + (1 - gate) * residual_input

### Step 3: model/mlp.py
- SwiGLU MLP: gate_proj → SiLU → * up_proj → down_proj
- Same shapes as teacher: gate[3584,1024], up[3584,1024], down[1024,3584]

### Step 4: model/layer.py
- `TransformerDecoderLayer(nn.Module)`:
  - input_layernorm → SageAttention + AttentionResidual → post_attention_layernorm → MLP
  - Pre-norm architecture (RMSNorm before attn and MLP)

### Step 5: model/student.py
- `StudentModel(nn.Module)`:
  - embed_tokens[248320, 1024]
  - 24x TransformerDecoderLayer
  - RMSNorm (final)
  - lm_head tied to embed_tokens
- `init_from_teacher(teacher_model)`:
  - Copy self_attn weights from teacher layers 3,7,11,15,19,23 → student layers 3,7,11,15,19,23
  - Copy MLP weights from same layers
  - Copy layernorms from same layers
  - Copy embed_tokens and final norm
  - Initialize remaining layers from normal distribution

### Step 6: train/data.py
- Load justindal/leetcode-python-dataset (train split)
- Format: concatenate system + user → prompt, assistant → target
- Tokenize with teacher tokenizer, max_length=4096
- Create DataLoader with padding/truncation

### Step 7: train/distill.py
- `DistillationTrainer`:
  - Forward student → student_logits [B, S, 248320]
  - Forward teacher (frozen) → teacher_logits [B, S, 248320]
  - Soft loss: KL divergence(student_softmax(T), teacher_softmax(T)) with T=2.0
  - Hard loss: CrossEntropy(student_logits, target_ids)
  - Total loss: alpha * soft_loss + (1-alpha) * hard_loss (alpha=0.7)
- Optimizer: AdamW 8-bit (bitsandbytes)
- Gradient checkpointing on student layers
- BF16 mixed precision
- LR: 2e-5, warmup 100 steps, cosine decay
- Epochs: 3-5, batch_size: 1-2 (gradient accumulation 4)

### Step 8: eval/leetcode_eval.py
- Load test split (228 problems)
- For each problem: generate code with student model
- Sandbox execute with subprocess + timeout (5s per test)
- Parse problem tests from examples, run assertions
- Report: pass@1 accuracy, per-difficulty breakdown (Easy/Medium/Hard)

### Step 9: scripts/run_distill.sh
- Single entry point: init student → load teacher → distill → eval

## Risk Mitigation
- **12GB VRAM**: Use gradient checkpointing, freeze teacher, BF16, batch_size=1
- **SageAttention compatibility**: Fallback to standard SDPA if kernel fails
- **LeetCode sandbox**: subprocess with timeout, no network, restricted imports
