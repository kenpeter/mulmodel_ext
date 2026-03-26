# Fact Seeker Report — Updated

## Teacher Model: Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled
- **Status**: Downloaded, loads, inference verified ✓
- **Architecture**: Qwen3.5ForCausalLM, 0.75B params
- **Layers**: 24 total — 18 linear_attn (Qwen3_5GatedDeltaNet) + 6 self_attn (Qwen3_5Attention) at positions 3,7,11,15,19,23
- **Hidden**: 1024, **Intermediate**: 3584 (SwiGLU)
- **Self-attn weights**: q_proj[4096,1024], k_proj[512,1024], v_proj[512,1024], o_proj[1024,2048], q_norm[256], k_norm[256]
- **MLP weights**: gate_proj[3584,1024], up_proj[3584,1024], down_proj[1024,3584]
- **Vocab**: 248320, **Max context**: 262144, **Tokenizer**: TokenizersBackend
- **LM head**: Tied with embed_tokens (tie_word_embeddings=True)
- **Vision params**: 0.0M (stripped from distillation — text-only model)
- **Config features**: attn_output_gate, partial_rotary_factor=0.25, rope_theta=10M

## Student Model: Custom Transformer — Design
- **Architecture**: Standard Transformer (all self-attn layers) + SageAttention + Attention Residual
- **Strategy**: Match teacher's self_attn architecture to enable direct weight init from teacher's 6 self_attn layers
- **Use teacher's tokenizer** (248320 vocab) — no vocab mismatch, direct logit distillation
- **Config**: hidden=1024, 24 layers, intermediate=3584, 8 Q heads, 2 KV heads, head_dim=256
- **Enhancements**: SageAttention kernel + learnable attention gate (attn_residual)
- **VRAM**: ~12GB RTX 4070 Ti — use bf16 + gradient checkpointing + 8-bit Adam

## Distillation Data
- **Training**: justindal/leetcode-python-dataset (2856 train samples, chat format)
- **Evaluation**: same dataset test split (228 problems) — real LeetCode with starter code + reference solutions
- **Format**: system prompt + user problem → assistant solution (Python code)

## Environment
- **GPU**: RTX 4070 Ti (12GB VRAM)
- **transformers**: 5.3.0 ✓
- **PyTorch**: 2.10.0, CUDA ✓
- **sageattention**: 1.0.6 ✓, **flash-attn**: 2.8.3
- **datasets**: 4.0.0, **peft**: 0.18.1, **trl**: 0.29.1
