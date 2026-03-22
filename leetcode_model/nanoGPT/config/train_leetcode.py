"""
nanoGPT config for LeetCode — v2 architecture.

Changes from v1:
- Attention Residuals (AttnRes) — learned depth-wise aggregation
- Sage attention — memory-efficient attention
- SwiGLU MLP — better than GELU for code
- Gradient checkpointing — saves VRAM
- torch.compile — already enabled
"""

# ── Paths ───────────────────────────────────────────────────────────────
out_dir = "out-leetcode"
eval_interval = 50
eval_iters = 20
log_interval = 10

# ── Data ────────────────────────────────────────────────────────────────
dataset = "leetcode"
gradient_accumulation_steps = 1

# ── Model ───────────────────────────────────────────────────────────────
n_layer = 10
n_head = 10
n_embd = 640
block_size = 512
dropout = 0.1
bias = False
use_gradient_checkpointing = False

# ── AdamW optimizer ─────────────────────────────────────────────────────
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ── Learning rate decay ─────────────────────────────────────────────────
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 1e-4

# ── System ──────────────────────────────────────────────────────────────
device = "cuda"
dtype = "bfloat16"
compile = False  # Disabled for stability
