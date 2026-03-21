"""
GPT Language Model with:
- Attention Residuals (AttnRes) from Kimi Team paper
- Sage attention (memory-efficient)
- SwiGLU MLP
- Gradient checkpointing support
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from sageattention import sageattn

    HAS_SAGE = True
except ImportError:
    HAS_SAGE = False


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if HAS_SAGE and not self.training:
            # Sage attention at inference only (in-place ops break torch.compile)
            y = sageattn(q, k, v, is_causal=True)
            # Check for NaN/Inf — sage int8 quantization can overflow
            if not torch.isfinite(y).all():
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
        elif self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    """SwiGLU MLP — better than GELU for code generation."""

    def __init__(self, config):
        super().__init__()
        hidden = int(8 / 3 * config.n_embd)  # SwiGLU hidden dim
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class AttnResBlock(nn.Module):
    """Block with Attention Residuals (AttnRes) from Kimi paper.

    Instead of fixed x = x + f(x), uses learned attention over
    all previous layer outputs to select what to keep.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = SwiGLU(config)

        # AttnRes: attention over previous layer outputs
        if layer_idx > 0:
            self.attnres_ln = LayerNorm(config.n_embd, bias=config.bias)
            self.attnres_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.attnres_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.attnres_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.attnres_proj = nn.Linear(
                config.n_embd, config.n_embd, bias=config.bias
            )
            self.attnres_scale = nn.Parameter(torch.zeros(1))  # learned gate

    def forward(self, x, prev_hidden):
        # Standard transformer block
        h_attn = self.attn(self.ln_1(x))
        h_mlp = self.mlp(self.ln_2(x + h_attn))

        # Attention Residuals: attend over previous layer outputs
        if self.layer_idx > 0 and len(prev_hidden) > 0:
            prev = torch.stack(prev_hidden, dim=2)  # (B, T, L_prev, D)
            B, T, L_prev, D = prev.shape
            prev_flat = prev.view(B * T, L_prev, D)
            x_flat = x.reshape(B * T, 1, D)

            q = self.attnres_q(self.attnres_ln(x_flat))  # (B*T, 1, D)
            k = self.attnres_k(self.attnres_ln(prev_flat))  # (B*T, L_prev, D)
            v = self.attnres_v(prev_flat)  # (B*T, L_prev, D)

            scale = 1.0 / math.sqrt(D)
            attn_weights = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
            attn_out = (attn_weights @ v).view(B, T, D)
            attn_out = self.attnres_proj(attn_out)

            # Learned gate
            gate = torch.sigmoid(self.attnres_scale)
            x = x + gate * attn_out

        x = x + h_attn + h_mlp
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_gradient_checkpointing: bool = False


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [AttnResBlock(config, i) for i in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("attnres_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
        print(f"Sage attention: {HAS_SAGE}")
        print(f"Gradient checkpointing: {config.use_gradient_checkpointing}")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward with Attention Residuals
        prev_hidden = []
        for block in self.transformer.h:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, prev_hidden, use_reentrant=False
                )
            else:
                x = block(x, prev_hidden)
            prev_hidden.append(x.detach())  # detach to save memory

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
