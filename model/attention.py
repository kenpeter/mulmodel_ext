import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StudentConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class KimiLinearAttention(nn.Module):
    """Kimi Delta Attention (KDA) from Kimi Linear paper.

    Uses the FLA library's KimiDeltaAttention implementation.
    Hybrid architecture: 3 KDA layers + 1 full attention layer (3:1 ratio).
    """

    def __init__(self, config: StudentConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        try:
            # Temporarily disabled KDA due to CUDA assert error in FLA library
            raise ImportError("KDA disabled - using standard SDPA attention")
            from fla.layers import KimiDeltaAttention

            self.kda = KimiDeltaAttention(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_attention_heads,
                num_v_heads=config.num_attention_heads,  # same as Q heads for KDA
                mode="chunk",
                use_short_conv=True,
                conv_size=4,
                conv_bias=False,
                layer_idx=layer_idx,
                norm_eps=config.rms_norm_eps,
            )
            self._use_kda = True
        except ImportError:
            # Fallback to standard SDPA attention
            self._use_kda = False
            self.q_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                bias=False,
            )
            self.k_proj = nn.Linear(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                bias=False,
            )
            self.v_proj = nn.Linear(
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                bias=False,
            )
            self.o_proj = nn.Linear(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                bias=False,
            )
            self.scaling = config.head_dim**-0.5
            self.num_kv_groups = (
                config.num_attention_heads // config.num_key_value_heads
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        if self._use_kda:
            attn_output, _, _ = self.kda(
                hidden_states,
                attention_mask=attention_mask,
            )
            return attn_output, None
        else:
            B, S, _ = hidden_states.shape
            H = self.config.num_attention_heads
            D = self.config.head_dim

            q = self.q_proj(hidden_states).view(B, S, H, D).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, S, -1, D).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, S, -1, D).transpose(1, 2)

            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)

            if attention_mask is not None and attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :].to(dtype=q.dtype)
                attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min
            else:
                attn_mask = None

            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=attn_mask is None and S > 1,
            )
            attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
            return self.o_proj(attn_output), None


class AttentionResidual(nn.Module):
    """Learnable gate that blends attention output with residual input."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, attn_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        return gate * attn_output + (1.0 - gate) * residual
