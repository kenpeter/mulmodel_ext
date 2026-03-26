import torch
import torch.nn as nn
from typing import Optional

from .config import StudentConfig
from .attention import KimiLinearAttention, AttentionResidual, RMSNorm
from .mlp import SwiGLUMLP


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: StudentConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = KimiLinearAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = SwiGLUMLP(config)

        if config.attn_residual:
            self.attn_residual = AttentionResidual(config.hidden_size)
        else:
            self.attn_residual = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        # Pre-norm attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
        )

        # Attention residual (skip connection gate)
        if self.attn_residual is not None:
            attn_output = self.attn_residual(attn_output, residual)

        hidden_states = residual + attn_output

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states
