import torch
import torch.nn as nn
from typing import Optional

from .config import StudentConfig
from .attention import RMSNorm
from .layer import TransformerDecoderLayer


class StudentModel(nn.Module):
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (tied to embeddings if configured)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict:
        hidden_states = self.embed_tokens(input_ids)

        if self.gradient_checkpointing and self.training:
            for layer in self.layers:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
        else:
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            next_logits = outputs["logits"][:, -1, :] / temperature

            if do_sample:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                    :, :-1
                ].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() in (151644, 151645):
                break

        return generated

    def init_from_teacher(self, teacher_model):
        """Initialize student from teacher's matching layers."""
        teacher_state = teacher_model.state_dict()
        student_state = self.state_dict()

        mapped = 0
        for t_name, param in teacher_state.items():
            s_name = t_name
            if s_name.startswith("model."):
                s_name = s_name[6:]

            if any(
                skip in s_name
                for skip in ["visual", "vision", "linear_attn", "mtp", "rotary"]
            ):
                continue

            if s_name in student_state:
                if student_state[s_name].shape == param.shape:
                    student_state[s_name].copy_(param)
                    mapped += 1

        self.load_state_dict(student_state)
        print(f"Initialized {mapped} tensors from teacher")
        return mapped
