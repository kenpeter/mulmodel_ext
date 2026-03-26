import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistillationTrainer:
    """Knowledge distillation trainer.

    Distills teacher (frozen) → student (trainable) using:
    - Soft loss: KL divergence on logits with temperature T
    - Hard loss: Cross-entropy with ground truth labels
    - Combined: alpha * soft + (1 - alpha) * hard
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        tokenizer,
        train_loader,
        val_loader,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        temperature: float = 2.0,
        alpha: float = 0.7,
        grad_accum_steps: int = 4,
        max_grad_norm: float = 1.0,
        output_dir: str = "./checkpoints",
        log_every: int = 10,
        save_every: int = 500,
        device: str = "cuda",
    ):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.temperature = temperature
        self.alpha = alpha
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.log_every = log_every
        self.save_every = save_every
        self.device = device

        os.makedirs(output_dir, exist_ok=True)

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Setup student for training
        self.student.train()
        self.student.gradient_checkpointing = True

        # Optimizer — try 8-bit Adam, fallback to standard
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=lr, weight_decay=weight_decay
            )
            print("Using 8-bit AdamW")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=lr, weight_decay=weight_decay
            )
            print("Using standard AdamW")

        # LR scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
        )

        # BFloat16 doesn't need GradScaler
        self.scaler = None

        self.global_step = 0
        self.best_val_loss = float("inf")

    def _lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _compute_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        # Shift for next-token prediction
        shift_student = student_logits[:, :-1, :].contiguous()
        shift_teacher = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask out prompt tokens (label == -100)
        mask = (shift_labels != -100).float()

        # Hard loss: cross-entropy on completion tokens only
        hard_loss = F.cross_entropy(
            shift_student.view(-1, shift_student.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Soft loss: KL divergence with temperature
        student_log_probs = F.log_softmax(shift_student / self.temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)  # [B, S-1]

        # Only compute on completion tokens
        kl_loss = (kl_loss * mask).sum() / mask.sum().clamp(min=1)
        soft_loss = kl_loss * (self.temperature**2)

        # Combined
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss.detach(),
            "soft_loss": soft_loss.detach(),
        }

    @torch.no_grad()
    def _get_teacher_logits(self, input_ids, attention_mask):
        outputs = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Teacher forward (frozen)
        teacher_logits = self._get_teacher_logits(input_ids, attention_mask)

        # BFloat16 autocast (no GradScaler needed for bf16)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            student_logits = student_outputs["logits"]

            losses = self._compute_distill_loss(student_logits, teacher_logits, labels)
            loss = losses["total_loss"] / self.grad_accum_steps

        loss.backward()

        return losses

    def train(self, max_epochs: int = 5):
        print(f"Starting distillation training for {max_epochs} epochs")
        print(f"Train batches per epoch: {len(self.train_loader)}")
        print(f"Max steps: {self.max_steps}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Temperature: {self.temperature}, Alpha: {self.alpha}")

        total_loss_sum = 0.0
        hard_loss_sum = 0.0
        soft_loss_sum = 0.0
        step_count = 0
        epoch_start = time.time()

        for epoch in range(max_epochs):
            self.student.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, batch in enumerate(self.train_loader):
                losses = self.train_step(batch)
                total_loss_sum += losses["total_loss"].item()
                hard_loss_sum += losses["hard_loss"].item()
                soft_loss_sum += losses["soft_loss"].item()
                step_count += 1
                epoch_loss += losses["total_loss"].item()
                batch_count += 1

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1

                    if self.global_step % self.log_every == 0:
                        avg_total = total_loss_sum / step_count
                        avg_hard = hard_loss_sum / step_count
                        avg_soft = soft_loss_sum / step_count
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - epoch_start
                        steps_left = self.max_steps - self.global_step
                        eta = (
                            (elapsed / self.global_step) * steps_left
                            if self.global_step > 0
                            else 0
                        )
                        print(
                            f"Step {self.global_step}/{self.max_steps} | "
                            f"Epoch {epoch + 1} | "
                            f"Loss: {avg_total:.4f} (hard: {avg_hard:.4f}, soft: {avg_soft:.4f}) | "
                            f"LR: {lr:.2e} | "
                            f"ETA: {eta / 3600:.1f}h"
                        )
                        total_loss_sum = 0.0
                        hard_loss_sum = 0.0
                        soft_loss_sum = 0.0
                        step_count = 0

                    if self.global_step % self.save_every == 0:
                        self.save_checkpoint(f"step_{self.global_step}")

                    if self.global_step >= self.max_steps:
                        print(f"Reached max steps ({self.max_steps})")
                        self.save_checkpoint("final")
                        return

            avg_epoch = epoch_loss / max(batch_count, 1)
            val_loss = self.validate()
            print(
                f"Epoch {epoch + 1} done | Train loss: {avg_epoch:.4f} | Val loss: {val_loss:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best")
                print(f"  New best model saved (val_loss: {val_loss:.4f})")

        self.save_checkpoint("final")
        print("Training complete.")

    @torch.no_grad()
    def validate(self) -> float:
        self.student.eval()
        total_loss = 0.0
        count = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            teacher_logits = self._get_teacher_logits(input_ids, attention_mask)

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                student_outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                losses = self._compute_distill_loss(
                    student_outputs["logits"], teacher_logits, labels
                )

            total_loss += losses["total_loss"].item()
            count += 1

            if count >= 50:  # Limit val batches for speed
                break

        self.student.train()
        return total_loss / max(count, 1)

    def save_checkpoint(self, name: str):
        path = os.path.join(self.output_dir, name)
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "config": self.student.config.to_dict(),
            },
            os.path.join(path, "checkpoint.pt"),
        )
        print(f"  Saved checkpoint: {path}")
