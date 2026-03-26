"""MuonClip optimizer — Kimi K2's optimizer combining Muon + QK-Clip.

Muon: Newton-Schulz orthogonalization of gradient updates for 2D weight matrices.
QK-Clip: Post-update rescaling of Q/K weights to prevent attention logit explosion.

Reference: Kimi K2 tech report (Moonshot AI, 2025)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import math


def newton_schulz_iteration(M: torch.Tensor, num_iterations: int = 5) -> torch.Tensor:
    """Orthogonalize gradient update using Newton-Schulz iteration.

    For M of shape (m, n), works with the smaller Gram matrix (min(m,n), min(m,n))
    to avoid OOM on large weight matrices.
    """
    if M.dim() != 2:
        return M

    m, n = M.shape
    if m == 0 or n == 0:
        return M

    # Work with the smaller dimension
    if m > n:
        # M = (m, n), work with M^T M = (n, n)
        G = M.T @ M
    else:
        # M = (m, n), work with M M^T = (m, m)
        G = M @ M.T

    # Normalize
    norm = G.norm()
    if norm < 1e-7:
        return M
    G = G / norm

    dim = G.shape[0]
    I = torch.eye(dim, device=G.device, dtype=G.dtype)

    # Newton-Schulz: X_{k+1} = X_k * (3I - X_k^T X_k) / 2
    # Since G is symmetric, X = G
    X = G
    for _ in range(num_iterations):
        X = X @ (3.0 * I - X @ X) / 2.0

    # Recover orthogonalized M: O = M @ (M^T M)^{-1/2} ≈ M @ X
    if m > n:
        return M @ X
    else:
        return X @ M


class MuonClip(torch.optim.Optimizer):
    """MuonClip optimizer from Kimi K2.

    Combines:
    1. Muon: Newton-Schulz orthogonalized updates for 2D weight matrices
    2. AdamW-style weight decay
    3. QK-Clip: Post-update rescaling to prevent attention logit explosion

    For 1D parameters (biases, norms), falls back to standard AdamW update.

    Args:
        params: Model parameters
        lr: Learning rate (default: 2e-4)
        momentum: Momentum coefficient for Muon (default: 0.95)
        weight_decay: L2 regularization (default: 0.1)
        qk_clip_threshold: Max attention logit before clipping (default: 100.0)
        qk_clip_alpha: Balance between Q/K scaling (default: 0.5)
        newton_schulz_iters: Newton-Schulz iterations (default: 5)
        rms_scale_factor: Scale factor to match Adam RMS (default: 0.2)
        betas: Adam betas for 1D params (default: (0.9, 0.95))
        eps: Adam epsilon (default: 1e-8)
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        qk_clip_threshold: float = 100.0,
        qk_clip_alpha: float = 0.5,
        newton_schulz_iters: int = 5,
        rms_scale_factor: float = 0.2,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            qk_clip_threshold=qk_clip_threshold,
            qk_clip_alpha=qk_clip_alpha,
            newton_schulz_iters=newton_schulz_iters,
            rms_scale_factor=rms_scale_factor,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_iters = group["newton_schulz_iters"]
            rms_scale = group["rms_scale_factor"]
            betas = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    if grad.dim() == 2:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)

                state["step"] += 1

                # Weight decay (applied to all params)
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                if grad.dim() == 2:
                    # === Muon update for 2D weight matrices ===
                    buf = state["momentum_buffer"]

                    # Momentum
                    buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                    # Newton-Schulz orthogonalization
                    update = newton_schulz_iteration(buf, num_iterations=ns_iters)

                    # Scale to match Adam RMS
                    scale = math.sqrt(max(p.shape[0], p.shape[1])) * rms_scale
                    update.mul_(scale)

                    # Apply update
                    p.add_(update, alpha=-lr)
                else:
                    # === AdamW update for 1D params (biases, norms) ===
                    m, v = state["m"], state["v"]
                    beta1, beta2 = betas

                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    bc1 = 1 - beta1 ** state["step"]
                    bc2 = 1 - beta2 ** state["step"]

                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    update = (m / bc1) / denom

                    p.add_(update, alpha=-lr)

        return loss

    @torch.no_grad()
    def apply_qk_clip(self, model: nn.Module):
        """Apply QK-Clip to prevent attention logit explosion.

        After each optimizer step, check if max QK score exceeds threshold.
        If so, rescale W_q and W_k:
            eta = threshold / (max_score + eps)
            W_q *= eta^alpha
            W_k *= eta^(1-alpha)
        """
        threshold = self.defaults["qk_clip_threshold"]
        alpha = self.defaults["qk_clip_alpha"]
        eps = 1e-7

        for name, module in model.named_modules():
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                W_q = module.q_proj.weight
                W_k = module.k_proj.weight

                # Estimate max QK score from weight norms
                # ||W_q|| * ||W_k|| gives upper bound on max dot product
                q_norm = W_q.norm(dim=-1).max()
                k_norm = W_k.norm(dim=-1).max()
                max_score = (q_norm * k_norm).item()

                if max_score > threshold:
                    eta = threshold / (max_score + eps)
                    scale_q = eta**alpha
                    scale_k = eta ** (1 - alpha)
                    W_q.mul_(scale_q)
                    W_k.mul_(scale_k)
