#!/usr/bin/env python3
"""Distillation training loop — runs until specified time."""

import torch, sys, time, os, signal, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import StudentConfig
from model.student import StudentModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from train.data import create_dataloaders
import torch.nn.functional as F


def main():
    teacher_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Jackrong--Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/af2f37de41af0bdcaea5e3790ad323030ea4af07"
    )
    output_dir = "./checkpoints"
    max_length = 512
    batch_size = 1
    lr = 5e-6
    grad_accum = 4
    max_steps = 15000
    log_every = 10
    save_every = 200
    target_hour, target_min = 6, 0  # Run until 06:00

    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()

    # Calculate available time
    now = time.localtime()
    target_sec = target_hour * 3600 + target_min * 60
    current_sec = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    if target_sec > current_sec:
        seconds_avail = target_sec - current_sec
    else:
        seconds_avail = (24 * 3600 - current_sec) + target_sec
    hours_avail = seconds_avail / 3600
    max_steps = min(max_steps, int(seconds_avail / 3.0))  # ~3s per step

    print(
        f"Running until {target_hour:02d}:{target_min:02d} ({hours_avail:.1f}h, ~{max_steps} steps)"
    )
    print(f"Config: lr={lr}, grad_accum={grad_accum}, max_length={max_length}")

    # Load teacher
    print("Loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_path, trust_remote_code=True, local_files_only=True
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Build student
    print("Building student...")
    config = StudentConfig(
        vocab_size=tokenizer.vocab_size + 1000,
        sage_attention=False,
        attn_residual=False,
    )
    # Use the teacher's actual embedding size to match
    config.vocab_size = teacher.model.embed_tokens.weight.shape[0]
    student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
    student.init_from_teacher(teacher)
    student.gradient_checkpointing = True
    student.train()

    # Data
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        tokenizer, batch_size=batch_size, max_length=max_length
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Optimizer
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=lr, weight_decay=0.01)
    except ImportError:
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    # LR schedule
    import math

    warmup_steps = 50

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

    # Training
    step = 0
    t0 = time.time()
    loss_sum = 0.0
    hard_sum = 0.0
    soft_sum = 0.0

    print(f"\n{'=' * 50}")
    print(f"Starting training: {max_steps} steps, {grad_accum} grad_accum")
    print(f"{'=' * 50}\n")

    try:
        for epoch in range(50):
            if step >= max_steps:
                break
            for batch_idx, batch in enumerate(train_loader):
                if step >= max_steps:
                    break

                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                labels = batch["labels"].to("cuda")

                # Teacher forward
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits

                # Student forward
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    s_out = student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    s_logits = s_out["logits"]

                    # Distillation loss
                    ss = s_logits[:, :-1].contiguous()
                    st = teacher_logits[:, :-1].contiguous()
                    sl = labels[:, 1:].contiguous()
                    mask = (sl != -100).float()

                    hard_loss = F.cross_entropy(
                        ss.view(-1, ss.size(-1)), sl.view(-1), ignore_index=-100
                    )
                    soft = F.kl_div(
                        F.log_softmax(ss / 2.0, -1),
                        F.softmax(st / 2.0, -1),
                        reduction="none",
                    ).sum(-1)
                    soft_loss = (soft * mask).sum() / mask.sum().clamp(min=1) * 4.0
                    loss = (0.7 * soft_loss + 0.3 * hard_loss) / grad_accum

                loss.backward()

                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    step += 1

                    h = hard_loss.item()
                    s = soft_loss.item()
                    if h == h and s == s:  # not NaN
                        loss_sum += 0.7 * s + 0.3 * h
                        hard_sum += h
                        soft_sum += s

                    if step % log_every == 0:
                        elapsed = time.time() - t0
                        sps = step / elapsed
                        eta_steps = max_steps - step
                        eta_min = eta_steps / sps / 60
                        avg_loss = (
                            loss_sum / log_every if loss_sum > 0 else float("nan")
                        )
                        avg_hard = (
                            hard_sum / log_every if hard_sum > 0 else float("nan")
                        )
                        avg_soft = (
                            soft_sum / log_every if soft_sum > 0 else float("nan")
                        )
                        lr_now = scheduler.get_last_lr()[0]
                        print(
                            f"Step {step}/{max_steps} | loss={avg_loss:.3f} (h={avg_hard:.3f} s={avg_soft:.3f}) | lr={lr_now:.1e} | {sps:.1f}st/s | ETA {eta_min:.0f}min"
                        )
                        loss_sum = hard_sum = soft_sum = 0.0

                    if step % save_every == 0:
                        path = os.path.join(output_dir, f"step_{step}.pt")
                        torch.save(
                            {
                                "step": step,
                                "model": student.state_dict(),
                                "config": config.to_dict(),
                            },
                            path,
                        )
                        print(f"  Saved: {path}")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    # Save final
    path = os.path.join(output_dir, "final.pt")
    torch.save(
        {"step": step, "model": student.state_dict(), "config": config.to_dict()}, path
    )
    print(f"\nFinal model saved: {path}")
    print(f"Total: {step} steps in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
