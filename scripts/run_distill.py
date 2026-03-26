#!/usr/bin/env python3
"""Distillation pipeline: Qwen3.5-0.8B teacher → custom student with SageAttention + attn_residual.

Usage:
    python run_distill.py [--max-steps 10000] [--batch-size 1] [--max-length 2048]
"""

import os
import sys
import argparse
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import StudentConfig
from model.student import StudentModel
from train.data import create_dataloaders
from train.distill import DistillationTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Distill Qwen3.5-0.8B into custom student"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Max training steps"
    )
    parser.add_argument("--max-epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--temperature", type=float, default=2.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="Soft loss weight (0-1)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument(
        "--save-every", type=int, default=500, help="Save every N steps"
    )
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument(
        "--output-dir", type=str, default="./checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--teacher-id",
        type=str,
        default="Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument(
        "--eval-problems", type=int, default=50, help="Number of eval problems"
    )
    parser.add_argument(
        "--run-until",
        type=str,
        default="06:00",
        help="Run training until this time (HH:MM), then eval and exit",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = getattr(
            torch.cuda.get_device_properties(0), "total_memory", None
        ) or getattr(torch.cuda.get_device_properties(0), "total_mem", 0)
        print(f"VRAM: {total_mem / 1e9:.1f} GB")

    # --- Load Teacher ---
    import os.path

    teacher_local = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{args.teacher_id.replace('/', '--')}/snapshots/"
    )
    if os.path.isdir(teacher_local):
        snapshots = sorted(os.listdir(teacher_local))
        if snapshots:
            teacher_path = os.path.join(teacher_local, snapshots[-1])
            print(f"\nLoading teacher from local: {teacher_path}")
        else:
            teacher_path = args.teacher_id
            print(f"\nLoading teacher from hub: {args.teacher_id}")
    else:
        teacher_path = args.teacher_id
        print(f"\nLoading teacher from hub: {args.teacher_id}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_path, trust_remote_code=True, local_files_only=True
    )
    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e9
    print(f"Teacher loaded: {teacher_params:.2f}B params, vocab={tokenizer.vocab_size}")

    # --- Build Student ---
    print("\nBuilding student model...")
    config = StudentConfig(
        vocab_size=tokenizer.vocab_size,
        sage_attention=True,
        attn_residual=True,
    )
    student = StudentModel(config).to(dtype=torch.bfloat16, device=device)
    student_params = sum(p.numel() for p in student.parameters()) / 1e9
    print(f"Student built: {student_params:.2f}B params")

    # --- Init from teacher ---
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(
            os.path.join(args.resume, "checkpoint.pt"), map_location=device
        )
        student.load_state_dict(ckpt["model_state_dict"])
        print(f"  Resumed at step {ckpt.get('global_step', 0)}")
    else:
        print("\nInitializing student from teacher self-attention layers...")
        mapped = student.init_from_teacher(teacher)
        print(f"  Mapped {mapped} tensors from teacher")

    # --- Eval only mode ---
    if args.eval_only:
        from eval.leetcode_eval import run_leetcode_eval

        print("\nRunning evaluation only...")
        results = run_leetcode_eval(
            student,
            tokenizer,
            device=device,
            num_problems=args.eval_problems,
        )
        return

    # --- Create dataloaders ---
    print("\nLoading LeetCode dataset...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")

    # --- Calculate max steps from run_until ---
    if args.run_until:
        now = time.localtime()
        target_parts = args.run_until.split(":")
        target_hour, target_min = int(target_parts[0]), int(target_parts[1])
        target_seconds = target_hour * 3600 + target_min * 60
        current_seconds = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec

        if target_seconds > current_seconds:
            seconds_available = target_seconds - current_seconds
        else:
            seconds_available = (24 * 3600 - current_seconds) + target_seconds

        # Rough estimate: ~2-3 sec per step at batch_size=1
        estimated_steps = int(seconds_available / 2.5)
        args.max_steps = min(args.max_steps, estimated_steps)
        hours_available = seconds_available / 3600
        print(f"\nRunning until {args.run_until} ({hours_available:.1f}h available)")
        print(f"Estimated steps: ~{args.max_steps}")

    # --- Train ---
    print(f"\n{'=' * 60}")
    print("Starting distillation training")
    print(f"{'=' * 60}")
    print(
        f"  Steps: {args.max_steps} | Batch: {args.batch_size} | Length: {args.max_length}"
    )
    print(f"  LR: {args.lr} | Temp: {args.temperature} | Alpha: {args.alpha}")
    print(f"  Grad accum: {args.grad_accum}")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 60}\n")

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        temperature=args.temperature,
        alpha=args.alpha,
        grad_accum_steps=args.grad_accum,
        output_dir=args.output_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        device=device,
    )

    trainer.train(max_epochs=args.max_epochs)

    # --- Final Eval ---
    print("\n" + "=" * 60)
    print("Training complete. Running final evaluation...")
    print("=" * 60)

    from eval.leetcode_eval import run_leetcode_eval

    eval_results = run_leetcode_eval(
        student,
        tokenizer,
        device=device,
        num_problems=args.eval_problems,
    )

    # Save eval results
    import json

    eval_path = os.path.join(args.output_dir, "eval_results.json")
    save_results = {k: v for k, v in eval_results.items() if k != "results"}
    save_results["results"] = [
        {k: v for k, v in r.items()} for r in eval_results["results"]
    ]
    with open(eval_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nEval results saved to {eval_path}")
    print("Done.")


if __name__ == "__main__":
    main()
