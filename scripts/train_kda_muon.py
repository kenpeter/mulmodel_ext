#!/usr/bin/env python3
"""Distillation training with Kimi Delta Attention + MuonClip optimizer."""

import torch, sys, time, os, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import StudentConfig
from model.student import StudentModel
from model.optimizer import MuonClip
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
from eval.leetcode_eval import run_leetcode_eval


class SimpleLeetCodeDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split="train", max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        ds = load_dataset("justindal/leetcode-python-dataset", split=split)
        for item in ds:
            msgs = item["messages"]
            if len(msgs) >= 3:
                prompt = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
                completion = msgs[2]["content"]
                self.samples.append({"prompt": prompt, "completion": completion})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        full = s["prompt"] + s["completion"]

        full_tok = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_tok = self.tokenizer(
            s["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_tok["attention_mask"].sum().item()

        ids = full_tok["input_ids"].squeeze(0)
        mask = full_tok["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[:prompt_len] = -100
        labels[mask == 0] = -100

        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def main():
    teacher_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Jackrong--Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/af2f37de41af0bdcaea5e3790ad323030ea4af07"
    )
    output_dir = "./checkpoints"
    max_length = 512
    batch_size = 1
    lr = 2e-4
    grad_accum = 2
    max_steps = 50000  # Extended for continuous training until 100% eval
    log_every = 10
    save_every = 200
    eval_every = 1000
    target_hour, target_min = 6, 0
    early_stop_accuracy = 100.0  # Stop when eval reaches 100%

    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()

    # Calculate time
    now = time.localtime()
    target_sec = target_hour * 3600 + target_min * 60
    current_sec = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    if target_sec > current_sec:
        seconds_avail = target_sec - current_sec
    else:
        seconds_avail = (24 * 3600 - current_sec) + target_sec
    hours_avail = seconds_avail / 3600
    max_steps = min(max_steps, int(seconds_avail / 3.0))

    print(
        f"Running until {target_hour:02d}:{target_min:02d} ({hours_avail:.1f}h, ~{max_steps} steps)"
    )
    print(f"Attention: Kimi Delta Attention (KDA)")
    print(f"Optimizer: MuonClip (lr={lr})")

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

    # Build student with KDA
    # Check for latest checkpoint first
    start_step = 0
    ckpt_files = sorted(
        [
            f
            for f in os.listdir(output_dir)
            if f.startswith("step_") and f.endswith(".pt")
        ],
        key=lambda x: int(x[len("step_") : -len(".pt")]),
    )

    if ckpt_files:
        latest = ckpt_files[-1]
        ckpt_path = os.path.join(output_dir, latest)
        print(f"Loading checkpoint: {latest}")
        ckpt_data = torch.load(ckpt_path, map_location="cuda")
        config = StudentConfig(**ckpt_data["config"])
        student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
        student.load_state_dict(ckpt_data["model"])
        start_step = ckpt_data.get("step", int(latest[len("step_") : -len(".pt")]))
        print(f"Resumed from {latest} (step {start_step})")
    else:
        print("Building student with KDA (fresh)...")
        config = StudentConfig(
            vocab_size=teacher.model.embed_tokens.weight.shape[0],
            sage_attention=False,
            attn_residual=True,
        )
        student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
        print("No checkpoint found, starting fresh")

    student.gradient_checkpointing = True
    student.train()

    # MuonClip optimizer
    optimizer = MuonClip(
        student.parameters(),
        lr=lr,
        momentum=0.95,
        weight_decay=0.1,
        qk_clip_threshold=100.0,
        newton_schulz_iters=5,
    )

    # Load data
    print("Loading LeetCode data...")
    train_ds = SimpleLeetCodeDataset(tokenizer, split="train", max_length=max_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    print(f"Train: {len(train_ds)} samples")

    # LR schedule
    warmup_steps = 50

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # Training
    step = 0
    resume_step = start_step
    t0 = time.time()
    loss_sum = 0.0
    hard_sum = 0.0
    soft_sum = 0.0

    print(f"\n{'=' * 50}")
    print(f"Starting: {max_steps} steps, {grad_accum} grad_accum")
    print(f"{'=' * 50}\n")

    try:
        for epoch in range(50):
            if step >= max_steps:
                break
            for batch_idx, batch in enumerate(train_loader):
                if step >= max_steps:
                    break

                # Update LR
                lr_now = lr * lr_fn(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

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

                    ss = s_logits[:, :-1].contiguous()
                    st = teacher_logits[:, :-1].contiguous()
                    sl = labels[:, 1:].contiguous()
                    mask = (sl != -100).float()

                    # Clamp logits to prevent NaN from extreme values
                    ss = ss.clamp(-50, 50)
                    st = st.clamp(-50, 50)

                    hard_loss = F.cross_entropy(
                        ss.view(-1, ss.size(-1)), sl.view(-1), ignore_index=-100
                    )
                    soft = F.kl_div(
                        F.log_softmax(ss / 1.5, -1),
                        F.softmax(st / 1.5, -1),
                        reduction="none",
                    ).sum(-1)
                    soft_loss = (soft * mask).sum() / mask.sum().clamp(min=1) * 4.0
                    loss = (0.7 * soft_loss + 0.3 * hard_loss) / grad_accum

                loss.backward()

                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    optimizer.apply_qk_clip(student)
                    optimizer.zero_grad()
                    step += 1

                    h = hard_loss.item()
                    s = soft_loss.item()
                    if h == h and s == s:
                        loss_sum += 0.7 * s + 0.3 * h
                        hard_sum += h
                        soft_sum += s

                    if step % log_every == 0:
                        elapsed = time.time() - t0
                        sps = step / elapsed
                        eta_min = (max_steps - step) / sps / 60
                        avg_l = loss_sum / log_every if loss_sum > 0 else float("nan")
                        avg_h = hard_sum / log_every if hard_sum > 0 else float("nan")
                        avg_s = soft_sum / log_every if soft_sum > 0 else float("nan")
                        print(
                            f"Step {step}/{max_steps} | loss={avg_l:.3f} (h={avg_h:.3f} s={avg_s:.3f}) | lr={lr_now:.1e} | {sps:.1f}st/s | ETA {eta_min:.0f}min"
                        )
                        loss_sum = hard_sum = soft_sum = 0.0

                    if step % save_every == 0:
                        path = os.path.join(output_dir, f"step_{resume_step + step}.pt")
                        torch.save(
                            {
                                "step": step,
                                "model": student.state_dict(),
                                "config": config.to_dict(),
                            },
                            path,
                        )
                        print(f"  Saved: {path}")

                    if step > 0 and step % eval_every == 0:
                        print(f"\n{'=' * 50}")
                        print(f"Running evaluation at step {step}")
                        print(f"{'=' * 50}")
                        student.eval()
                        eval_results = run_leetcode_eval(
                            model=student,
                            tokenizer=tokenizer,
                            device="cuda",
                            max_new_tokens=256,
                            num_problems=20,
                            split="test",
                            dataset_name="justindal/leetcode-python-dataset",
                        )
                        eval_path = os.path.join(
                            output_dir, f"eval_step_{resume_step + step}.json"
                        )
                        save_results = {
                            k: v for k, v in eval_results.items() if k != "results"
                        }
                        save_results["detailed_results"] = [
                            {k: v for k, v in r.items()}
                            for r in eval_results["results"]
                        ]
                        with open(eval_path, "w") as f:
                            json.dump(save_results, f, indent=2)
                        print(f"  Eval saved: {eval_path}")
                        accuracy = eval_results['accuracy']
                        print(
                            f"  Accuracy: {eval_results['passed']}/{eval_results['total']} ({accuracy:.1f}%)"
                        )

                        # Early stopping when all questions pass
                        if accuracy >= early_stop_accuracy:
                            print(f"\n{'=' * 50}")
                            print(f"🎉 EARLY STOP: Reached {accuracy:.1f}% accuracy!")
                            print(f"{'=' * 50}")
                            step = max_steps  # Force exit training loop

                        student.train()

    except KeyboardInterrupt:
        print("\nInterrupted!")

    path = os.path.join(output_dir, "final.pt")
    torch.save(
        {
            "step": resume_step + step,
            "model": student.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )
    print(f"\nFinal saved: {path}")
    print(
        f"Total: {step} steps this run ({resume_step + step} cumulative) in {(time.time() - t0) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
