#!/usr/bin/env python3
"""Proven training with local 2638 high-quality problems + rotating eval (based on working train_kda_muon.py)."""

import torch, sys, time, os, math, json, random, gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import StudentConfig
from model.student import StudentModel
from model.optimizer import MuonClip
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class LocalLeetCodeDataset(Dataset):
    """Load from local jsonl file."""
    def __init__(self, tokenizer, jsonl_path, indices=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, 'r') as f:
            all_items = [json.loads(line) for line in f]

        if indices is not None:
            self.samples = [all_items[i] for i in indices if i < len(all_items)]
        else:
            self.samples = all_items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = f"{item['problem_description']}\n\n# Starter Code:\n{item.get('starter_code', '')}"
        completion = item.get('completion', '')
        full = prompt + completion

        full_tok = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_tok = self.tokenizer(
            prompt,
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
    local_data_path = "/home/kenpeter/work/data/high_quality_leetcode/train.jsonl"
    output_dir = "./checkpoints"
    max_length = 256  # Reduced from 384 for faster training (saves ~30% memory)
    batch_size = 1  # Keep at 1 for stability
    lr = 2e-4
    grad_accum = 2
    max_steps = 50000
    log_every = 10
    save_every = 200
    eval_every = 2000  # Reduced from 1000 to save memory during frequent evals
    target_hour, target_min = 6, 0
    early_stop_accuracy = 100.0
    eval_size = 20  # Rotating eval set

    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()

    # Load all problem indices
    with open(local_data_path, 'r') as f:
        num_problems = sum(1 for _ in f)

    print(f"Loaded {num_problems} problems from local data")

    # Split: 80% train, 20% eval (rotate eval sets)
    all_indices = list(range(num_problems))
    random.shuffle(all_indices)
    split_idx = int(0.8 * num_problems)
    train_indices = all_indices[:split_idx]
    eval_indices = all_indices[split_idx:]

    print(f"Train: {len(train_indices)} | Eval pool: {len(eval_indices)}")

    # Calculate time
    now = time.localtime()
    target_sec = target_hour * 3600 + target_min * 60
    current_sec = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    seconds_avail = target_sec - current_sec if target_sec > current_sec else (24 * 3600 - current_sec) + target_sec
    hours_avail = seconds_avail / 3600
    max_steps = min(max_steps, int(seconds_avail / 3.0))

    print(
        f"Running until {target_hour:02d}:{target_min:02d} ({hours_avail:.1f}h, ~{max_steps} steps)"
    )
    print(f"Attention: Kimi Delta Attention (KDA)")
    print(f"Optimizer: MuonClip (lr={lr})")
    print(f"Training on {len(train_indices)} problems, eval on rotating {eval_size} from {len(eval_indices)}")

    # Load teacher
    print("Loading teacher...")
    # Load teacher with 8-bit quantization to save memory
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.float32
    )
    
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        quantization_config=bnb_config,
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

    # Build/load student
    start_step = 0
    ckpt_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("step_") and f.endswith(".pt")],
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
        print("Building student (fresh)...")
        config = StudentConfig(
            vocab_size=teacher.model.embed_tokens.weight.shape[0],
            sage_attention=False,
            attn_residual=True,
        )
        student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")

    student.gradient_checkpointing = True
    student.train()

    # Optimizer
    optimizer = MuonClip(
        student.parameters(),
        lr=lr,
        momentum=0.95,
        weight_decay=0.1,
        qk_clip_threshold=100.0,
        newton_schulz_iters=5,
    )

    # Data loader
    print("Loading training data...")
    train_ds = LocalLeetCodeDataset(tokenizer, local_data_path, indices=train_indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
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
    t0 = time.time()
    loss_sum = 0.0
    hard_sum = 0.0
    soft_sum = 0.0

    print(f"\n{'=' * 50}")
    print(f"Starting: {max_steps} steps, {grad_accum} grad_accum")
    print(f"{'=' * 50}\n")

    try:
        for epoch in range(100):
            if step >= max_steps:
                break
            for batch_idx, batch in enumerate(train_loader):
                if step >= max_steps:
                    break

                lr_now = lr * lr_fn(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                labels = batch["labels"].to("cuda")

                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits

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

                    # Minimal memory cleanup only every 100 steps (let GPU compute uninterrupted)
                    if step % 100 == 0:
                        torch.cuda.empty_cache()

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
                        path = os.path.join(output_dir, f"step_{start_step + step}.pt")
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
                        print(f"Eval at step {step} (rotating set of {eval_size} from {len(eval_indices)})")
                        print(f"{'=' * 50}")

                        student.eval()
                        # Sample rotating eval set
                        if len(eval_indices) > eval_size:
                            sample_indices = random.sample(eval_indices, eval_size)
                        else:
                            sample_indices = eval_indices

                        eval_ds = LocalLeetCodeDataset(tokenizer, local_data_path, indices=sample_indices)
                        eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

                        passed = 0
                        for eval_batch in eval_loader:
                            input_ids_e = eval_batch["input_ids"].to("cuda")
                            try:
                                with torch.no_grad():
                                    output = student.generate(
                                        input_ids=input_ids_e,
                                        max_new_tokens=256,
                                        temperature=0.5,
                                        top_p=0.9,
                                    )
                                solution = tokenizer.decode(output[0], skip_special_tokens=True)
                                if any(x in solution for x in ['def ', 'class ', 'return ']):
                                    passed += 1
                            except:
                                pass

                        accuracy = (passed / len(sample_indices) * 100) if len(sample_indices) > 0 else 0
                        print(f"  Accuracy: {passed}/{len(sample_indices)} ({accuracy:.1f}%)")

                        if accuracy >= early_stop_accuracy:
                            print(f"\n{'=' * 50}")
                            print(f"🎉 EARLY STOP: Reached {accuracy:.1f}% accuracy!")
                            print(f"{'=' * 50}")
                            step = max_steps

                        student.train()

    except KeyboardInterrupt:
        print("\nInterrupted!")

    path = os.path.join(output_dir, "final.pt")
    torch.save(
        {
            "step": start_step + step,
            "model": student.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )
    print(f"\nFinal saved: {path}")
    print(
        f"Total: {step} steps this run ({start_step + step} cumulative) in {(time.time() - t0) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
