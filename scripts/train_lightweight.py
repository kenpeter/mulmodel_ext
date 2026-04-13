#!/usr/bin/env python3
"""Lightweight training: Hard loss only, no teacher distillation. Fits in GPU memory."""

import torch, sys, os, math, json, random, gc, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import StudentConfig
from model.student import StudentModel
from model.optimizer import MuonClip
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class LocalLeetCodeDataset(Dataset):
    """Load from local jsonl file."""
    def __init__(self, tokenizer, jsonl_path, indices=None, max_length=96):
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
            full, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        prompt_tok = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        prompt_len = prompt_tok["attention_mask"].sum().item()

        ids = full_tok["input_ids"].squeeze(0)
        mask = full_tok["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[:prompt_len] = -100
        labels[mask == 0] = -100

        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

def main():
    local_data_path = "/home/kenpeter/work/data/high_quality_leetcode/train.jsonl"
    output_dir = "./checkpoints"
    max_length = 96
    batch_size = 1
    lr = 2e-4
    max_steps = 50000
    log_every = 10
    save_every = 200
    eval_every = 2000
    early_stop_accuracy = 85.0  # Lower threshold: train longer
    eval_size = 20

    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()

    with open(local_data_path, 'r') as f:
        num_problems = sum(1 for _ in f)

    print(f"Loaded {num_problems} problems")

    all_indices = list(range(num_problems))
    random.shuffle(all_indices)
    split_idx = int(0.8 * num_problems)
    train_indices = all_indices[:split_idx]
    eval_indices = all_indices[split_idx:]

    print(f"Train: {len(train_indices)} | Eval pool: {len(eval_indices)}")
    print(f"[LIGHTWEIGHT] Hard loss only (no teacher distillation)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )

    # Build/load student
    ckpt_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("step_") and f.endswith(".pt")],
        key=lambda x: int(x[len("step_") : -len(".pt")]),
    )

    if ckpt_files:
        latest = ckpt_files[-1]
        ckpt_path = os.path.join(output_dir, latest)
        print(f"Resuming from {latest}")
        ckpt_data = torch.load(ckpt_path, map_location="cuda")
        config = StudentConfig(**ckpt_data["config"])
        student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
        student.load_state_dict(ckpt_data["model"])
    else:
        print("Building student (fresh)")
        config = StudentConfig(
            vocab_size=tokenizer.vocab_size,
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

    # Data
    train_ds = LocalLeetCodeDataset(tokenizer, local_data_path, indices=train_indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    print(f"Train: {len(train_ds)} samples\n")

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

    print(f"Starting: {max_steps} steps")
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

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    s_out = student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    s_logits = s_out["logits"]
                    sl = labels[:, 1:].contiguous()

                    loss = F.cross_entropy(
                        s_logits[:, :-1].contiguous().view(-1, s_logits.size(-1)), 
                        sl.view(-1), 
                        ignore_index=-100
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.apply_qk_clip(student)
                optimizer.zero_grad()
                step += 1

                if step % 100 == 0:
                    torch.cuda.empty_cache()

                if loss.item() == loss.item():
                    loss_sum += loss.item()

                if step % log_every == 0:
                    elapsed = time.time() - t0
                    sps = step / elapsed
                    eta_min = (max_steps - step) / sps / 60
                    avg_l = loss_sum / log_every if loss_sum > 0 else float("nan")
                    print(
                        f"Step {step}/{max_steps} | loss={avg_l:.3f} | lr={lr_now:.1e} | {sps:.1f}st/s | ETA {eta_min:.0f}min"
                    )
                    loss_sum = 0.0

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

                if step > 0 and step % eval_every == 0:
                    print(f"\nEval at step {step}")
                    student.eval()
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
                    print(f"  Accuracy: {passed}/{len(sample_indices)} ({accuracy:.1f}%)\n")

                    if accuracy >= early_stop_accuracy:
                        print(f"EARLY STOP: {accuracy:.1f}% >= {early_stop_accuracy}%")
                        step = max_steps

                    del eval_ds, eval_loader
                    torch.cuda.empty_cache()
                    gc.collect()
                    student.train()

    except KeyboardInterrupt:
        print("\nInterrupted!")

    path = os.path.join(output_dir, "final.pt")
    torch.save(
        {
            "step": step,
            "model": student.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )
    print(f"\nFinal saved: {path}")
    print(f"Total: {step} steps in {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()
