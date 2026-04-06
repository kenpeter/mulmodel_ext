#!/usr/bin/env python3
"""Quick eval on latest checkpoint."""

import torch, sys, os, json, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import StudentConfig
from model.student import StudentModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

class LocalLeetCodeDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path, indices=None, max_length=128):
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
            padding="max_length", return_tensors="pt"
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

    # Find latest checkpoint
    ckpt_path = None

    # Find latest checkpoint (prioritize step_*.pt over model.pt)
    ckpt_files = sorted(
        [f for f in os.listdir("checkpoints") if f.startswith("step_") and f.endswith(".pt")],
        key=lambda x: int(x[len("step_"):].replace(".pt", ""))
    )

    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        ckpt_path = f"checkpoints/{latest_ckpt}"
    elif os.path.exists("checkpoints/model.pt"):
        ckpt_path = "checkpoints/model.pt"
        latest_ckpt = "model.pt"
    else:
        ckpt_path = None

    if not ckpt_path:
        print("No checkpoints found")
        return

    print(f"Loading checkpoint: {latest_ckpt}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )

    # Load student (load on CPU first to avoid OOM)
    torch.cuda.empty_cache()
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    config = StudentConfig(**ckpt_data["config"])
    student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
    student.load_state_dict(ckpt_data["model"])
    student.eval()
    torch.cuda.empty_cache()

    print(f"Model loaded")

    # Get eval set (20 random from full dataset)
    with open(local_data_path, 'r') as f:
        num_problems = sum(1 for _ in f)

    all_indices = list(range(num_problems))
    random.seed(42)  # Fixed seed for reproducibility
    eval_indices = random.sample(all_indices, min(20, num_problems))

    print(f"\nEval on {len(eval_indices)} random problems from {num_problems} total")
    print("=" * 60)

    eval_ds = LocalLeetCodeDataset(tokenizer, local_data_path, indices=eval_indices)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    passed = 0
    for i, batch in enumerate(eval_loader):
        input_ids = batch["input_ids"].to("cuda")
        try:
            with torch.no_grad():
                output = student.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    temperature=0.5,
                    top_p=0.9,
                )
            solution = tokenizer.decode(output[0], skip_special_tokens=True)
            if any(x in solution for x in ['def ', 'class ', 'return ']):
                passed += 1
                status = "✓"
            else:
                status = "✗"
        except Exception as e:
            status = "E"

        print(f"[{i+1:2d}/{len(eval_indices)}] {status}")

    accuracy = (passed / len(eval_indices) * 100) if len(eval_indices) > 0 else 0
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   Passed: {passed}/{len(eval_indices)}")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"\n📁 Checkpoint: {latest_ckpt}")

if __name__ == "__main__":
    main()
