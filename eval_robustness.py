#!/usr/bin/env python3
"""Test robustness: Evaluate on multiple 20-problem rotations with different seeds."""

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

def eval_with_seed(student, tokenizer, local_data_path, seed, eval_num):
    """Evaluate on 20 problems with a specific random seed."""
    with open(local_data_path, 'r') as f:
        num_problems = sum(1 for _ in f)

    all_indices = list(range(num_problems))
    random.seed(seed)
    eval_indices = random.sample(all_indices, 20)

    eval_ds = LocalLeetCodeDataset(tokenizer, local_data_path, indices=eval_indices)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    passed = 0
    for batch in eval_loader:
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
        except:
            pass

    accuracy = (passed / 20) * 100
    return accuracy, passed

def main():
    local_data_path = "/home/kenpeter/work/data/high_quality_leetcode/train.jsonl"

    # Load checkpoint
    if not os.path.exists("checkpoints/final.pt"):
        print("❌ final.pt not found")
        return

    print(f"📁 Loading checkpoint: final.pt")
    print(f"🧪 Robustness Test: Multiple 20-problem rotations")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )

    torch.cuda.empty_cache()
    ckpt_data = torch.load("checkpoints/final.pt", map_location="cpu")
    config = StudentConfig(**ckpt_data["config"])
    student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
    student.load_state_dict(ckpt_data["model"])
    student.eval()
    torch.cuda.empty_cache()

    print(f"✅ Model loaded\n")

    # Test with multiple seeds
    seeds = [42, 123, 456, 789, 999]  # 5 different rotations
    results = []

    print(f"Evaluating {len(seeds)} different 20-problem rotations:\n")

    for i, seed in enumerate(seeds):
        accuracy, passed = eval_with_seed(student, tokenizer, local_data_path, seed, i+1)
        results.append((seed, accuracy, passed))
        status = "✓" if accuracy >= 95 else "⚠️" if accuracy >= 80 else "❌"
        print(f"  Rotation {i+1} (seed={seed:3d}): {accuracy:5.1f}% ({passed:2d}/20) {status}")

    print("\n" + "=" * 70)

    avg_acc = sum(r[1] for r in results) / len(results)
    min_acc = min(r[1] for r in results)
    max_acc = max(r[1] for r in results)

    print(f"\n📊 ROBUSTNESS ANALYSIS:")
    print(f"   Average: {avg_acc:.1f}%")
    print(f"   Min:     {min_acc:.1f}%")
    print(f"   Max:     {max_acc:.1f}%")
    print(f"   Range:   {max_acc - min_acc:.1f}% (variability)")

    if avg_acc >= 95:
        print(f"\n✅ ROBUST: Model consistently performs well across different problem sets")
    elif avg_acc >= 85:
        print(f"\n⚠️  MODERATE: Some variability, but generally acceptable")
    else:
        print(f"\n❌ UNSTABLE: High variability suggests dataset-specific overfitting")

    print("=" * 70)

if __name__ == "__main__":
    main()
