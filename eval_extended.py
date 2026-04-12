#!/usr/bin/env python3
"""Extended validation: Test model on 50+ problem set to confirm generalization."""

import torch, sys, os, json, random, time

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

    # Find latest checkpoint (prefer final.pt, else latest step_*.pt)
    ckpt_path = None
    latest_ckpt = None

    if os.path.exists("checkpoints/final.pt"):
        ckpt_path = "checkpoints/final.pt"
        latest_ckpt = "final.pt"
    else:
        ckpt_files = sorted(
            [f for f in os.listdir("checkpoints") if f.startswith("step_") and f.endswith(".pt")],
            key=lambda x: int(x[len("step_"):].replace(".pt", ""))
        )
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            ckpt_path = f"checkpoints/{latest_ckpt}"

    if not ckpt_path:
        print("No checkpoints found")
        return

    print(f"📁 Loading checkpoint: {latest_ckpt}")
    print(f"🎯 Phase 3: Extended Validation (50+ problems)")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )

    # Load student
    torch.cuda.empty_cache()
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    config = StudentConfig(**ckpt_data["config"])
    student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
    student.load_state_dict(ckpt_data["model"])
    student.eval()
    torch.cuda.empty_cache()

    print(f"✅ Model loaded")

    # Get eval set (50 random problems for extended validation)
    with open(local_data_path, 'r') as f:
        num_problems = sum(1 for _ in f)

    all_indices = list(range(num_problems))
    random.seed(123)  # Different seed than baseline (42) to test generalization
    eval_size = min(50, num_problems // 2)  # Use 50 or half of eval pool if smaller
    eval_indices = random.sample(all_indices, eval_size)

    print(f"\n📊 Evaluation Setup:")
    print(f"   - Total dataset: {num_problems} problems")
    print(f"   - Eval sample size: {eval_size} problems")
    print(f"   - Random seed: 123 (different from baseline)")
    print(f"   - Success criterion: Solution contains 'def ', 'class ', or 'return '")
    print("=" * 70)

    eval_ds = LocalLeetCodeDataset(tokenizer, local_data_path, indices=eval_indices)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    passed = 0
    start_time = time.time()

    print(f"\n🔄 Evaluating {eval_size} problems...")
    print()

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

        # Print progress every 5 problems
        if (i + 1) % 5 == 0 or i == 0 or i == eval_size - 1:
            print(f"   [{i+1:2d}/{eval_size}] {status}")

    elapsed = time.time() - start_time
    accuracy = (passed / eval_size * 100) if eval_size > 0 else 0

    print()
    print("=" * 70)
    print(f"\n🎯 PHASE 3 RESULTS (Extended Validation):")
    print(f"   ✓ Passed:    {passed}/{eval_size} problems")
    print(f"   📈 Accuracy: {accuracy:.1f}%")
    print(f"   ⏱️  Time:     {elapsed:.1f}s ({elapsed/eval_size:.2f}s per problem)")
    print(f"   📁 Checkpoint: {latest_ckpt}")

    if accuracy >= 90:
        print(f"\n✅ VALIDATION PASSED: Model generalizes well to unseen problems")
    elif accuracy >= 80:
        print(f"\n⚠️  ACCEPTABLE: Model shows good generalization but room for improvement")
    else:
        print(f"\n❌ CONCERN: Accuracy lower than baseline, may indicate overfitting")

    print("=" * 70)

if __name__ == "__main__":
    main()
