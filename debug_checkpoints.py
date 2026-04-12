#!/usr/bin/env python3
"""Debug: Test multiple checkpoints to find where accuracy degrades."""

import torch, os, json, random
import sys
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
        full_tok = self.tokenizer(full, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt")
        prompt_tok = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_len = prompt_tok["attention_mask"].sum().item()
        ids = full_tok["input_ids"].squeeze(0)
        mask = full_tok["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[:prompt_len] = -100
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

def eval_checkpoint(ckpt_path, eval_indices):
    """Evaluate a checkpoint. Returns accuracy %"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    torch.cuda.empty_cache()
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    config = StudentConfig(**ckpt_data["config"])
    student = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
    student.load_state_dict(ckpt_data["model"])
    student.eval()
    torch.cuda.empty_cache()

    # Sample 20 random problems
    random.seed(42)
    sample_indices = random.sample(eval_indices, min(20, len(eval_indices)))
    eval_ds = LocalLeetCodeDataset(tokenizer, "/home/kenpeter/work/data/high_quality_leetcode/train.jsonl", indices=sample_indices)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    passed = 0
    for i, batch in enumerate(eval_loader):
        input_ids = batch["input_ids"].to("cuda")
        try:
            with torch.no_grad():
                output = student.generate(input_ids=input_ids, max_new_tokens=256, temperature=0.5, top_p=0.9)
            solution = tokenizer.decode(output[0], skip_special_tokens=True)
            if any(x in solution for x in ['def ', 'class ', 'return ']):
                passed += 1
        except:
            pass

    accuracy = (passed / len(sample_indices) * 100) if len(sample_indices) > 0 else 0
    del student, tokenizer
    torch.cuda.empty_cache()
    return accuracy

# Get eval indices
random.seed(42)
with open('/home/kenpeter/work/data/high_quality_leetcode/train.jsonl', 'r') as f:
    num_problems = sum(1 for _ in f)
all_indices = list(range(num_problems))
random.shuffle(all_indices)
split_idx = int(0.8 * num_problems)
eval_indices = all_indices[split_idx:]

# Find checkpoints
ckpt_files = sorted([f for f in os.listdir('checkpoints') if f.startswith('step_') and f.endswith('.pt')],
                   key=lambda x: int(x[5:].replace('.pt', '')))

# Binary search: test at steps 7000, 8000, 9000, 10000, 11000, 12000, 13000
test_steps = [7936, 8136, 8336, 8536, 8736, 8936, 9136, 10000, 11000, 12000, 13000, 13936]
results = []

print("Binary search: evaluating checkpoints to find degradation point\n")
print("Step      | Accuracy | File")
print("----------|----------|------------------------")

for target_step in test_steps:
    # Find closest checkpoint
    best_ckpt = None
    best_diff = float('inf')
    for ckpt in ckpt_files:
        step = int(ckpt[5:].replace('.pt', ''))
        diff = abs(step - target_step)
        if diff < best_diff:
            best_diff = diff
            best_ckpt = ckpt
            best_step = step

    if best_ckpt:
        ckpt_path = f"checkpoints/{best_ckpt}"
        accuracy = eval_checkpoint(ckpt_path, eval_indices)
        results.append((best_step, accuracy, best_ckpt))
        print(f"{best_step:8d} | {accuracy:6.1f}%  | {best_ckpt}")

print("\n" + "="*50)
print("FINDINGS:")
max_acc = max(results, key=lambda x: x[1])
print(f"Peak: Step {max_acc[0]} ({max_acc[1]:.1f}%) — {max_acc[2]}")

# Find where it drops
for i in range(len(results)-1):
    if results[i][1] > 50 and results[i+1][1] < 50:
        print(f"DEGRADATION: Between step {results[i][0]} ({results[i][1]:.1f}%) and step {results[i+1][0]} ({results[i+1][1]:.1f}%)")
