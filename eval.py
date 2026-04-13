#!/usr/bin/env python3
"""OFFICIAL evaluation script - runs BOTH metrics together.

NEW METRIC (eval_real): Test case passing - does solution actually work?
GUARD METRIC (eval_code): Code structure - does output look like code?

This script CANNOT CHANGE. Both metrics MUST ALWAYS be reported together.
"""

import torch
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import StudentConfig
from model.student import StudentModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

class LocalLeetCodeDataset(Dataset):
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
    ckpt_files = sorted(
        [f for f in os.listdir("checkpoints") if f.startswith("step_") and f.endswith(".pt")],
        key=lambda x: int(x[len("step_"):].replace(".pt", ""))
    )

    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        ckpt_path = f"checkpoints/{latest_ckpt}"
    elif os.path.exists("checkpoints/final.pt"):
        ckpt_path = "checkpoints/final.pt"
        latest_ckpt = "final.pt"
    else:
        print("No checkpoints found")
        return

    print(f"Loading checkpoint: {latest_ckpt}")

    # Load tokenizer
    teacher_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Jackrong--Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/af2f37de41af0bdcaea5e3790ad323030ea4af07"
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True, local_files_only=True)

    # Load student
    torch.cuda.empty_cache()
    ckpt_data = torch.load(ckpt_path, map_location="cuda")
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
    random.seed(42)
    eval_indices = random.sample(all_indices, min(20, num_problems))

    print(f"\nEval on {len(eval_indices)} random problems from {num_problems} total")
    print("=" * 70)

    eval_ds = LocalLeetCodeDataset(tokenizer, local_data_path, indices=eval_indices)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    # Load all data items once
    with open(local_data_path, 'r') as f:
        all_items = [json.loads(line) for line in f]

    # Track both metrics
    code_passed = 0  # Has keywords: def, class, return
    test_passed = 0  # Actually passes all test cases

    for i, batch in enumerate(eval_loader):
        input_ids = batch["input_ids"].to("cuda")
        problem_idx = eval_indices[i]
        problem_item = all_items[problem_idx]

        try:
            with torch.no_grad():
                output = student.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    temperature=0.5,
                    top_p=0.9,
                )
            solution = tokenizer.decode(output[0], skip_special_tokens=True)

            # GUARD METRIC: Does output contain code keywords?
            has_code = any(x in solution for x in ['def ', 'class ', 'return '])
            if has_code:
                code_passed += 1
                code_status = "✓"
            else:
                code_status = "✗"

            # REAL METRIC: Run the actual test cases from the data
            test_status = "✗"
            try:
                # Extract the test function and entry point from problem data
                test_code = problem_item.get('test', '')
                entry_point = problem_item.get('entry_point', '')

                if test_code and entry_point:
                    # Parse entry point (e.g., "Solution().twoSum" -> "Solution", "twoSum")
                    # Combine generated solution + test code
                    full_code = solution + "\n\n" + test_code

                    # Execute the code in a safe namespace
                    namespace = {}
                    exec(full_code, namespace)

                    # Call the check function with the solution
                    # Parse the entry point to get the function
                    parts = entry_point.split('(')
                    class_name = parts[0]  # e.g., "Solution"

                    # Instantiate the class and get the check function
                    if class_name in namespace:
                        solution_instance = namespace[class_name]()
                        check_fn = namespace['check']
                        check_fn(solution_instance)

                        # If we get here, all assertions passed
                        test_passed += 1
                        test_status = "✓"
            except AssertionError as e:
                # Test failed
                test_status = "✗"
            except Exception as e:
                # Execution error
                test_status = "E"

        except Exception as e:
            code_status = "E"
            test_status = "E"

        print(f"[{i+1:2d}/{len(eval_indices)}] Code:{code_status}  Test:{test_status}")

    code_accuracy = (code_passed / len(eval_indices) * 100) if len(eval_indices) > 0 else 0
    test_accuracy = (test_passed / len(eval_indices) * 100) if len(eval_indices) > 0 else 0

    print("=" * 70)
    print(f"\n📊 OFFICIAL EVALUATION RESULTS:")
    print(f"\n   🔴 REAL METRIC (Test Case Passing):")
    print(f"      {test_passed}/{len(eval_indices)} = {test_accuracy:.1f}%")
    print(f"      ↳ Does generated solution actually solve the problem?")
    print(f"\n   🟡 GUARD METRIC (Code Structure):")
    print(f"      {code_passed}/{len(eval_indices)} = {code_accuracy:.1f}%")
    print(f"      ↳ Does output contain code keywords (def, class, return)?")
    print(f"\n   📁 Checkpoint: {latest_ckpt}")
    print()

    # Save results
    results = {
        "checkpoint": latest_ckpt,
        "test_metric": {
            "name": "Test Case Passing (REAL)",
            "passed": test_passed,
            "total": len(eval_indices),
            "accuracy": test_accuracy
        },
        "code_metric": {
            "name": "Code Structure (GUARD)",
            "passed": code_passed,
            "total": len(eval_indices),
            "accuracy": code_accuracy
        }
    }

    with open("checkpoints/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to checkpoints/eval_results.json")

if __name__ == "__main__":
    main()
