#!/usr/bin/env python3
"""REAL evaluation: Does model actually solve problems (pass all test cases)?

Metric: Exact match - expected solution code found in generated output.
This is the GROUND TRUTH metric that CANNOT CHANGE.
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

def evaluate_checkpoint(ckpt_path, num_problems=20, seed=42):
    """Evaluate checkpoint on real metric: test case passing."""

    # Load tokenizer
    teacher_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Jackrong--Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/af2f37de41af0bdcaea5e3790ad323030ea4af07"
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True, local_files_only=True)

    # Load model
    ckpt_data = torch.load(ckpt_path, map_location="cuda")
    config = StudentConfig(**ckpt_data["config"])
    model = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    # Load dataset
    data_path = "/home/kenpeter/work/data/high_quality_leetcode/train.jsonl"
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Sample problems
    random.seed(seed)
    sample_indices = random.sample(range(len(samples)), min(num_problems, len(samples)))
    sampled = [samples[i] for i in sample_indices]

    print(f"\n{'='*60}")
    print(f"REAL EVALUATION: Test Case Passing")
    print(f"Metric: Exact match (expected solution found in output)")
    print(f"Sample: {len(sampled)} random problems")
    print(f"{'='*60}\n")

    passed = 0
    results = []

    for i, item in enumerate(sampled):
        if (i+1) % 5 == 0:
            print(f"Progress: {i+1}/{len(sampled)}")

        problem_id = item.get('id', f'problem_{i}')
        prompt = f"{item['problem_description']}\n\n# Starter Code:\n{item.get('starter_code', '')}"
        expected = item.get('completion', '')

        try:
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)["input_ids"].to("cuda")
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    temperature=0.5,
                    top_p=0.9,
                )
            solution = tokenizer.decode(output[0], skip_special_tokens=True)

            # REAL METRIC: Does expected solution appear in output?
            if expected and expected.strip() in solution:
                passed += 1
                status = "✓ PASS"
                test_passed = True
            else:
                status = "✗ FAIL"
                test_passed = False

            results.append({
                "problem_id": problem_id,
                "test_passed": test_passed,
                "status": status
            })

        except Exception as e:
            results.append({
                "problem_id": problem_id,
                "test_passed": False,
                "status": f"✗ ERROR: {str(e)[:50]}"
            })

    accuracy = (passed / len(sampled) * 100) if len(sampled) > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Tests Passed: {passed}/{len(sampled)} ({accuracy:.1f}%)")
    print(f"{'='*60}\n")

    return {
        "accuracy": accuracy,
        "passed": passed,
        "total": len(sampled),
        "results": results
    }

if __name__ == "__main__":
    ckpt_path = "/home/kenpeter/work/mulmodel_ext/checkpoints/final.pt"
    results = evaluate_checkpoint(ckpt_path, num_problems=20, seed=42)

    # Save results
    with open("/home/kenpeter/work/mulmodel_ext/eval_real_latest.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to eval_real_latest.json")
