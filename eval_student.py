#!/usr/bin/env python3
"""Evaluate student model on LeetCode problems."""

import os
import sys
import argparse
import torch
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.config import StudentConfig
from model.student import StudentModel
from transformers import AutoTokenizer
from eval.leetcode_eval import run_leetcode_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index of problems")
    parser.add_argument(
        "--num", type=int, default=50, help="Number of problems to evaluate"
    )
    args = parser.parse_args()

    # Configuration
    checkpoint_path = "/home/kenpeter/work/mulmodel_ext/checkpoints/final.pt"
    teacher_id = "Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = getattr(
            torch.cuda.get_device_properties(0), "total_memory", None
        ) or getattr(torch.cuda.get_device_properties(0), "total_mem", 0)
        print(f"VRAM: {total_mem / 1e9:.1f} GB")

    # Load tokenizer from teacher
    print(f"\nLoading tokenizer from: {teacher_id}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_id, trust_remote_code=True)

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config from checkpoint
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = StudentConfig.from_dict(config_dict)
        print(f"Loaded config from checkpoint")
    else:
        # Use default config
        config = StudentConfig()
        print(f"Using default config (no config in checkpoint)")

    print(
        f"Model config: {config.hidden_size} hidden, {config.num_hidden_layers} layers, {config.num_attention_heads} heads"
    )

    # Create and load model
    model = StudentModel(config)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")

    # Check if we have eval results for comparison
    eval_results_path = "/home/kenpeter/work/mulmodel_ext/checkpoints/eval_results.json"
    if os.path.exists(eval_results_path):
        with open(eval_results_path, "r") as f:
            prev_results = json.load(f)
        print(f"Previous eval results: {prev_results.get('accuracy', 'N/A')}% accuracy")

    # Run evaluation
    print("\n" + "=" * 60)
    print("Starting LeetCode Evaluation")
    print("=" * 60)

    results = run_leetcode_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=512,
        num_problems=args.num,
        split="test",
        dataset_name="justindal/leetcode-python-dataset",
        start_index=args.start,
    )

    # Save results
    output_dir = "/home/kenpeter/work/mulmodel_ext/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    eval_path = os.path.join(
        output_dir, f"eval_results_final_{args.start}_{args.start + args.num - 1}.json"
    )
    save_results = {k: v for k, v in results.items() if k != "results"}
    save_results["detailed_results"] = [
        {k: v for k, v in r.items()} for r in results["results"]
    ]

    with open(eval_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nEval results saved to: {eval_path}")
    print(
        f"\nSummary: {results['passed']}/{results['total']} problems passed ({results['accuracy']:.1f}% accuracy)"
    )


if __name__ == "__main__":
    main()
