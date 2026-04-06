#!/usr/bin/env python3
"""Ultra simple: Just paste your problem and get solution."""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import StudentConfig
from model.student import StudentModel
from transformers import AutoTokenizer


def load_model(checkpoint_path: str = "./checkpoints/step_13200.pt"):
    """Load ONLY the student model (no teacher needed)."""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("Loading student model...", end="", flush=True)

    # Load tokenizer (Qwen tokenizer - no teacher model needed)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )

    # Load ONLY the student model from checkpoint
    ckpt_data = torch.load(checkpoint_path, map_location="cuda")
    config = StudentConfig(**ckpt_data["config"])
    model = StudentModel(config).to(dtype=torch.bfloat16, device="cuda")
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    print(" ✅\n")
    return model, tokenizer


def clean_input(text: str) -> str:
    """Clean up messy LeetCode pasted text."""
    # Remove common HTML artifacts
    text = text.replace('&nbsp;', ' ')
    text = text.replace('<br>', '\n')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('<p>', '\n')
    text = text.replace('</p>', '\n')

    # Remove "Code", "Testcase", "Test Result" headers
    lines = text.split('\n')
    cleaned = []
    skip_next = False

    for i, line in enumerate(lines):
        # Skip navigation headers
        if any(x in line.lower() for x in ['code', 'testcase', 'test result', 'submit', 'solutions', 'discuss', '...']):
            if len(cleaned) == 0 or len(line.strip()) < 3:
                continue

        cleaned.append(line)

    # Join and clean up whitespace
    text = '\n'.join(cleaned)

    # Remove excessive blank lines
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')

    # Remove any weird unicode/control characters
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')

    return text.strip()


def solve(problem_text: str, model, tokenizer, max_tokens: int = 512):
    """Solve a problem."""
    try:
        # Clean the input first
        problem_text = clean_input(problem_text)

        # Limit to first 400 chars to avoid token overflow
        if len(problem_text) > 400:
            problem_text = problem_text[:400]

        inputs = tokenizer(
            problem_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to("cuda")

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=256,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
            )

        solution = tokenizer.decode(output[0], skip_special_tokens=True)
        return solution

    except Exception as e:
        return f"❌ Error: {str(e)}"


def main():
    """Main entry point."""
    print("=" * 70)
    print("🚀 LeetCode Solver")
    print("=" * 70)
    print()

    model, tokenizer = load_model()

    while True:
        print("=" * 70)
        print("📋 Paste your LeetCode problem below")
        print("(Press Ctrl+D or Cmd+D when done)")
        print("=" * 70)
        print()

        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        problem_text = "\n".join(lines).strip()

        if not problem_text:
            print("❌ Empty input. Try again.\n")
            continue

        # Show what we got
        preview = "\n".join(problem_text.split("\n")[:3])
        if len(problem_text.split("\n")) > 3:
            preview += f"\n... ({len(problem_text.split('\n')) - 3} more lines)"

        print("\n" + "=" * 70)
        print("📖 Got:")
        print("=" * 70)
        print(preview)
        print("=" * 70)

        print("\n🤖 Solving...", end="", flush=True)
        solution = solve(problem_text, model, tokenizer)
        print(" ✅\n")

        print("=" * 70)
        print("✅ SOLUTION:")
        print("=" * 70)
        print(solution)
        print("=" * 70)

        print("\nSolve another? (y/n): ", end="")
        if input().strip().lower() != "y":
            print("Bye! 👋")
            break

        print()


if __name__ == "__main__":
    main()
