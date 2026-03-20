"""
Evaluate fresh-trained nanoGPT model on LeetCode test set.
Checks: code compilation + test case execution.
"""

import os
import sys
import json
import re
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import tiktoken
import torch
import numpy as np

# Add nanoGPT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nanoGPT"))
from model import GPTConfig, GPT

# ── Config ──────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "nanoGPT", "out-leetcode")
TEST_DATA = "/home/kenpeter/work/data/newfacade_LeetCodeDataset/leetcode_test.jsonl"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.8
TOP_K = 200
MAX_PROBLEMS = 228  # All test problems
TIMEOUT_SECONDS = 10


# ── Load model ──────────────────────────────────────────────────────────
def load_model():
    ckpt_path = os.path.join(OUT_DIR, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        print("Run training first: python nanoGPT/train.py config/train_leetcode.py")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # Remove _orig_mod prefix from compiled model state dict
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model


# ── Generate ────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, prompt_text, enc):
    """Generate code given a problem prompt."""
    prompt_tokens = enc.encode_ordinary(prompt_text)
    x = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")

    # Use model's built-in generate
    y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
    generated = enc.decode(y[0][len(prompt_tokens) :].tolist())
    return generated


# ── Extract code ────────────────────────────────────────────────────────
def extract_code(text):
    # Try ```python ... ``` block
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try ``` ... ``` without language
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Look for class Solution
    if "class Solution" in text:
        idx = text.index("class Solution")
        # Find end of code (next ### or end of text)
        end = len(text)
        for marker in ["###", "```", "\n\n\n"]:
            pos = text.find(marker, idx + 20)
            if pos != -1:
                end = min(end, pos)
        return text[idx:end].strip()

    return text.strip()


# ── Compile check ───────────────────────────────────────────────────────
def check_compile(code):
    try:
        compile(code, "<generated>", "exec")
        return True, None
    except SyntaxError as e:
        return False, str(e)


# ── Run tests ───────────────────────────────────────────────────────────
def run_tests(code, test_code, entry_point, prompt_code=""):
    namespace = {}
    try:
        exec(prompt_code, namespace)
        exec(code, namespace)
    except Exception as e:
        return 0, 0, f"Code exec error: {e}"

    try:
        parts = entry_point.split(".")
        obj = namespace
        for part in parts:
            obj = getattr(obj, part)
    except AttributeError:
        return 0, 0, f"Entry point '{entry_point}' not found"

    test_ns = dict(namespace)
    try:
        exec(test_code, test_ns)
        check_fn = test_ns.get("check")
        if check_fn is None:
            return 0, 0, "No check function"

        def run_check():
            check_fn(obj)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_check)
            future.result(timeout=TIMEOUT_SECONDS)

        return 1, 1, None
    except AssertionError as e:
        return 0, 1, f"AssertionError: {e}"
    except FuturesTimeout:
        return 0, 1, f"Timeout"
    except Exception as e:
        return 0, 1, f"{type(e).__name__}: {e}"


# ── Main ────────────────────────────────────────────────────────────────
def main():
    enc = tiktoken.get_encoding("gpt2")
    model = load_model()

    problems = []
    with open(TEST_DATA) as f:
        for i, line in enumerate(f):
            if i >= MAX_PROBLEMS:
                break
            problems.append(json.loads(line))
    print(f"Evaluating on {len(problems)} problems...\n")

    results = {"total": 0, "compiles": 0, "passes_tests": 0, "details": []}

    for i, prob in enumerate(problems):
        task_id = prob.get("task_id", f"problem_{i}")
        desc = prob.get("problem_description", "")
        prompt = f"### Problem:\n{desc}\n\n### Solution:\n```python\n"

        print(
            f"[{i + 1}/{len(problems)}] {task_id} ({prob.get('difficulty', '?')})",
            end=" ... ",
        )

        generated = generate(model, prompt, enc)
        code = extract_code(generated)
        compiles, cerr = check_compile(code)

        passes, total, terr = 0, 0, None
        if compiles:
            passes, total, terr = run_tests(
                code,
                prob.get("test", ""),
                prob.get("entry_point", "Solution"),
                prob.get("prompt", ""),
            )

        results["total"] += 1
        if compiles:
            results["compiles"] += 1
        if passes > 0:
            results["passes_tests"] += 1

        status = "PASS" if passes > 0 else ("COMPILE" if compiles else "FAIL")
        print(f"{status} (compile={'Y' if compiles else 'N'}, tests={passes}/{total})")

        results["details"].append(
            {
                "task_id": task_id,
                "difficulty": prob.get("difficulty"),
                "compiles": compiles,
                "compile_error": cerr,
                "tests_passed": passes,
                "tests_total": total,
                "test_error": terr,
                "generated_code": code[:500],
            }
        )

    # Summary
    t = results["total"]
    print(f"\n{'=' * 60}")
    print(
        f"RESULTS: {results['compiles']}/{t} compile ({results['compiles'] / max(t, 1) * 100:.0f}%), "
        f"{results['passes_tests']}/{t} pass ({results['passes_tests'] / max(t, 1) * 100:.0f}%)"
    )
    for diff in ["Easy", "Medium", "Hard"]:
        sub = [d for d in results["details"] if d["difficulty"] == diff]
        if sub:
            pc = sum(1 for d in sub if d["tests_passed"] > 0)
            print(f"  {diff}: {pc}/{len(sub)}")

    out = os.path.join(OUT_DIR, "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
