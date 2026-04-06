"""Inline eval harness — called from the training loop every eval_every steps."""
import ast
import json
import random
import sys
import os
from pathlib import Path
import torch

DATASET_PATH = "/home/kenpeter/work/data/high_quality_leetcode/train.jsonl"

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval_quick import extract_solution, run_test


def load_eval_problems(n: int = 5, seed: int = 42) -> list[dict]:
    """Load N held-out problems (question_id % 10 == 0), balanced by difficulty."""
    by_diff: dict[str, list] = {}
    with open(DATASET_PATH) as f:
        for line in f:
            d = json.loads(line)
            if int(d.get("question_id", 1)) % 10 != 0:
                continue
            diff = d.get("difficulty", "Medium")
            by_diff.setdefault(diff, []).append(d)

    rng = random.Random(seed)  # local RNG — does NOT affect global random state
    tiers = sorted(by_diff.keys())
    per_tier = n // len(tiers)
    remainder = n % len(tiers)

    selected = []
    for i, tier in enumerate(tiers):
        k = per_tier + (1 if i < remainder else 0)
        pool = by_diff.get(tier, [])
        selected.extend(rng.sample(pool, min(k, len(pool))))

    rng.shuffle(selected)
    return selected[:n]


def run_inline_eval(model, tokenizer, device, n_problems=5, max_new_tokens=256) -> dict:
    """
    Eval model on n_problems held-out LeetCode problems.
    Caller MUST call model.train() after this returns.
    """
    model.eval()
    problems = load_eval_problems(n=n_problems)
    results = []

    with torch.no_grad():
        for prob in problems:
            prompt = prob["problem_description"] + "\n\n" + prob.get("starter_code", "")
            test_code = prob["test"]
            entry_pt = prob["entry_point"]

            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
            )

            new_ids = output_ids[0, input_ids.shape[1]:]
            generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            solution_code = extract_solution(generated_text)

            syntax_ok = False
            try:
                ast.parse(solution_code)
                syntax_ok = True
            except SyntaxError:
                pass

            test_passed = False
            if syntax_ok:
                test_passed, _ = run_test(solution_code, test_code, entry_pt)

            results.append({
                "task_id": prob.get("task_id", "?"),
                "difficulty": prob.get("difficulty", "?"),
                "syntax_ok": syntax_ok,
                "test_passed": test_passed,
            })

    n_total = len(results)
    n_syntax = sum(r["syntax_ok"] for r in results)
    n_passed = sum(r["test_passed"] for r in results)

    return {
        "n_total": n_total,
        "n_syntax": n_syntax,
        "n_passed": n_passed,
        "pass_rate": n_passed / max(n_total, 1),
        "per_problem": results,
    }
