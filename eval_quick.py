"""
eval_quick.py — Quick eval: can the student model solve LeetCode problems?

Usage:
    python eval_quick.py                          # auto-detect latest checkpoint, 10 problems, GPU
    python eval_quick.py --cpu                    # run on CPU
    python eval_quick.py --n 20                   # evaluate 20 problems
    python eval_quick.py --checkpoint checkpoints/step_4400.pt
"""

import argparse
import ast
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Helper data structures needed by some LeetCode test harnesses
# ──────────────────────────────────────────────────────────────────────────────

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    def __repr__(self):
        vals, cur = [], self
        while cur:
            vals.append(cur.val)
            cur = cur.next
        return f"ListNode({vals})"


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def list_node(vals):
    """Build a linked list from a list of values (None entries are skipped)."""
    if not vals:
        return None
    nodes = [ListNode(v) if v is not None else None for v in vals]
    for i in range(len(nodes) - 1):
        if nodes[i] is not None:
            nodes[i].next = nodes[i + 1]
    return nodes[0]


def tree_node(vals):
    """Build a binary tree from level-order list (None = missing node)."""
    if not vals:
        return None
    root = TreeNode(vals[0])
    queue = [root]
    i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if node is None:
            continue
        if i < len(vals):
            v = vals[i]; i += 1
            node.left = TreeNode(v) if v is not None else None
            queue.append(node.left)
        if i < len(vals):
            v = vals[i]; i += 1
            node.right = TreeNode(v) if v is not None else None
            queue.append(node.right)
    return root


def is_same_list(a, b):
    while a and b:
        if a.val != b.val:
            return False
        a, b = a.next, b.next
    return a is None and b is None


def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    return p.val == q.val and is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


# Namespace available inside every executed test block
_TEST_GLOBALS = {
    "ListNode": ListNode,
    "TreeNode": TreeNode,
    "list_node": list_node,
    "tree_node": tree_node,
    "is_same_list": is_same_list,
    "is_same_tree": is_same_tree,
}


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return the path to the highest-step checkpoint (or final.pt)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None

    # Prefer final.pt
    final = ckpt_dir / "final.pt"
    if final.exists():
        return str(final)

    # Find step_NNN.pt files and pick the highest
    step_files = list(ckpt_dir.glob("step_*.pt"))
    if not step_files:
        return None

    def step_num(p):
        m = re.search(r"step_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    return str(max(step_files, key=step_num))


def load_model(checkpoint_path: str, device: torch.device):
    """Load StudentModel from a checkpoint."""
    # Import here so this file is importable without the model package installed
    sys.path.insert(0, str(Path(__file__).parent))
    from model.student import StudentModel
    from model.config import StudentConfig

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # The checkpoint may store config as a dict or as a StudentConfig instance
    if "config" in ckpt:
        cfg_raw = ckpt["config"]
        config = StudentConfig.from_dict(cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw
    else:
        config = StudentConfig()

    # Disable sage_attention for eval (it may require special GPU kernels)
    config.sage_attention = False

    model = StudentModel(config)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    # Strip unexpected "module." prefix from DDP checkpoints
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"Model loaded — {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Dataset sampling
# ──────────────────────────────────────────────────────────────────────────────

DATASET_PATH = "/home/kenpeter/work/data/high_quality_leetcode/train.jsonl"


def load_problems(n: int):
    """
    Sample n problems with an even easy/medium/hard distribution.
    Falls back to random sampling if there are not enough of a given tier.
    """
    by_diff: dict[str, list] = {"Easy": [], "Medium": [], "Hard": []}
    with open(DATASET_PATH) as f:
        for line in f:
            d = json.loads(line)
            diff = d.get("difficulty", "Medium")
            by_diff.setdefault(diff, []).append(d)

    import random
    random.seed(42)

    tiers = list(by_diff.keys())
    per_tier = n // len(tiers)
    remainder = n % len(tiers)

    selected = []
    for i, tier in enumerate(tiers):
        k = per_tier + (1 if i < remainder else 0)
        pool = by_diff.get(tier, [])
        selected.extend(random.sample(pool, min(k, len(pool))))

    random.shuffle(selected)
    return selected[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Code extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_solution(text: str) -> str:
    """
    Try to pull a class Solution block out of generated text.
    Returns the raw extracted string (may be empty if nothing found).
    """
    # 1. Prefer fenced code block
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # 2. Try to grab from "class Solution" to end
    cls_match = re.search(r"(class Solution\b.*)", text, re.DOTALL)
    if cls_match:
        return cls_match.group(1).strip()

    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Test execution
# ──────────────────────────────────────────────────────────────────────────────

def run_test(solution_code: str, test_code: str, entry_point: str, timeout: int = 5) -> tuple[bool, str]:
    """
    Execute the generated Solution against the dataset's check() function.

    entry_point examples: "Solution().twoSum"  →  method = "twoSum"

    The test code defines:
        def check(candidate): ...

    We call:
        check(Solution().{method})

    Returns (passed: bool, error_message: str).
    """
    # Derive the method name from entry_point
    # e.g. "Solution().twoSum" → "twoSum"
    method_match = re.search(r"Solution\(\)\.(\w+)", entry_point)
    if not method_match:
        return False, f"Cannot parse entry_point: {entry_point!r}"
    method = method_match.group(1)

    full_code = f"""
{solution_code}

{test_code}

check(Solution().{method})
"""

    globs = dict(_TEST_GLOBALS)
    try:
        exec(compile(full_code, "<eval>", "exec"), globs)
        return True, ""
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cpu" if args.cpu else "cuda")
    if not args.cpu and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        device = torch.device("cpu")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    TEACHER_TOKENIZER = (
        os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--Jackrong--Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled/"
            "snapshots/af2f37de41af0bdcaea5e3790ad323030ea4af07"
        )
    )
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {TEACHER_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER, local_files_only=True)

    # ── Find / load checkpoint ───────────────────────────────────────────────
    ckpt_path = args.checkpoint or find_latest_checkpoint("checkpoints")
    if ckpt_path is None:
        sys.exit("No checkpoint found in checkpoints/. Use --checkpoint.")

    model = load_model(ckpt_path, device)

    # ── Load problems ────────────────────────────────────────────────────────
    problems = load_problems(args.n)
    print(f"\nEvaluating {len(problems)} problems\n{'=' * 60}")

    results = []
    for idx, prob in enumerate(problems):
        task_id    = prob.get("task_id", f"problem_{idx}")
        difficulty = prob.get("difficulty", "?")
        prompt     = prob["problem_description"] + "\n\n" + prob.get("starter_code", "")
        test_code  = prob["test"]
        entry_pt   = prob["entry_point"]

        print(f"\n[{idx+1}/{len(problems)}] {task_id}  ({difficulty})")
        print(f"  Entry: {entry_pt}")

        # ── Tokenise & generate ──────────────────────────────────────────────
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        # ── Extract solution code ────────────────────────────────────────────
        solution_code = extract_solution(generated_text)

        # ── Syntax check ────────────────────────────────────────────────────
        syntax_ok = False
        try:
            ast.parse(solution_code)
            syntax_ok = True
        except SyntaxError:
            pass

        # ── Functional test ─────────────────────────────────────────────────
        test_passed = False
        test_error  = "syntax error — skipped"
        if syntax_ok:
            test_passed, test_error = run_test(solution_code, test_code, entry_pt)

        # ── Print per-problem result ─────────────────────────────────────────
        status = "PASS" if test_passed else ("SYNTAX_ERR" if not syntax_ok else "FAIL")
        print(f"  Status   : {status}")
        if not test_passed:
            print(f"  Error    : {test_error[:120]}")
        preview = solution_code[:200].replace("\n", " | ")
        print(f"  Preview  : {preview}")

        results.append(
            dict(
                task_id=task_id,
                difficulty=difficulty,
                syntax_ok=syntax_ok,
                test_passed=test_passed,
                error=test_error,
                generated_preview=solution_code[:300],
            )
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    n_total   = len(results)
    n_syntax  = sum(r["syntax_ok"]    for r in results)
    n_passed  = sum(r["test_passed"]  for r in results)

    print(f"\n{'=' * 60}")
    print(f"RESULTS  ({n_total} problems)")
    print(f"  Syntax OK    : {n_syntax}/{n_total}  ({100*n_syntax/n_total:.0f}%)")
    print(f"  Tests Passed : {n_passed}/{n_total}  ({100*n_passed/n_total:.0f}%)")

    by_diff: dict[str, list] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r["test_passed"])
    for diff, passed_list in sorted(by_diff.items()):
        k = sum(passed_list)
        print(f"  {diff:8s}: {k}/{len(passed_list)}")

    print(f"{'=' * 60}\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quick LeetCode eval for student model")
    parser.add_argument("--n",          type=int,   default=10,   help="Number of problems (default: 10)")
    parser.add_argument("--checkpoint", type=str,   default=None, help="Checkpoint path (default: auto-detect latest)")
    parser.add_argument("--cpu",        action="store_true",      help="Force CPU (for when GPU is busy)")
    parser.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens",
                        help="Max new tokens to generate (default: 256)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
