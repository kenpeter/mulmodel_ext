"""
Evaluate pre-trained code model on LeetCode test set.
Checks: code compilation + test case execution.
"""

import os
import sys
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ──────────────────────────────────────────────────────────────
OUT_DIR = "./eval_results"
TEST_DATA = "/home/kenpeter/work/data/newfacade_LeetCodeDataset/leetcode_test.jsonl"
MAX_NEW_TOKENS = 1024
MAX_PROBLEMS = 30
TIMEOUT_SECONDS = 10
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


# ── Load model ──────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── Generate ────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, prompt_text):
    """Generate code using official Qwen chat template."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant. Write ONLY valid Python code for the following problem. No explanations, no markdown.",
        },
        {"role": "user", "content": f"Solve this LeetCode problem:\n\n{prompt_text}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
        )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
    except RuntimeError as e:
        print(f"[RuntimeError] {e}")
        return ""
    return generated


# ── Extract code ────────────────────────────────────────────────────────
def extract_code(text):
    # Generated Python (no markdown blocks)
    # Look for class Solution first (most reliable indicator)
    if "class Solution" in text:
        idx = text.index("class Solution")
        end = len(text)
        for marker in ["###", "##", "\n\n\n"]:
            pos = text.find(marker, idx + 20)
            if pos != -1:
                end = min(end, pos)
        return text[idx:end].strip()

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

    return text.strip()


def normalize_indentation(code):
    """Replace inconsistent indentation with consistent tabs"""
    lines = code.split("\n")
    new_lines = []
    for line in lines:
        # Replace leading spaces with tabs (4, 3, or 2 spaces)
        if line.startswith("    "):
            new_lines.append("\t" + line[4:])
        elif line.startswith("   "):
            new_lines.append("\t" + line[3:])
        elif line.startswith("  "):
            new_lines.append("\t" + line[2:])
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def truncate_redundant_code(code):
    """Remove redundant code after main function (LlmFix Step 2)"""
    code = code.split('if __name__ == "__main__":')[0]
    code = code.split("if __name__ == '__main__':")[0]
    return code


def add_missing_imports(code, error_message):
    """Add missing imports based on error message (LlmFix Step 3)"""
    if not error_message:
        return code

    # Common modules that models forget to import
    import_map = {
        "np": "import numpy as np",
        "pd": "import pandas as pd",
        "List": "from typing import List",
        "Optional": "from typing import Optional",
        "Dict": "from typing import Dict",
        "Tuple": "from typing import Tuple",
        "Set": "from typing import Set",
        "deque": "from collections import deque",
        "defaultdict": "from collections import defaultdict",
        "Counter": "from collections import Counter",
        "heapq": "import heapq",
        "bisect": "import bisect",
        "math": "import math",
        "re": "import re",
    }

    imports_to_add = []
    for name, import_stmt in import_map.items():
        if f"NameError: name '{name}' is not defined" in error_message:
            if import_stmt not in code:
                imports_to_add.append(import_stmt)

    if imports_to_add:
        return "\n".join(imports_to_add) + "\n\n" + code
    return code


def extract_method_name(problem_description):
    """Extract expected method name from problem description (RADAR approach)"""
    import re

    patterns = [
        r"def (\w+)\(",  # def method_name(
        r"Implement (\w+)",  # Implement method_name
        r"method (\w+)",  # method method_name
        r"function (\w+)",  # function method_name
        r"class (\w+)]\s*\(?\s*$",  # class name at end of line
        r"(\w+)\s*\(\s*self",  # method name before self
    ]
    for pattern in patterns:
        match = re.search(pattern, problem_description, re.MULTILINE)
        if match:
            return match.group(1)
    return None


def replace_method_name(code, expected_method):
    """Replace generated method name with expected method name from problem"""
    if not expected_method:
        return code

    # Find the class Solution definition
    class_match = re.search(r"class\s+Solution\s*[:(]", code)
    if not class_match:
        return code

    # Find the method definition inside the class (after class Solution)
    class_end = class_match.end()
    method_match = re.search(r"def\s+(\w+)\s*\(\s*self", code[class_end:])
    if not method_match:
        return code

    generated_method = method_match.group(1)

    # Replace the method name (only the first occurrence inside the class)
    if generated_method != expected_method:
        # Find the first def self.method_name pattern after class Solution
        pattern = rf"(class\s+Solution\s*[:(].*?def\s+){generated_method}(\s*\()"
        replacement = rf"\g<1>{expected_method}\2"
        code = re.sub(pattern, replacement, code, count=1, flags=re.DOTALL)

    return code


def add_common_imports(code):
    """Proactively add common imports that LeetCode solutions typically need"""
    imports = []
    if "List[" in code or "list[" in code or ": List" in code:
        imports.append("from typing import List")
    if "deque" in code and "from collections import deque" not in code:
        imports.append("from collections import deque")
    if "defaultdict" in code and "from collections import defaultdict" not in code:
        imports.append("from collections import defaultdict")
    if "Counter" in code and "from collections import Counter" not in code:
        imports.append("from collections import Counter")
    if "heapq" in code and "import heapq" not in code:
        imports.append("import heapq")
    if imports:
        return "\n".join(imports) + "\n\n" + code
    return code


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
    global MAX_PROBLEMS
    if "--full" in sys.argv:
        MAX_PROBLEMS = 228

    model, tokenizer = load_model()

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
        prompt = f"### Problem:\n{desc}\n\n### Solution:\n"

        print(
            f"[{i + 1}/{len(problems)}] {task_id} ({prob.get('difficulty', '?')})",
            end=" ... ",
        )

        generated = generate(model, tokenizer, prompt)
        code = extract_code(generated)
        code = normalize_indentation(code)
        code = truncate_redundant_code(code)
        code = add_common_imports(code)

        # RADAR: Extract and fix method name from problem description or entry_point
        entry_point = prob.get("entry_point", "")
        if "()." in entry_point:
            expected_method = entry_point.split("().")[1]
        else:
            expected_method = extract_method_name(desc)
        if expected_method:
            code = replace_method_name(code, expected_method)

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
