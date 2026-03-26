import os
import re
import sys
import subprocess
import tempfile
import signal
import traceback
from typing import Optional
from datasets import load_dataset


def extract_code_from_response(response: str) -> str:
    """Extract Python code from model output."""
    # Try to find code block
    code_block = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    # If no code fences, take everything after first class/function definition
    lines = response.split("\n")
    code_lines = []
    started = False
    for line in lines:
        if not started and (
            line.strip().startswith("class ")
            or line.strip().startswith("def ")
            or line.strip().startswith("from ")
            or line.strip().startswith("import ")
        ):
            started = True
        if started:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)

    return response.strip()


def extract_test_cases(prompt: str) -> list[str]:
    """Extract test assertions from problem description examples."""
    tests = []
    # Look for Example blocks with Input/Output
    example_blocks = re.findall(
        r"(?:Example\s*\d*|示例\s*\d*)[:\s]*\n*(?:Input|输入)[:\s]*(.*?)\n*(?:Output|输出)[:\s]*(.*?)(?=\n*(?:Example|示例|Constraints|提示|$))",
        prompt,
        re.DOTALL | re.IGNORECASE,
    )

    for inp, out in example_blocks:
        inp = inp.strip()
        out = out.strip()
        # Skip if too complex (multi-line)
        if "\n" in inp and not inp.startswith("l1"):
            continue
        tests.append((inp, out))

    return tests


def sandbox_execute(
    code: str,
    timeout: int = 5,
) -> dict:
    """Execute Python code in a sandboxed subprocess."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""},
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Timeout", "returncode": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}
    finally:
        os.unlink(tmp_path)


def run_leetcode_eval(
    model,
    tokenizer,
    device: str = "cuda",
    max_new_tokens: int = 512,
    num_problems: int = 50,
    split: str = "test",
    dataset_name: str = "justindal/leetcode-python-dataset",
) -> dict:
    """Evaluate model on LeetCode problems from the dataset."""
    ds = load_dataset(dataset_name, split=split)
    results = []
    passed = 0
    total = 0

    system_prompt = ds[0]["messages"][0]["content"]

    print(f"\n{'=' * 60}")
    print(
        f"LeetCode Evaluation — {min(num_problems, len(ds))} problems from '{split}' split"
    )
    print(f"{'=' * 60}\n")

    for i, item in enumerate(ds):
        if i >= num_problems:
            break

        messages = item["messages"]
        if len(messages) < 3:
            continue

        user_msg = messages[1]["content"]
        reference_solution = messages[2]["content"]

        # Format prompt for generation
        prompt_text = system_prompt + "\n\n" + user_msg

        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        model.eval()
        with torch.no_grad():
            import torch

            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
            )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        generated_code = extract_code_from_response(response)

        # Check if code is syntactically valid
        try:
            compile(generated_code, "<generated>", "exec")
            syntax_ok = True
        except SyntaxError as e:
            syntax_ok = False

        total += 1
        result = {
            "index": i,
            "syntax_ok": syntax_ok,
            "passed": False,
            "code_preview": generated_code[:200],
        }

        if syntax_ok:
            exec_result = sandbox_execute(generated_code, timeout=5)
            if exec_result["success"]:
                result["passed"] = True
                passed += 1

        results.append(result)
        status = (
            "PASS"
            if result["passed"]
            else ("SYNTAX_ERR" if not syntax_ok else "RUNTIME_ERR")
        )
        print(
            f"  [{i + 1}/{min(num_problems, len(ds))}] {status} | {generated_code[:80].replace(chr(10), ' ')}..."
        )

    accuracy = passed / max(total, 1) * 100
    summary = {
        "total": total,
        "passed": passed,
        "accuracy": accuracy,
        "results": results,
    }

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed ({accuracy:.1f}%)")
    print(f"{'=' * 60}\n")

    return summary
