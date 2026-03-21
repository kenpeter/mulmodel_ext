"""
Clean and unify all LeetCode datasets into a consistent format.

Format with CoT:
    ### Problem:
    {problem}

    ### Explanation:
    {reasoning/cot}

    ### Solution:
    ```python
    {code}
    ```

    <|end_of_text|>

Format without CoT:
    ### Problem:
    {problem}

    ### Solution:
    ```python
    {code}
    ```

    <|end_of_text|>
"""

import os
import json
import re
import pandas as pd
import numpy as np
import pickle
import tiktoken

OUTPUT_DIR = "/home/kenpeter/work/mulmodel_ext/leetcode_model/nanoGPT/data/leetcode"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_examples = []


def extract_python_code(text):
    """Extract python code from markdown code blocks or raw code."""
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "def " in code or "class " in code:
            return code
    if "def " in text or "class " in text:
        return text.strip()
    return None


def clean_text(text):
    """Clean text."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if len(text) < 20:
        return None
    return text


def extract_explanation_from_content(content):
    """Extract explanation (non-code part) from markdown content."""
    if not content or not isinstance(content, str):
        return None
    # Remove code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", content)
    cleaned = cleaned.strip()
    if len(cleaned) < 30:
        return None
    return cleaned


# ── Source 1: greengerong (code only) ────────────────────────────────
print("Loading greengerong...")
with open("/home/kenpeter/work/data/greengerong_LeetCode/leetcode-train.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        problem = clean_text(rec.get("content", ""))
        code = extract_python_code(rec.get("python", ""))
        if problem and code:
            all_examples.append(
                {
                    "problem": problem,
                    "explanation": None,
                    "solution": code,
                    "source": "greengerong",
                }
            )
print(
    f"  greengerong: {len([e for e in all_examples if e['source'] == 'greengerong'])}"
)


# ── Source 2: LimYeri (has explanation after code) ───────────────────
print("Loading LimYeri...")
df = pd.read_parquet("/home/kenpeter/work/data/LimYeri_LeetCode/train.parquet")
count = 0
for _, row in df.iterrows():
    convs = row.get("conversations", [])
    title = row.get("title", "")
    if convs is None or not hasattr(convs, "__iter__"):
        continue
    for conv in convs:
        if isinstance(conv, dict) and conv.get("from") == "assistant":
            value = conv.get("value", "")
            code = extract_python_code(value)
            if code:
                # Extract explanation (text after code block)
                explanation = extract_explanation_from_content(value)
                problem = f"LeetCode: {title}" if title else "LeetCode Problem"
                all_examples.append(
                    {
                        "problem": problem,
                        "explanation": explanation,
                        "solution": code,
                        "source": "LimYeri",
                    }
                )
                count += 1
                break
print(f"  LimYeri: {count}")


# ── Source 3: juyoungml (code only) ──────────────────────────────────
print("Loading juyoungml...")
df = pd.read_parquet(
    "/home/kenpeter/work/data/juyoungml_LeetCodeRosetta/data/train-00000-of-00001.parquet"
)
count = 0
for _, row in df.iterrows():
    problem = clean_text(row.get("content", ""))
    code = extract_python_code(row.get("python_code", ""))
    if problem and code:
        all_examples.append(
            {
                "problem": problem,
                "explanation": None,
                "solution": code,
                "source": "juyoungml",
            }
        )
        count += 1
print(f"  juyoungml: {count}")


# ── Source 4: mesolitica (has QwQ reasoning) ─────────────────────────
print("Loading mesolitica...")
df = pd.read_parquet("/home/kenpeter/work/data/mesolitica_LeetCodeQwQ/train.parquet")
count = 0
for _, row in df.iterrows():
    problem = clean_text(row.get("content", ""))
    solution = row.get("solution", "")
    code = extract_python_code(solution) or extract_python_code(str(solution))
    qwq = row.get("qwq", "")
    explanation = (
        clean_text(qwq) if qwq and isinstance(qwq, str) and len(str(qwq)) > 50 else None
    )
    if problem and code:
        all_examples.append(
            {
                "problem": problem,
                "explanation": explanation,
                "solution": code,
                "source": "mesolitica",
            }
        )
        count += 1
print(f"  mesolitica: {count}")


# ── Source 5: DenCT (has explanation in content) ─────────────────────
print("Loading DenCT...")
df = pd.read_parquet(
    "/home/kenpeter/work/data/DenCT_LeetCode/leetcode-java-python.parquet"
)
count = 0
for _, row in df.iterrows():
    problem = clean_text(row.get("question_content", "") or row.get("content", ""))
    content = row.get("content", "")
    code = extract_python_code(content)
    explanation = extract_explanation_from_content(content)
    if problem and code:
        all_examples.append(
            {
                "problem": problem,
                "explanation": explanation,
                "solution": code,
                "source": "DenCT",
            }
        )
        count += 1
print(f"  DenCT: {count}")


# ── Source 6: vovw (code only) ───────────────────────────────────────
print("Loading vovw...")
df = pd.read_parquet("/home/kenpeter/work/data/vovw_LeetCode/dataset.parquet")
count = 0
for _, row in df.iterrows():
    problem = clean_text(row.get("content", ""))
    code = extract_python_code(row.get("python", ""))
    if problem and code:
        all_examples.append(
            {
                "problem": problem,
                "explanation": None,
                "solution": code,
                "source": "vovw",
            }
        )
        count += 1
print(f"  vovw: {count}")


# ── Source 7: newfacade (code only) ──────────────────────────────────
print("Loading newfacade...")
for path in [
    "/home/kenpeter/work/data/newfacade_LeetCodeDataset/leetcode_train.jsonl",
    "/home/kenpeter/work/data/newfacade_LeetCodeDataset/leetcode_test.jsonl",
]:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            problem = clean_text(rec.get("problem_description", ""))
            code = rec.get("completion", "").strip()
            if problem and code:
                all_examples.append(
                    {
                        "problem": problem,
                        "explanation": None,
                        "solution": code,
                        "source": "newfacade",
                    }
                )
print(f"  newfacade: {len([e for e in all_examples if e['source'] == 'newfacade'])}")


# ── Source 8: YT_CoT (has Summary) ───────────────────────────────────
print("Loading YT_CoT...")
for parquet_file in [
    "/home/kenpeter/work/data/LeetCode_YT_CC_CoT_Summary/data/train-00000-of-00002.parquet",
    "/home/kenpeter/work/data/LeetCode_YT_CC_CoT_Summary/data/train-00001-of-00002.parquet",
]:
    if not os.path.exists(parquet_file):
        continue
    df = pd.read_parquet(parquet_file)
    for _, row in df.iterrows():
        title = row.get("title", "")
        problem = clean_text(row.get("question_content", ""))
        code = extract_python_code(row.get("python", ""))
        summary = row.get("Summary", "")
        explanation = (
            clean_text(summary)
            if summary and isinstance(summary, str) and len(summary) > 50
            else None
        )
        if problem and code:
            full_problem = f"{title}\n{problem}" if title else problem
            all_examples.append(
                {
                    "problem": full_problem,
                    "explanation": explanation,
                    "solution": code,
                    "source": "YT_CoT",
                }
            )
print(f"  YT_CoT: {len([e for e in all_examples if e['source'] == 'YT_CoT'])}")


# ── Deduplicate ──────────────────────────────────────────────────────
print(f"\nTotal before dedup: {len(all_examples)}")
seen = set()
unique_examples = []
for ex in all_examples:
    key = (ex["problem"][:100], ex["solution"][:100])
    if key not in seen:
        seen.add(key)
        unique_examples.append(ex)
all_examples = unique_examples
print(f"Total after dedup: {len(all_examples)}")

from collections import Counter

source_counts = Counter(e["source"] for e in all_examples)
cot_count = sum(1 for e in all_examples if e.get("explanation"))
print(
    f"\nWith CoT: {cot_count}/{len(all_examples)} ({cot_count / len(all_examples) * 100:.0f}%)"
)
print(f"\nBy source:")
for src, count in source_counts.most_common():
    has_cot = sum(
        1 for e in all_examples if e["source"] == src and e.get("explanation")
    )
    print(f"  {src}: {count} ({has_cot} with CoT)")


# ── Convert to training text ─────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
training_texts = []
for ex in all_examples:
    if ex.get("explanation"):
        text = f"### Problem:\n{ex['problem']}\n\n### Explanation:\n{ex['explanation']}\n\n### Solution:\n```python\n{ex['solution']}\n```\n\n<|endoftext|>"
    else:
        text = f"### Problem:\n{ex['problem']}\n\n### Solution:\n```python\n{ex['solution']}\n```\n\n<|endoftext|>"
    training_texts.append(text)

full_text = "".join(training_texts)
tokens = np.array(
    enc.encode(full_text, allowed_special={"<|endoftext|>"}), dtype=np.uint16
)
print(f"\nTotal tokens: {len(tokens):,} ({len(tokens) / 1e6:.2f}M)")


# ── Train/val split ──────────────────────────────────────────────────
n = len(tokens)
split = int(n * 0.9)
train_ids = tokens[:split]
val_ids = tokens[split:]

# ── Save ─────────────────────────────────────────────────────────────
train_ids.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
val_ids.tofile(os.path.join(OUTPUT_DIR, "val.bin"))

meta = {
    "vocab_size": enc.n_vocab,
    "encoder": "gpt2",
    "eot_token": enc._special_tokens["<|endoftext|>"],
}
with open(os.path.join(OUTPUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"\nSaved to {OUTPUT_DIR}/")
print(
    f"  train.bin: {os.path.getsize(os.path.join(OUTPUT_DIR, 'train.bin')) / 1e6:.1f} MB"
)
print(
    f"  val.bin:   {os.path.getsize(os.path.join(OUTPUT_DIR, 'val.bin')) / 1e6:.1f} MB"
)
