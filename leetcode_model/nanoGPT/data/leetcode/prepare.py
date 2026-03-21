"""
Prepare LeetCode CoT data for nanoGPT training.
Combines four sources:
  1. newfacade_LeetCodeDataset (JSONL)
  2. LeetCode_YT_CC_CoT_Summary (parquet)
  3. greengerong_LeetCode (JSONL) - content + python
  4. LimYeri_LeetCode (parquet) - conversations with python code

Output format per problem:
    ### Problem:
    {problem_description}

    ### Solution:
    ```python
    {code}
    ```

"""

import os
import json
import pickle
import glob
import re
import numpy as np
import pandas as pd
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────
TRAIN_JSONL = "/home/kenpeter/work/data/newfacade_LeetCodeDataset/leetcode_train.jsonl"
TEST_JSONL = "/home/kenpeter/work/data/newfacade_LeetCodeDataset/leetcode_test.jsonl"
PARQUET_DIR = "/home/kenpeter/work/data/LeetCode_YT_CC_CoT_Summary/data/"
GREENGERONG_JSONL = "/home/kenpeter/work/data/greengerong_LeetCode/leetcode-train.jsonl"
LIMYERI_PARQUET = "/home/kenpeter/work/data/LimYeri_LeetCode/train.parquet"
OUTPUT_DIR = os.path.dirname(__file__)
VAL_RATIO = 0.1
LOG_PATH = "/home/kenpeter/work/mulmodel_ext/leetcode_model/research-log.md"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load and format ─────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def format_problem_jsonl(rec):
    """Format a single JSONL problem as training text."""
    desc = rec.get("problem_description", "").strip()
    code = rec.get("completion", "").strip()
    if not desc or not code:
        return None
    return f"### Problem:\n{desc}\n\n### Solution:\n```python\n{code}\n```\n\n<|endoftext|>"


def format_problem_parquet(row):
    """Format a single parquet row as training text."""
    title = str(row.get("title", "")).strip()
    content = str(row.get("question_content", "")).strip()
    code = str(row.get("python", "")).strip()
    if not content or not code or code == "nan":
        return None
    return f"### Problem:\n{title}\n{content}\n\n### Solution:\n```python\n{code}\n```\n\n<|endoftext|>"


def format_problem_greengerong(rec):
    """Format greengerong JSONL record (content + python)."""
    content = rec.get("content", "").strip()
    raw_python = rec.get("python", "").strip()
    if not content or not raw_python:
        return None
    # Strip ```python ... ``` wrapper if present
    code = re.sub(r"^```python\s*\n?", "", raw_python)
    code = re.sub(r"\n?```\s*$", "", code)
    code = code.strip()
    if not code:
        return None
    return f"### Problem:\n{content}\n\n### Solution:\n```python\n{code}\n```\n\n<|endoftext|>"


def format_problem_limyeri(row):
    """Format LimYeri parquet row (conversations with python code)."""
    title = str(row.get("title", "")).strip()
    convs = row.get("conversations", [])
    # Extract user question and assistant code
    user_msg = ""
    assistant_msg = ""
    for msg in convs:
        if msg.get("from") == "user":
            user_msg = msg.get("value", "").strip()
        elif msg.get("from") == "assistant":
            assistant_msg = msg.get("value", "").strip()
    if not user_msg or not assistant_msg:
        return None
    # Extract python code from ```python ... ``` blocks
    code_match = re.search(r"```python\s*\n(.*?)```", assistant_msg, re.DOTALL)
    if not code_match:
        return None
    code = code_match.group(1).strip()
    if not code:
        return None
    problem = f"{title}\n{user_msg}"
    return f"### Problem:\n{problem}\n\n### Solution:\n```python\n{code}\n```\n\n<|endoftext|>"


print("Loading and formatting data...")
all_text = []

# Source 1: JSONL from newfacade_LeetCodeDataset
jsonl_count = 0
for jsonl_path in [TRAIN_JSONL, TEST_JSONL]:
    if not os.path.exists(jsonl_path):
        print(f"  Skipping {jsonl_path} (not found)")
        continue
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            text = format_problem_jsonl(rec)
            if text:
                all_text.append(text)
                jsonl_count += 1
print(f"  JSONL source: {jsonl_count} examples")

# Source 2: Parquet from LeetCode_YT_CC_CoT_Summary
parquet_count = 0
parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
print(f"  Found {len(parquet_files)} parquet files in {PARQUET_DIR}")
for pf in parquet_files:
    df = pd.read_parquet(pf)
    for _, row in df.iterrows():
        text = format_problem_parquet(row)
        if text:
            all_text.append(text)
            parquet_count += 1
print(f"  Parquet source: {parquet_count} examples")

# Source 3: JSONL from greengerong_LeetCode
greengerong_count = 0
if os.path.exists(GREENGERONG_JSONL):
    with open(GREENGERONG_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            text = format_problem_greengerong(rec)
            if text:
                all_text.append(text)
                greengerong_count += 1
print(f"  Greengerong source: {greengerong_count} examples")

# Source 4: Parquet from LimYeri_LeetCode
limyeri_count = 0
if os.path.exists(LIMYERI_PARQUET):
    df_limyeri = pd.read_parquet(LIMYERI_PARQUET)
    for _, row in df_limyeri.iterrows():
        text = format_problem_limyeri(row)
        if text:
            all_text.append(text)
            limyeri_count += 1
print(f"  LimYeri source: {limyeri_count} examples")

print(f"Total: {len(all_text)} examples")

# Join documents (EOT is already at end of each document)
full_text = "".join(all_text)

# ── Tokenize ────────────────────────────────────────────────────────────
print("Tokenizing...")
tokens = enc.encode(full_text, allowed_special={"<|endoftext|>"})
tokens = np.array(tokens, dtype=np.uint16)
print(f"Total tokens: {len(tokens):,} ({len(tokens) / 1e6:.2f}M)")

# ── Train/val split ─────────────────────────────────────────────────────
n = len(tokens)
split = int(n * (1 - VAL_RATIO))
train_ids = tokens[:split]
val_ids = tokens[split:]

print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")

# ── Save ────────────────────────────────────────────────────────────────
train_ids.tofile(os.path.join(OUTPUT_DIR, "train.bin"))
val_ids.tofile(os.path.join(OUTPUT_DIR, "val.bin"))

meta = {
    "vocab_size": enc.n_vocab,
    "encoder": "gpt2",  # tiktoken encoder name
    "eot_token": eot,
}
with open(os.path.join(OUTPUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"\nSaved to {OUTPUT_DIR}/")
print(
    f"  train.bin ({os.path.getsize(os.path.join(OUTPUT_DIR, 'train.bin')) / 1e6:.1f} MB)"
)
print(
    f"  val.bin   ({os.path.getsize(os.path.join(OUTPUT_DIR, 'val.bin')) / 1e6:.1f} MB)"
)
print(f"  meta.pkl")
print(f"\nVocab size: {enc.n_vocab}")

# ── Log to research-log ─────────────────────────────────────────────────
import datetime

log_entry = (
    f"\n### [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}] Data Preparation\n"
    f"- newfacade_LeetCodeDataset (JSONL): {jsonl_count} examples\n"
    f"- LeetCode_YT_CC_CoT_Summary (parquet): {parquet_count} examples\n"
    f"- greengerong_LeetCode (JSONL): {greengerong_count} examples\n"
    f"- LimYeri_LeetCode (parquet): {limyeri_count} examples\n"
    f"- Total examples: {len(all_text)}\n"
    f"- Total tokens: {len(tokens):,} ({len(tokens) / 1e6:.2f}M)\n"
    f"- Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}\n"
)
with open(LOG_PATH, "a") as f:
    f.write(log_entry)
print(f"Logged to {LOG_PATH}")
