"""
Prepare LeetCode CoT data for nanoGPT training.
Converts JSONL -> tiktoken BPE -> binary format (train.bin, val.bin).

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
import numpy as np
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────
TRAIN_JSONL = "/home/kenpeter/work/data/newfacade_LeetCodeDataset/leetcode_train.jsonl"
TEST_JSONL = "/home/kenpeter/work/data/newfacade_LeetCodeDataset/leetcode_test.jsonl"
OUTPUT_DIR = os.path.dirname(__file__)
VAL_RATIO = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load and format ─────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def format_problem(rec):
    """Format a single problem as training text."""
    desc = rec.get("problem_description", "").strip()
    code = rec.get("completion", "").strip()
    if not desc or not code:
        return None
    return f"### Problem:\n{desc}\n\n### Solution:\n```python\n{code}\n```\n\n"


print("Loading and formatting data...")
all_text = []
problem_count = 0

for jsonl_path in [TRAIN_JSONL, TEST_JSONL]:
    if not os.path.exists(jsonl_path):
        print(f"  Skipping {jsonl_path} (not found)")
        continue
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            text = format_problem(rec)
            if text:
                all_text.append(text)
                problem_count += 1

print(f"Formatted {problem_count} problems")

# Join with EOT token (nanoGPT uses this as document separator)
full_text = enc.decode([eot]).join(all_text)

# ── Tokenize ────────────────────────────────────────────────────────────
print("Tokenizing...")
tokens = enc.encode_ordinary(full_text)
tokens.append(eot)  # Final EOT
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
