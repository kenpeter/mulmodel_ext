#!/usr/bin/env python3
import sys
import time

sys.path.insert(0, ".")
from datasets import load_dataset

print("Loading dataset...")
t0 = time.time()
ds = load_dataset("justindal/leetcode-python-dataset", split="train")
print(f"Loaded in {time.time() - t0:.2f}s, {len(ds)} samples")
print("First item keys:", ds[0].keys())
print("Iterating...")
t0 = time.time()
count = 0
for item in ds:
    msgs = item["messages"]
    if len(msgs) >= 3:
        prompt = msgs[0]["content"] + "\n\n" + msgs[1]["content"]
        completion = msgs[2]["content"]
        count += 1
    if count >= 10:
        break
print(f"Iterated {count} items in {time.time() - t0:.2f}s")
print("Done")
