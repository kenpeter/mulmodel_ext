#!/usr/bin/env python3
import sys
import time
import os
import torch

sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer

teacher_path = os.path.expanduser(
    "~/.cache/huggingface/hub/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled"
)

print("Loading teacher...")
t0 = time.time()
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_path,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
print(f"Teacher loaded in {time.time() - t0:.2f}s")
print("Loading tokenizer...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    teacher_path, trust_remote_code=True, local_files_only=True
)
print(f"Tokenizer loaded in {time.time() - t0:.2f}s")
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print("Done")
