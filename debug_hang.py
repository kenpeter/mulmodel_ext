#!/usr/bin/env python3
import sys
import time
import os
import torch

sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.config import StudentConfig
from model.student import StudentModel

teacher_path = os.path.expanduser(
    "~/.cache/huggingface/hub/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled"
)

print("1. Loading teacher...")
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
print(f"   Teacher loaded in {time.time() - t0:.2f}s")

print("2. Loading tokenizer...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    teacher_path, trust_remote_code=True, local_files_only=True
)
print(f"   Tokenizer loaded in {time.time() - t0:.2f}s")

teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

print("3. Building student with KDA...")
t0 = time.time()
config = StudentConfig(
    vocab_size=teacher.model.embed_tokens.weight.shape[0],
    sage_attention=False,
    attn_residual=True,
)
print(f"   Config created: {config}")
student = StudentModel(config)
print(f"   Student created in {time.time() - t0:.2f}s")

print("4. Moving student to CUDA bf16...")
t0 = time.time()
student = student.to(dtype=torch.bfloat16, device="cuda")
print(f"   Student moved in {time.time() - t0:.2f}s")

print("5. Setting gradient checkpointing...")
student.gradient_checkpointing = True
student.train()

print("6. Building optimizer...")
from model.optimizer import MuonClip

optimizer = MuonClip(
    student.parameters(),
    lr=5e-6,
    momentum=0.95,
    weight_decay=0.1,
    qk_clip_threshold=100.0,
    newton_schulz_iters=5,
)
print("   Optimizer created")

print("7. Loading dataset...")
from datasets import load_dataset

t0 = time.time()
ds = load_dataset("justindal/leetcode-python-dataset", split="train")
print(f"   Dataset loaded in {time.time() - t0:.2f}s, {len(ds)} samples")

print("8. Creating SimpleLeetCodeDataset...")
from scripts.train_kda_muon import SimpleLeetCodeDataset

t0 = time.time()
train_ds = SimpleLeetCodeDataset(tokenizer, split="train", max_length=512)
print(
    f"   SimpleLeetCodeDataset created in {time.time() - t0:.2f}s, {len(train_ds)} samples"
)

print("All steps completed successfully.")
