#!/usr/bin/env python3
"""Test loading 4B teacher with 4-bit quantization."""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

teacher_path = os.path.expanduser(
    "~/.cache/huggingface/hub/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled"
)

print("Loading teacher with 4-bit quantization...")
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"Teacher loaded successfully. Device: {teacher.device}")
    print(f"Parameter count: {sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B")

    # Quick memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory allocated: {allocated:.2f}GB")
        print(f"GPU memory reserved: {reserved:.2f}GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_path, trust_remote_code=True, local_files_only=True
    )
    print("Tokenizer loaded.")

    # Test forward pass with dummy input
    print("Testing forward pass...")
    input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(
        teacher.device
    )
    with torch.no_grad():
        outputs = teacher(input_ids)
    print(f"Forward pass succeeded. Logits shape: {outputs.logits.shape}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("All tests passed.")
