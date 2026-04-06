#!/usr/bin/env python3
"""Quick test of training script to see where it hangs."""

import sys
import os

sys.path.insert(0, ".")
sys.argv = ["train_kda_muon.py"]  # simulate command line

# Monkey-patch time.sleep to detect hangs
import time

original_sleep = time.sleep


def debug_sleep(seconds):
    import traceback

    print(f"!!! SLEEP CALLED for {seconds}s from:")
    traceback.print_stack()
    original_sleep(seconds)


time.sleep = debug_sleep

# Reduce max_steps
import scripts.train_kda_muon as train_module

train_module.max_steps = 5
train_module.save_every = 100  # no saves
train_module.log_every = 1
train_module.grad_accum = 1

# Run main
print("Starting training script...")
train_module.main()
