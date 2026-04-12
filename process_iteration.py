#!/usr/bin/env python3
"""Process iteration results and update log."""
import subprocess
import json
import os
from datetime import datetime

def get_accuracy():
    """Run eval and extract accuracy."""
    try:
        result = subprocess.run(
            "bash get_accuracy.sh",
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        accuracy_str = result.stdout.strip()
        if accuracy_str:
            return float(accuracy_str)
    except Exception as e:
        print(f"Error getting accuracy: {e}")
    return None

def get_latest_checkpoint():
    """Find the latest checkpoint."""
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        return None

    pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not pt_files:
        return None

    # Sort by step number
    def get_step(f):
        if f.startswith('step_'):
            return int(f[5:-3])
        return -1

    pt_files.sort(key=get_step, reverse=True)
    return pt_files[0]

def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            "git rev-parse --short HEAD",
            shell=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except:
        return "unknown"

def log_iteration(iteration, change, accuracy, status, notes=""):
    """Append to results log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit = get_git_commit()

    line = f"{iteration}\t{timestamp}\t{change}\t{accuracy}\t{status}\t{notes}\t{commit}\n"

    with open("AUTORESEARCH_RESULTS.tsv", "a") as f:
        f.write(line)

    print(f"[Iter {iteration}] {change}: {accuracy}% | {status}")
    return float(accuracy) if accuracy and accuracy != '-' else None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python process_iteration.py <iteration> <change> <expected_change>")
        sys.exit(1)

    iteration = sys.argv[1]
    change = sys.argv[2]

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}: {change}")
    print(f"{'='*60}")

    ckpt = get_latest_checkpoint()
    if not ckpt:
        print("ERROR: No checkpoint found!")
        log_iteration(iteration, change, "-", "FAILED", "No checkpoint")
        sys.exit(1)

    print(f"Checkpoint: {ckpt}")

    accuracy = get_accuracy()
    if accuracy is None:
        print("ERROR: Could not get accuracy!")
        log_iteration(iteration, change, "-", "FAILED", "Eval error")
        sys.exit(1)

    print(f"Accuracy: {accuracy:.1f}%")
    log_iteration(iteration, change, f"{accuracy:.1f}", "COMPLETED", ckpt)
    print(f"{'='*60}\n")
