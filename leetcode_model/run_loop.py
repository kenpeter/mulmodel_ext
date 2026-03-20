"""
Autoresearch loop: train 5 min → eval on real LeetCode → record → repeat.

Usage:
    python run_loop.py              # run forever
    python run_loop.py --once       # one cycle (train + eval)
    python run_loop.py --eval-only  # skip training, just eval
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from engram_memory import (
    store_result,
    recall_relevant,
    get_patterns,
    consolidate,
    stats,
)

PROJECT = Path(__file__).parent
NANOGPT = PROJECT / "nanoGPT"
TRAIN_CONFIG = NANOGPT / "config" / "train_leetcode.py"
EVAL_SCRIPT = PROJECT / "evaluate.py"
STATE_FILE = PROJECT / "research-state.md"
FINDINGS_FILE = PROJECT / "findings.md"

# ── Helpers ─────────────────────────────────────────────────────────────


def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    with open(PROJECT / "autoresearch.log", "a") as f:
        f.write(line + "\n")


def run(cmd, cwd=None):
    """Run a command, stream output, return exit code."""
    log(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd or str(PROJECT))
    return proc.returncode


def has_checkpoint():
    return (NANOGPT / "out-leetcode" / "ckpt.pt").exists()


def update_state_file(cycle, metric, compile_count, total, notes=""):
    """Update research-state.md with latest results."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    delta = f"+{metric:.1f}" if metric > 0 else "0"

    # Read current file
    text = STATE_FILE.read_text()

    # Update total runs
    text = text.replace("Total runs**: 0", f"Total runs**: {cycle}")

    # Update best value
    text = text.replace("Best value**: null", f"Best value**: {metric:.1f}%")

    # Add trajectory row
    row = f"| {cycle} | H1 | {metric:.1f}% | {delta} | 5 | compile={compile_count}/{total}, pass={metric:.1f}% |"
    if "|---|" in text:
        text = text.replace("|---|", f"| {row}\n|---|")

    STATE_FILE.write_text(text)
    log(f"Updated research-state.md (cycle {cycle}, pass={metric:.1f}%)")


# ── Main loop ───────────────────────────────────────────────────────────


def train():
    """Train for one 5-minute cycle."""
    log("=" * 50)
    log("STEP 1: TRAIN")

    if has_checkpoint():
        log("Checkpoint found — resuming training")
        # Patch config to resume
        config_text = TRAIN_CONFIG.read_text()
        if "init_from = 'scratch'" in config_text:
            config_text = config_text.replace(
                "init_from = 'scratch'", "init_from = 'resume'"
            )
            TRAIN_CONFIG.write_text(config_text)
            log("Switched config to init_from = 'resume'")

    return run(
        [sys.executable, str(NANOGPT / "train.py"), f"config/train_leetcode.py"],
        cwd=str(NANOGPT),
    )


def evaluate():
    """Run eval on all 228 test problems."""
    log("=" * 50)
    log("STEP 2: EVALUATE on real LeetCode problems")
    return run([sys.executable, str(EVAL_SCRIPT)])


def get_eval_results():
    """Read eval results from last run."""
    results_path = NANOGPT / "out-leetcode" / "eval_results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def one_cycle(cycle_num):
    """Run one full cycle: recall → train → eval → store."""
    log(f"\n{'#' * 50}")
    log(f"CYCLE {cycle_num}")
    log(f"{'#' * 50}\n")

    # RECALL — what do we know from past experiments?
    past = recall_relevant("what training changes worked or failed?", limit=3)
    if past:
        log("RECALL from Engram:")
        for r in past:
            log(f"  - {r.get('content', r)}")

    # Train
    train_rc = train()
    if train_rc != 0:
        log(f"WARNING: Training exited with code {train_rc}")

    # Eval
    eval_rc = evaluate()
    if eval_rc != 0:
        log(f"WARNING: Eval exited with code {eval_rc}")

    # Read results
    results = get_eval_results()
    if results:
        total = results.get("total", 0)
        passes = results.get("passes_tests", 0)
        compiles = results.get("compiles", 0)
        pass_rate = (passes / total * 100) if total > 0 else 0
        log(
            f"RESULTS: {passes}/{total} pass ({pass_rate:.1f}%), {compiles}/{total} compile"
        )

        # STORE result in Engram
        store_result(cycle_num, pass_rate, compiles, total)

        # Update state
        update_state_file(cycle_num, pass_rate, compiles, total)
    else:
        log("No eval results found")
        pass_rate = 0

    # CONSOLIDATE every 10 cycles
    if cycle_num % 10 == 0:
        consolidate()
        log(f"Engram consolidated. Stats: {stats()}")

    return pass_rate


def main():
    args = sys.argv[1:]

    if "--eval-only" in args:
        evaluate()
        results = get_eval_results()
        if results:
            print(f"\n{results['passes_tests']}/{results['total']} pass")
        return

    if "--once" in args:
        one_cycle(1)
        return

    log("Starting autoresearch loop")
    log(f"Each cycle: train ~5min → eval 228 problems → record → repeat")
    log(f"Press Ctrl+C to stop\n")

    cycle = 0
    stable_count = 0
    last_pass_rate = -1

    while True:
        cycle += 1
        pass_rate = one_cycle(cycle)

        # Log status
        if pass_rate > 0:
            log(f"Model is learning! {pass_rate:.1f}% pass rate")
        if stable_count >= 10 and pass_rate == 0:
            log(
                "0% pass for 10+ cycles — consider changing model size, lr, or data format"
            )

        log(f"Cycle {cycle} done. Looping forever...\n")
        time.sleep(5)


if __name__ == "__main__":
    main()
