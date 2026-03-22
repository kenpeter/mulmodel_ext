"""
Run Loop — auto research loop.

Loop: TRAIN → EVAL → if stale → spawn Reviewer + Code Updater → REPEAT
If improving → just keep training.

Agents:
  - Reviewer: searches arxiv + github, finds solutions
  - Code Updater: implements the fix

Usage:
    python run_loop.py              # run forever
    python run_loop.py --once       # one cycle
    python run_loop.py --eval-only  # just eval
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).parent
NANOGPT = PROJECT / "nanoGPT"
CONDA = "mulmodel"

RESULTS_FILE = NANOGPT / "out-leetcode" / "eval_results.json"
REVIEWER_PROMPT = PROJECT / "agents" / "reviewer.md"
UPDATER_PROMPT = PROJECT / "agents" / "code_updater.md"

prev_compile_rate = 0


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(PROJECT / "autoresearch.log", "a") as f:
        f.write(line + "\n")


def run_train():
    log("TRAIN")
    cmd = [
        "conda",
        "run",
        "-n",
        CONDA,
        "python",
        str(NANOGPT / "train.py"),
        "config/train_leetcode.py",
    ]
    return subprocess.run(cmd, cwd=str(NANOGPT)).returncode


def run_eval():
    log("EVAL")
    cmd = ["conda", "run", "-n", CONDA, "python", str(PROJECT / "evaluate.py")]
    return subprocess.run(cmd, cwd=str(NANOGPT)).returncode


def get_results():
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_stale(results):
    global prev_compile_rate
    if results is None:
        return True
    total = results.get("total", 1)
    compiles = results.get("compiles", 0)
    compile_rate = (compiles / total) * 100
    stale = compile_rate <= prev_compile_rate + 2  # allow 2% fluctuation
    prev_compile_rate = compile_rate
    return stale


def spawn_agent(prompt_file, task_desc):
    """Spawn an opencode instance with opencode run."""
    if not prompt_file.exists():
        log(f"WARNING: {prompt_file} not found")
        return

    log(f"SPAWN: {prompt_file.name} — {task_desc}")
    prompt = prompt_file.read_text()
    subprocess.run(["opencode", "run", prompt], timeout=600)


def run_reviewer():
    """Spawn Reviewer to search arxiv/github."""
    results = get_results()
    if results is None:
        return

    total = results.get("total", 1)
    compiles = results.get("compiles", 0)
    compile_rate = (compiles / total) * 100

    task = f"Eval results: {compiles}/{total} compile ({compile_rate:.0f}%). Search arxiv and github for solutions to improve compile rate. Write findings to research-log.md with URLs."
    spawn_agent(REVIEWER_PROMPT, task)


def run_updater():
    """Spawn Code Updater to implement fix."""
    task = "Read research-log.md for the latest research findings. Implement ONE code fix based on the findings. Write what you changed to research-log.md."
    spawn_agent(UPDATER_PROMPT, task)


def one_cycle(cycle):
    global prev_compile_rate
    log(f"\n{'=' * 50}")
    log(f"CYCLE {cycle}")
    log(f"{'=' * 50}\n")

    # TRAIN
    run_train()

    # EVAL
    run_eval()

    # CHECK
    results = get_results()
    total = results.get("total", 1) if results else 0
    compiles = results.get("compiles", 0) if results else 0
    compile_rate = (compiles / max(total, 1)) * 100

    log(f"RESULT: {compiles}/{total} compile ({compile_rate:.0f}%)")

    # DECIDE
    if compile_rate > prev_compile_rate + 2:
        log("IMPROVING — keep training, no agents")
    else:
        log("STALE — spawning agents")
        # Sequential: reviewer must finish before updater reads findings
        spawn_agent(
            REVIEWER_PROMPT, "Search arxiv/github, write findings to research-log.md"
        )
        spawn_agent(UPDATER_PROMPT, "Read research-log.md, implement ONE code fix")

    prev_compile_rate = compile_rate

    # LOG
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PROJECT / "research-log.md", "a") as f:
        f.write(f"\n## {ts} — Cycle {cycle}\n")
        f.write(f"- Compile: {compiles}/{total} ({compile_rate:.0f}%)\n")
        f.write(f"- Pass: {results.get('passes_tests', 0) if results else 0}/{total}\n")

    with open(PROJECT / "findings.md", "a") as f:
        f.write(f"\n### {ts} — Cycle {cycle}\n")
        f.write(f"- Compile: {compiles}/{total} ({compile_rate:.0f}%)\n")


def run():
    log("Starting auto research loop")
    log("Loop: TRAIN → EVAL → if stale → Reviewer + Code Updater → REPEAT")
    log(f"Reviewer: {REVIEWER_PROMPT}")
    log(f"Updater: {UPDATER_PROMPT}\n")

    cycle = 0
    while True:
        cycle += 1
        try:
            one_cycle(cycle)
        except Exception as e:
            log(f"ERROR cycle {cycle}: {e}")
            time.sleep(10)
            continue
        time.sleep(5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--test-agents", action="store_true", help="Force stale to test agent spawning"
    )
    args = parser.parse_args()

    if args.eval_only:
        run_eval()
        results = get_results()
        if results:
            print(
                f"\n{results.get('passes_tests', 0)}/{results.get('total', 0)} pass, {results.get('compiles', 0)}/{results.get('total', 0)} compile"
            )
    elif args.test_agents:
        # Force stale to verify agents spawn
        prev_compile_rate = 999
        log("TEST MODE: forcing stale to verify agents spawn")
        run_reviewer()
        run_updater()
    elif args.once:
        one_cycle(1)
    else:
        run()
