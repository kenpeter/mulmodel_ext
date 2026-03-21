"""
Single ReAct Agent.

Loop: OBSERVE → REASON → ACT → REPEAT

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


class Agent:
    def __init__(
        self,
        project_dir,
        train_cmd,
        eval_cmd,
        eval_results_path,
        train_cwd=None,
        stop_hour=6,
    ):
        self.project = Path(project_dir)
        self.train_cmd = train_cmd
        self.eval_cmd = eval_cmd
        self.train_cwd = Path(train_cwd) if train_cwd else self.project
        self.eval_results = Path(eval_results_path)
        self.stop_hour = stop_hour
        self.state_file = self.project / "research-state.md"
        self.findings_file = self.project / "findings.md"
        self.log_file = self.project / "research-log.md"
        self.heartbeat = self.project / "heartbeat.txt"
        self.log = self.project / "autoresearch.log"

    def _log(self, msg):
        now = datetime.now().strftime("%H:%M:%S")
        line = f"[{now}] {msg}"
        print(line)
        with open(self.log, "a") as f:
            f.write(line + "\n")

    def _heartbeat(self):
        with open(self.heartbeat, "w") as f:
            f.write(str(time.time()))

    def _run(self, cmd, cwd=None):
        self._log(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=cwd or str(self.train_cwd)).returncode

    def _get_eval_results(self):
        try:
            with open(self.eval_results) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # ── OBSERVE ──────────────────────────────────────────────────────────

    def observe(self, cycle):
        """Read results, categorize errors, update findings."""
        results = self._get_eval_results()
        if results is None:
            self._log("OBSERVE: No eval results yet.")
            return None

        total = results.get("total", 0)
        passes = results.get("passes_tests", 0)
        compiles = results.get("compiles", 0)
        details = results.get("details", [])

        # Categorize errors
        indent_errs = 0
        wrong_method = 0
        for item in details:
            if not item.get("compiles"):
                err = (item.get("compile_error") or "").lower()
                if "indent" in err or "unindent" in err:
                    indent_errs += 1
            else:
                err = (item.get("test_error") or "").lower()
                if "entry point" in err or "not found" in err:
                    wrong_method += 1

        self._log(
            f"OBSERVE: pass={passes}/{total} ({passes / max(total, 1) * 100:.1f}%), "
            f"compile={compiles}/{total} ({compiles / total * 100:.0f}%), "
            f"indent={indent_errs}, wrong_method={wrong_method}"
        )

        # Write findings
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n### Cycle {cycle} ({ts})\n"
        entry += f"- Pass: {passes}/{total}, Compile: {compiles}/{total}\n"
        entry += f"- Errors: {indent_errs} indent, {wrong_method} wrong method\n"
        with open(self.findings_file, "a") as f:
            f.write(entry)

        # Update trajectory
        if self.state_file.exists():
            text = self.state_file.read_text()
            pass_rate = (passes / total * 100) if total > 0 else 0
            delta = f"+{pass_rate:.1f}" if pass_rate > 0 else "0"
            row = f"| {cycle} | {pass_rate:.1f}% | {compiles}/{total} | {delta} | train+eval |"
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("|---|"):
                    lines.insert(i + 1, row)
                    break
            self.state_file.write_text("\n".join(lines))

        return results

    # ── REASON ───────────────────────────────────────────────────────────

    def reason(self, cycle, results):
        """Think about what happened, decide what to do."""
        if results is None:
            self._log("REASON: First cycle. Just train.")
            return

        total = max(results.get("total", 1), 1)
        passes = results.get("passes_tests", 0)
        compiles = results.get("compiles", 0)

        # Log decision
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        if passes > 0:
            decision = f"pass={passes}/{total}. Keep training."
        elif compiles / total > 0.3:
            decision = f"compile={compiles / total * 100:.0f}% improving. Keep going."
        else:
            decision = f"compile={compiles / total * 100:.0f}%. Still learning."

        self._log(f"REASON: {decision}")

        with open(self.log_file, "a") as f:
            f.write(f"\n## {ts} — Cycle {cycle}\n\n- {decision}\n")

    # ── ACT ──────────────────────────────────────────────────────────────

    def act(self):
        """Train the model, then evaluate."""
        self._log("=" * 40)
        self._log("ACT: TRAIN")
        train_rc = self._run(self.train_cmd, cwd=str(self.train_cwd))
        if train_rc != 0:
            self._log(f"WARNING: train exit {train_rc}")

        self._log("=" * 40)
        self._log("ACT: EVAL")
        eval_rc = self._run(self.eval_cmd)
        if eval_rc != 0:
            self._log(f"WARNING: eval exit {eval_rc}")

        self._heartbeat()

    # ── LOOP ─────────────────────────────────────────────────────────────

    def one_cycle(self, cycle):
        self._log(f"\n{'=' * 50}")
        self._log(f"CYCLE {cycle}")
        self._log(f"{'=' * 50}\n")

        # OBSERVE (skip first cycle — no data yet)
        results = None
        if cycle > 1:
            results = self.observe(cycle)

        # REASON
        self.reason(cycle, results)

        # ACT
        self.act()

    def run(self):
        self._log("Starting single ReAct agent")
        self._log("Loop: OBSERVE → REASON → ACT → REPEAT\n")
        cycle = 0
        while True:
            cycle += 1
            try:
                self.one_cycle(cycle)
            except Exception as e:
                self._log(f"ERROR cycle {cycle}: {e}")
                time.sleep(10)
                continue
            if datetime.now().hour >= self.stop_hour:
                self._log(f"Reached hour {self.stop_hour}. Stopping.")
                break
            time.sleep(5)

    def run_once(self):
        self.one_cycle(1)

    def eval_only(self):
        self._run(self.eval_cmd)
        results = self._get_eval_results()
        if results:
            print(f"\n{results['passes_tests']}/{results['total']} pass")


# ── Auto-detect project from script location ────────────────────────────


def _default():
    project = Path(__file__).parent.parent
    nanogpt = project / "nanoGPT"
    return Agent(
        project_dir=str(project),
        train_cmd=["python", str(nanogpt / "train.py"), "config/train_leetcode.py"],
        eval_cmd=["python", str(project / "evaluate.py")],
        eval_results_path=str(nanogpt / "out-leetcode" / "eval_results.json"),
        train_cwd=str(nanogpt),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    agent = _default()
    if args.eval_only:
        agent.eval_only()
    elif args.once:
        agent.run_once()
    else:
        agent.run()
