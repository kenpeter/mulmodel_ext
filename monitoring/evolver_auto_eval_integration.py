#!/usr/bin/env python3
"""
Auto-Eval Capsule Integration for mulmodel_ext.
Integrates EvoMap's Auto-Evaluation Pipeline with training.
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import re


class AutoEvalCapsuleIntegration:
    """Integrate EvoMap Auto-Evaluation Pipeline with training."""

    def __init__(self, project_dir="/home/kenpeter/work/mulmodel_ext"):
        self.project_dir = Path(project_dir)
        self.log_file = Path("/tmp/training.log")
        self.eval_history = []
        self.results_file = self.project_dir / "AUTORESEARCH_RESULTS.tsv"

    def get_current_step(self):
        """Extract current step from training log."""
        if not self.log_file.exists():
            return 0

        with open(self.log_file) as f:
            lines = f.readlines()

        for line in reversed(lines):
            if "Step" in line:
                match = re.search(r'Step (\d+)/', line)
                if match:
                    return int(match.group(1))
        return 0

    def should_run_eval(self, current_step, eval_interval=2000):
        """Check if evaluation should run at this step."""
        # Auto-eval runs every 2000 steps (aligned with training script)
        return current_step > 0 and current_step % eval_interval == 0

    def parse_eval_results(self):
        """Parse latest evaluation results from training log."""
        if not self.log_file.exists():
            return None

        with open(self.log_file) as f:
            content = f.read()

        # Look for eval block
        if "Eval on rotating" in content or "Accuracy:" in content:
            # Parse accuracy percentage
            match = re.search(r'Accuracy:\s*([\d.]+)%', content)
            if match:
                return {
                    "accuracy": float(match.group(1)),
                    "timestamp": datetime.now().isoformat()
                }
        return None

    def track_eval_metrics(self, step, accuracy):
        """Track evaluation metrics for trending."""
        metric = {
            "step": step,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        }
        self.eval_history.append(metric)

        # Save to file
        with open(self.project_dir / "AUTO_EVAL_HISTORY.json", "a") as f:
            f.write(json.dumps(metric) + "\n")

    def get_accuracy_trend(self):
        """Analyze accuracy trend to detect issues."""
        if len(self.eval_history) < 2:
            return None

        recent = self.eval_history[-3:]  # Last 3 evals
        accuracies = [m["accuracy"] for m in recent]

        trend = {
            "current": accuracies[-1],
            "previous": accuracies[-2] if len(accuracies) > 1 else None,
            "change": accuracies[-1] - (accuracies[-2] if len(accuracies) > 1 else accuracies[-1]),
            "direction": "improving" if accuracies[-1] > (accuracies[-2] if len(accuracies) > 1 else 0) else "declining"
        }
        return trend

    def generate_eval_report(self):
        """Generate comprehensive evaluation report."""
        if not self.eval_history:
            return None

        report = {
            "capsule": "Auto-Evaluation Pipeline",
            "status": "active",
            "evals_completed": len(self.eval_history),
            "best_accuracy": max(m["accuracy"] for m in self.eval_history),
            "latest_accuracy": self.eval_history[-1]["accuracy"],
            "trend": self.get_accuracy_trend(),
            "timestamp": datetime.now().isoformat()
        }
        return report

    def log_eval_decision(self, step, accuracy, decision):
        """Log auto-eval decision to results file."""
        with open(self.results_file, "a") as f:
            f.write(f"\nAuto-Eval Step {step}: Accuracy={accuracy:.1f}% | Decision: {decision}\n")

    def monitor_evals(self, check_interval=30, eval_interval=2000):
        """Continuously monitor and trigger auto-evaluations."""
        print("🎯 Auto-Eval Capsule Monitor Started")
        print("=" * 60)
        print("Capsule: Auto-Evaluation Pipeline")
        print(f"Interval: Every {eval_interval} training steps")
        print(f"Check frequency: Every {check_interval}s")
        print("=" * 60)

        last_eval_step = 0
        iteration = 0

        while True:
            try:
                iteration += 1
                current_step = self.get_current_step()

                if current_step > last_eval_step and self.should_run_eval(current_step, eval_interval):
                    print(f"\n✅ Auto-Eval Triggered at Step {current_step}")

                    # Wait for eval results to appear in log
                    time.sleep(60)  # Give eval 1 minute to complete

                    eval_result = self.parse_eval_results()
                    if eval_result:
                        accuracy = eval_result["accuracy"]
                        self.track_eval_metrics(current_step, accuracy)

                        trend = self.get_accuracy_trend()
                        print(f"   Accuracy: {accuracy:.1f}%")
                        if trend:
                            print(f"   Trend: {trend['direction']} ({trend['change']:+.1f}%)")

                            # Make decision
                            if trend['direction'] == "improving":
                                decision = "KEEP current capsules, continue"
                            elif accuracy < 56:  # Below baseline
                                decision = "ALERT: Accuracy decreased, may need rollback"
                            else:
                                decision = "MONITOR: Small change, continue"

                            self.log_eval_decision(current_step, accuracy, decision)
                            print(f"   Decision: {decision}")

                        last_eval_step = current_step

                # Status every 5 iterations
                if iteration % 5 == 0:
                    report = self.generate_eval_report()
                    if report:
                        print(f"\n📊 Eval Report:")
                        print(f"   Completed: {report['evals_completed']}")
                        print(f"   Best: {report['best_accuracy']:.1f}%")
                        print(f"   Latest: {report['latest_accuracy']:.1f}%")

                time.sleep(check_interval)

            except KeyboardInterrupt:
                print("\n✅ Monitor stopped")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(check_interval)

    def report(self):
        """Print comprehensive auto-eval report."""
        report = self.generate_eval_report()
        if report:
            print("\n" + "=" * 60)
            print("🎯 AUTO-EVAL CAPSULE REPORT")
            print("=" * 60)
            print(f"Capsule: {report['capsule']}")
            print(f"Status: {report['status']}")
            print(f"Evaluations Completed: {report['evals_completed']}")
            print(f"Best Accuracy: {report['best_accuracy']:.1f}%")
            print(f"Latest Accuracy: {report['latest_accuracy']:.1f}%")
            if report['trend']:
                print(f"Trend: {report['trend']['direction']} ({report['trend']['change']:+.1f}%)")
            print("=" * 60)
        else:
            print("No evaluation data yet")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--report", action="store_true", help="Show evaluation report")
    args = parser.parse_args()

    monitor = AutoEvalCapsuleIntegration()

    if args.report:
        monitor.report()
    elif args.monitor:
        monitor.monitor_evals()
    else:
        # Quick status
        step = monitor.get_current_step()
        print(f"Current Step: {step}")
        print(f"Next Eval: Step {((step // 2000) + 1) * 2000}")


if __name__ == "__main__":
    main()
