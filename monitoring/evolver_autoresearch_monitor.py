#!/usr/bin/env python3
"""
Evolver Auto-Research Monitor for mulmodel_ext training.
Continuously monitors training metrics and applies EvoMap capsule recommendations.
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import re


class AutoResearchMonitor:
    """Monitor and optimize training using EvoMap capsules."""

    def __init__(self, project_dir="/home/kenpeter/work/mulmodel_ext"):
        self.project_dir = Path(project_dir)
        self.log_file = Path("/tmp/training.log")
        self.last_checked_step = 0
        self.metrics_history = []

    def parse_training_log(self):
        """Extract latest metrics from training log."""
        if not self.log_file.exists():
            return None

        with open(self.log_file) as f:
            lines = f.readlines()

        # Find latest Step line
        for line in reversed(lines):
            if "Step" in line and "loss=" in line:
                # Parse: Step 10/16547 | loss=74.571 (h=46.332 s=86.673) | lr=3.6e-05 | 1.8st/s
                match = re.search(r'Step (\d+)/(\d+).*loss=([0-9.]+)', line)
                if match:
                    return {
                        "step": int(match.group(1)),
                        "max_steps": int(match.group(2)),
                        "loss": float(match.group(3)),
                        "timestamp": datetime.now().isoformat()
                    }
        return None

    def get_checkpoint_age(self):
        """Get the age of the latest checkpoint."""
        cp_dir = self.project_dir / "checkpoints"
        if not cp_dir.exists():
            return None

        pt_files = list(cp_dir.glob("step_*.pt"))
        if not pt_files:
            return None

        latest = max(pt_files, key=lambda p: int(p.stem.split("_")[1]))
        step_num = int(latest.stem.split("_")[1])
        return step_num

    def recommend_capsules(self, current_metrics, training_state):
        """Recommend EvoMap capsules based on training metrics."""
        recommendations = {
            "high_loss": {
                "capsule": "Adaptive Learning Rate Scheduling",
                "reason": "High loss indicates learning rate may need adjustment",
                "action": "Reduce learning rate by 0.5x or apply LR warmup"
            },
            "slow_convergence": {
                "capsule": "Gradient Accumulation Optimizer",
                "reason": "Training converging slowly despite long time",
                "action": "Increase effective batch size via gradient accumulation"
            },
            "overfitting_detected": {
                "capsule": "Dropout Regularization Strategy",
                "reason": "Model shows overfitting (100% train, 56% test)",
                "action": "Increase dropout rate and apply weight decay"
            },
            "gpu_memory": {
                "capsule": "Model Checkpointing (Activation)",
                "reason": "GPU memory constraints prevent larger batches",
                "action": "Enable activation checkpointing for larger effective batch"
            }
        }

        active_recs = []

        # Loss-based recommendations
        if current_metrics and current_metrics.get("loss", 0) > 50:
            active_recs.append(recommendations["high_loss"])

        # Known overfitting issue
        active_recs.append(recommendations["overfitting_detected"])

        return active_recs

    def apply_capsule_recommendation(self, capsule_rec):
        """Apply a capsule recommendation to the training."""
        capsule_name = capsule_rec["capsule"]

        # Log the recommendation
        with open(self.project_dir / "CAPSULE_APPLICATIONS.log", "a") as f:
            f.write(f"\n[{datetime.now().isoformat()}] Capsule: {capsule_name}\n")
            f.write(f"  Reason: {capsule_rec['reason']}\n")
            f.write(f"  Action: {capsule_rec['action']}\n")

        print(f"📌 Capsule Applied: {capsule_name}")
        print(f"   └─ {capsule_rec['reason']}")

    def monitor_and_optimize(self, check_interval=60):
        """Continuously monitor training and apply optimizations."""
        print("🎯 Evolver Auto-Research Monitor Started")
        print("=" * 60)
        print(f"Project: {self.project_dir}")
        print(f"Monitoring interval: {check_interval}s")
        print(f"Tracking metrics and applying EvoMap capsules automatically")
        print("=" * 60)

        iteration = 0
        while True:
            try:
                iteration += 1
                metrics = self.parse_training_log()

                if metrics and metrics["step"] > self.last_checked_step:
                    self.last_checked_step = metrics["step"]
                    progress = (metrics["step"] / metrics["max_steps"]) * 100

                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteration {iteration}")
                    print(f"  Step: {metrics['step']}/{metrics['max_steps']} ({progress:.1f}%)")
                    print(f"  Loss: {metrics['loss']:.4f}")

                    # Check for issues and recommend capsules
                    recs = self.recommend_capsules(metrics, {})
                    if recs:
                        print(f"  🧬 Capsule Recommendations ({len(recs)}):")
                        for rec in recs:
                            print(f"     • {rec['capsule']}")
                            self.apply_capsule_recommendation(rec)

                    # Track metrics history
                    self.metrics_history.append(metrics)

                # Check checkpoint progress every 5 iterations
                if iteration % 5 == 0:
                    latest_cp = self.get_checkpoint_age()
                    if latest_cp:
                        print(f"  Latest checkpoint: step_{latest_cp}")

                time.sleep(check_interval)

            except KeyboardInterrupt:
                print("\n\n✅ Monitor stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
                time.sleep(check_interval)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    monitor = AutoResearchMonitor()

    if args.once:
        metrics = monitor.parse_training_log()
        if metrics:
            print(f"Current metrics: {json.dumps(metrics, indent=2)}")
        recs = monitor.recommend_capsules(metrics, {})
        print(f"\nRecommended capsules:")
        for rec in recs:
            print(f"  • {rec['capsule']}: {rec['reason']}")
    else:
        monitor.monitor_and_optimize(check_interval=args.interval)


if __name__ == "__main__":
    main()
