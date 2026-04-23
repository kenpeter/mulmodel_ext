#!/usr/bin/env python3
"""
Project Organization Script - Clean & Simple Structure.
Organizes files into logical directories.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

project_dir = Path("/home/kenpeter/work/mulmodel_ext")

# Define directory structure
STRUCTURE = {
    "scripts/": [
        "scripts/train_proven_local.py",
        "apply_correctness_capsules.py",
        "train_with_correctness_focus.py",
        "ACTIVATE_RECOVERY_CAPSULES.py",
        "ORGANIZE_PROJECT.py"
    ],
    "config/": [
        "AUTO_RESEARCH_CONFIG.json",
        "AUTO_RESEARCH_ARCHITECTURE_CONSTRAINTS.json",
        "CAPSULE_TRAINING_CONFIG.json",
        "TRAINING_PLAN_CORRECTNESS.json",
        "RECOVERY_CAPSULES_CONFIG.json"
    ],
    "docs/": [
        "AUTO_RESEARCH_SETUP.md",
        "EVOLVER_INTEGRATION.md",
        "EVOMAP_CAPSULES_INTEGRATED.md",
        "TRAINING_FAILURE_RECOVERY.md",
        "EVOLVER_OPTIMIZATION_REQUEST.md"
    ],
    "monitoring/": [
        "evolver_autoresearch_monitor.py",
        "evolver_auto_eval_integration.py",
        "correctness_verifier.py"
    ],
    "results/": [
        "AUTORESEARCH_RESULTS.tsv",
        "CAPSULE_APPLICATIONS.log",
        "AUTO_EVAL_HISTORY.json",
        "CORRECTNESS_VERIFICATION.json"
    ]
}

def create_structure():
    """Create organized directory structure."""
    print("📁 Creating organized project structure...")
    print("=" * 70)

    for dir_name in STRUCTURE.keys():
        dir_path = project_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created: {dir_name}")

    print()
    return STRUCTURE

def show_new_structure():
    """Display the new organized structure."""
    print("=" * 70)
    print("📊 NEW PROJECT STRUCTURE")
    print("=" * 70)

    structure_tree = """
    mulmodel_ext/
    ├── scripts/
    │   ├── train_proven_local.py          (Main training entry point)
    │   ├── apply_correctness_capsules.py  (Apply capsule optimizations)
    │   ├── train_with_correctness_focus.py (Strategy setup)
    │   └── ACTIVATE_RECOVERY_CAPSULES.py  (Safety mechanisms)
    │
    ├── config/
    │   ├── AUTO_RESEARCH_CONFIG.json              (Auto-research settings)
    │   ├── AUTO_RESEARCH_ARCHITECTURE_CONSTRAINTS.json (Protected components)
    │   ├── CAPSULE_TRAINING_CONFIG.json           (Active capsules)
    │   ├── TRAINING_PLAN_CORRECTNESS.json         (Training strategy)
    │   └── RECOVERY_CAPSULES_CONFIG.json          (Safety config)
    │
    ├── docs/
    │   ├── AUTO_RESEARCH_SETUP.md                 (How auto-research works)
    │   ├── EVOLVER_INTEGRATION.md                 (Evolver setup guide)
    │   ├── EVOMAP_CAPSULES_INTEGRATED.md          (Capsule descriptions)
    │   ├── TRAINING_FAILURE_RECOVERY.md           (Recovery mechanisms)
    │   └── EVOLVER_OPTIMIZATION_REQUEST.md        (Optimization details)
    │
    ├── monitoring/
    │   ├── evolver_autoresearch_monitor.py        (Monitor auto-research)
    │   ├── evolver_auto_eval_integration.py       (Monitor evaluation)
    │   └── correctness_verifier.py                (Verify test cases)
    │
    ├── results/
    │   ├── AUTORESEARCH_RESULTS.tsv               (Iteration log)
    │   ├── CAPSULE_APPLICATIONS.log               (Applied capsules)
    │   ├── AUTO_EVAL_HISTORY.json                 (Eval history)
    │   └── CORRECTNESS_VERIFICATION.json          (Test results)
    │
    ├── checkpoints/
    │   ├── step_*.pt                              (Model snapshots)
    │   ├── final.pt                               (Final model)
    │   └── eval_results.json                      (Latest eval)
    │
    ├── model/
    │   ├── config.py                              (Model config)
    │   └── student.py                             (Student model)
    │
    ├── scripts/
    │   └── train_proven_local.py                  (Training entry point)
    │
    ├── QUICK_START.md                             (Start here!)
    ├── PROJECT_STATUS.md                          (Current status)
    └── README.md                                  (Project overview)
    """

    print(structure_tree)

def create_quickstart():
    """Create a quick start guide."""
    quickstart = """# 🚀 Quick Start Guide

## Current Status
- Training: RUNNING (PID 120476)
- Correctness Monitor: ACTIVE
- Auto-Research Monitor: ACTIVE
- Recovery Capsules: 5 ACTIVE

## What's Running
```bash
# Check training progress
tail -f /tmp/training.log

# Monitor correctness
tail -f /tmp/correctness_monitor.log

# Watch auto-research decisions
tail -f /tmp/autoresearch_monitor.log
```

## Key Commands
```bash
# Check training status
ps aux | grep train_proven_local

# View latest eval results
cat results/CORRECTNESS_VERIFICATION.json

# See all auto-research iterations
tail -20 results/AUTORESEARCH_RESULTS.tsv

# Read capsule decisions
cat results/CAPSULE_APPLICATIONS.log
```

## Configuration
- Auto-research: `config/AUTO_RESEARCH_CONFIG.json`
- Protected components: `config/AUTO_RESEARCH_ARCHITECTURE_CONSTRAINTS.json`
- Recovery settings: `config/RECOVERY_CAPSULES_CONFIG.json`

## Documentation
- How Auto-Research works: `docs/AUTO_RESEARCH_SETUP.md`
- Recovery mechanisms: `docs/TRAINING_FAILURE_RECOVERY.md`
- All capsules: `docs/EVOMAP_CAPSULES_INTEGRATED.md`

## Goals
🎯 Current: 0% test-case passing
🎯 Target: 90%+ test-case passing
⏱️ Timeline: ~15 hours
"""

    quickstart_path = project_dir / "QUICK_START.md"
    with open(quickstart_path, "w") as f:
        f.write(quickstart)

    return quickstart_path

def create_project_status():
    """Create project status document."""
    status = """# 📊 Project Status

## Active Training
- **Process**: train_proven_local.py (PID 120476)
- **Step**: ~20-30/10351 (0.2% complete)
- **Loss**: Decreasing (69-76)
- **ETA**: ~73 minutes

## Capsules Active
1. ✅ Dropout Regularization Strategy
2. ✅ Weight Decay Optimization
3. ✅ Data Augmentation Pipeline
4. ✅ Adaptive Learning Rate Scheduling
5. ✅ Distillation Quality Monitor (every 500 steps)

## Recovery Mechanisms
1. ✅ Regression Detection (alert > 5%, rollback > 15%)
2. ✅ Model Checkpointing (every 200 steps)
3. ✅ Gradient Accumulation (OOM prevention)
4. ✅ Adaptive Learning Rate (loss divergence)
5. ✅ Gradient Clipping (gradient explosion)

## Monitoring
- Correctness Verifier: /tmp/correctness_monitor.log (PID 120616)
- Auto-Research Monitor: /tmp/autoresearch_monitor.log (PID 120617)
- Training Log: /tmp/training.log

## Target Metrics
- Test-Case Passing: 0% → 90%+
- Code Structure: 25% → 80%+
- Train/Val Gap: 44% → <20%

## Protected Components
- ✅ OpenMythos architecture (Prelude → Recurrent → Coda)
- ✅ Kimi Linear Attention
- ✅ Attention Residual connections
- ✅ Distillation from teacher

## What's Optimizable
- Learning rate: 1e-5 → 5e-4
- Dropout: 0.1 → 0.5
- Weight decay: 0.0 → 0.1
- Gradient clipping: 0.5 → 2.0
- Data augmentation strategies
- Loss function weights

---
Last Updated: 2026-04-23 21:25 UTC
"""

    status_path = project_dir / "PROJECT_STATUS.md"
    with open(status_path, "w") as f:
        f.write(status)

    return status_path

def main():
    print()
    print("🧹 PROJECT ORGANIZATION & CLEANUP")
    print("=" * 70)
    print()

    # Create structure
    create_structure()

    # Show new structure
    show_new_structure()

    # Create documentation
    print()
    print("📝 Creating documentation...")
    quickstart = create_quickstart()
    status = create_project_status()
    print(f"✓ Quick start: {quickstart}")
    print(f"✓ Project status: {status}")

    print()
    print("=" * 70)
    print("✅ PROJECT ORGANIZED!")
    print("=" * 70)
    print()
    print("📂 Directory structure created for:")
    print("  • scripts/      - Python execution files")
    print("  • config/       - JSON configuration files")
    print("  • docs/         - Documentation & guides")
    print("  • monitoring/   - Monitoring & verification scripts")
    print("  • results/      - Training results & logs")
    print()
    print("📖 Start here: QUICK_START.md")
    print("📊 Check status: PROJECT_STATUS.md")
    print()

if __name__ == "__main__":
    main()
