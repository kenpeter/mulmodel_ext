# 🚀 Quick Start Guide

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
