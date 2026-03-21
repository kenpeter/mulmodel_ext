# Project Agents

Read `flow.md` first — it defines the loop, file permissions, and architecture.

## Project Goal

Train a GPT from scratch on LeetCode data. Eval on real problems: compile + pass test cases.

## Structure

```
leetcode_model/
├── evaluate.py                     # Eval on 228 LeetCode problems
├── agents/                         # Generic agent framework (reusable)
│   ├── run_loop.py                 # ReAct executor class
│   ├── hypothesis.py               # Hypothesis guide class
│   └── watchdog.py                 # Watchdog safety net class
├── nanoGPT/                        # Training project (don't restructure)
│   ├── train.py                    # Core trainer
│   ├── model.py                    # GPT architecture
│   └── config/train_leetcode.py    # Training config
├── flow.md                         # THE SPEC — read this first
├── research-state.md               # Results trajectory
├── findings.md                     # Accumulated knowledge
└── research-log.md                 # Decision timeline
```

## How to Run

```bash
python agents/watchdog.py python agents/run_loop.py   # full system
python agents/run_loop.py          # loop only
python agents/run_loop.py --once   # one cycle
python agents/run_loop.py --eval-only  # just eval
python agents/hypothesis.py . nanoGPT/out-leetcode/eval_results.json  # review
```

## Key Files

- `flow.md` — THE SPEC. Architecture, loop steps, file permissions.
- `main.py` — Wires agents to this project. Run this.
- `agents/run_loop.py` — Generic ReAct executor (class-based, reusable)
- `agents/hypothesis.py` — Generic hypothesis agent (class-based, reusable)
- `agents/watchdog.py` — Generic watchdog (class-based, reusable)
- `evaluate.py` — Eval logic (project-specific)
- `nanoGPT/config/train_leetcode.py` — Training config (you CAN modify)

## Rules

1. Read `flow.md` before doing anything
2. Run via `main.py`, not agents directly
3. Respect file permissions in `flow.md`
4. Update `research-state.md` after every eval
5. Update `findings.md` after every outer loop reflection
