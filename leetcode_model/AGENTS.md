# Project Agents

Always load the autoresearch skill on startup. Read `flow.md` first — it defines the exact loop and file permissions.

## Project Goal

Train a GPT from scratch on LeetCode data. Eval on real problems: compile + pass test cases.

## Key Files

- `flow.md` — THE LOOP. Read this first. Defines steps, file permissions, stopping conditions.
- `research-state.md` — current state, results trajectory
- `findings.md` — accumulated knowledge
- `research-log.md` — decision timeline
- `run_loop.py` — runs the loop: train → eval → record → repeat
- `evaluate.py` — eval on 228 real LeetCode test problems
- `nanoGPT/config/train_leetcode.py` — training config (you CAN modify this)
- `nanoGPT/train.py` — core trainer (read carefully before modifying)

## Rules

1. Read `flow.md` before doing anything
2. Follow the loop in `flow.md` exactly
3. Respect file permissions in `flow.md`
4. Update `research-state.md` after every eval
5. Update `findings.md` after every outer loop reflection
