# Project Agents

Read `flow.md` first — it defines the loop and 3-agent team.

## 3-Agent Team

Each agent is an opencode instance spawned via `task()`.

| Agent | File | Role |
|-------|------|------|
| **PM** | `agents/pm.md` | Reads eval results, searches arxiv/github, decides what to do |
| **Code Change** | `agents/code_change.md` | Implements code fixes based on research |
| **Code Review** | `agents/code_review.md` | Reviews code changes, debates with Code Change agent |

## Loop

```
TRAIN → EVAL → PM reads results → decides:
  improving? → TRAIN again
  stale? → PM spawns:
    → Research (arxiv/github search)
    → Code Change (implements fix)
    → Code Review (reviews change, debates with Code Change)
  → TRAIN again
```

## How to Run

PM opencode reads `flow.md` and runs the loop:

```bash
# Full loop
opencode --prompt "Read leetcode_model/flow.md. Run the loop."

# One cycle
opencode --prompt "Read leetcode_model/flow.md. Run ONE cycle."

# Just eval
opencode --prompt "Run python leetcode_model/evaluate.py and review results."
```

## Key Files

- `flow.md` — THE SPEC. Loop, agents, file permissions.
- `agents/*.md` — Agent persona prompts
- `evaluate.py` — Eval on 30 LeetCode problems
- `nanoGPT/train.py` — Core trainer (read-only)
- `nanoGPT/model.py` — GPT architecture (careful)
- `nanoGPT/config/train_leetcode.py` — Training config (you CAN modify)

## Rules

1. Read `flow.md` before doing anything
2. Spawn agents via `task()` with persona from `agents/*.md`
3. Respect file permissions in `flow.md`
4. Update `research-state.md` after every eval
5. Update `findings.md` after every review
6. Always log decisions to `research-log.md` with URLs
