# Project Agents

Read `flow.md` first — it defines the loop and 5-agent team.

## 5-Agent Team

Each agent is an opencode instance spawned via `task()`.

| Agent | File | Role |
|-------|------|------|
| **PM** | `agents/pm.md` | Reads review, assigns tasks, decides next action |
| **Research** | `agents/research.md` | Searches arxiv + github, logs findings with URLs |
| **Code Change** | `agents/code_change.md` | Makes one small code change per invocation |
| **Review** | `agents/review_agent.md` | Audits research, code, eval, and progress quality |
| **Code Simplify** | `agents/code_simplify.md` | Deletes dead code, removes bloat |

## Loop

```
TRAIN → EVAL → PM reads review → decides:
  improving? → TRAIN again
  stale/worse? → PM assigns:
    → Research (search arxiv+github)
    → Code Change (implement fix)
    → Review (audit the change)
    → Simplify (trim if needed)
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
- `evaluate.py` — Eval on 228 LeetCode problems
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
