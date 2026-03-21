# Flow

## Code Rules

- Keep code simple, clean, small
- No over-engineering, no unnecessary abstractions
- Each file should do one thing
- If you can write it in 20 lines, don't write 200
- **Always use `conda`** — never use bare `pip` or `python` without conda env

## Research Sources

Only two sources. Everything else is noise.

| Source | Tool | Use for |
|--------|------|---------|
| **arXiv** | `websearch` query `site:arxiv.org` | Papers on training, architecture, code generation |
| **GitHub** | `websearch` query `site:github.com` | Issues, repos, implementations, error fixes |

### Verification

Before citing any finding:
1. **arXiv**: `webfetch` the paper URL — confirm it exists and says what you claim
2. **GitHub**: `webfetch` the repo/issue URL — confirm the issue/solution is real
3. **Log it**: Add citation (URL + key claim) to `findings.md` or `research-log.md`

Rule: Never cite without fetching. Never guess a URL.

## Ultimate Goal

Model generates Python code for LeetCode problems that:
1. **Compiles** — no syntax errors
2. **Passes test cases** — correct output for all inputs in the test set

Success metric: `pass@1` on `leetcode_test.jsonl` (228 problems).
Current: 0%.
Target: >0% = learning. >10% = good. >30% = great.

## Hardware

- **GPU**: NVIDIA RTX 4070 Ti — 12GB VRAM
- **CPU RAM**: 93GB total
- **OS**: Linux

---

## Research Loop

Everything is inside the loop. One cycle = one full pass.

```
┌─────────────────────────────────────────┐
│              RESEARCH LOOP              │
│                                         │
│  1. TRAIN                               │
│  2. EVAL                                │
│  3. REVIEW (review agent)               │
│  4. SEARCH (research agent)             │
│  5. CODE UPDATE (code change agent)     │
│                                         │
│  improving? → back to 1 (no changes)    │
│  worse/stable? → 4→5→1                  │
│  code bloated? → simplify → 1           │
│                                         │
│  NEVER STOP                             │
└─────────────────────────────────────────┘
```

### Loop Steps

| Step | What | Who |
|------|------|-----|
| 1. TRAIN | Run training script | main.py / PM |
| 2. EVAL | Run eval script | main.py / PM |
| 3. REVIEW | Audit results, check quality | Review agent |
| 4. SEARCH | Search arxiv + github for fixes | Research agent |
| 5. CODE UPDATE | Implement fix from research | Code Change agent |
| 6. SIMPLIFY | Remove bloat if needed | Code Simplify agent |

### When to Skip Steps

| Situation | Skip |
|-----------|------|
| Pass rate improving | Skip SEARCH, CODE UPDATE, SIMPLIFY → go straight to TRAIN |
| Pass rate stable | Run all steps |
| Pass rate dropped | Run all steps |
| Code is bloated | Run SIMPLIFY after CODE UPDATE |

Search happens every cycle except when improving. Code update only when worse/stable.

### Search Protocol

Every cycle after eval:
1. Read `findings.md` — what failed, what we tried
2. Search arXiv: `websearch("site:arxiv.org <specific problem>")`
3. Search GitHub: `websearch("site:github.com <error or technique>")`
4. Fetch top result with `webfetch` — confirm it's real
5. Write finding to `research-log.md` with URL citation

### Code Update Rule

**DO NOT update code if:**
- Pass rate is improving
- Last code change was <3 cycles ago and showing improvement

**DO update code if:**
- Pass rate stale for 3+ cycles
- Pass rate dropped
- Research found a specific fix with evidence (URL)

### Evidence of Research

Every research action leaves a trail:
- `research-log.md` — timestamped entry with arxiv/github URL
- `findings.md` — updated with new knowledge
- Code diff — the actual change made

If `research-log.md` has no URLs, research didn't happen.

### Recovery: Never Stop

The loop NEVER stops. Every cycle has an action. Use this fallback chain:

```
Code change failed?
  → REVERT the change
  → Log failure in research-log.md
  → Try next hypothesis from findings.md
  → No more hypotheses? → Spawn Research agent for new ideas
  → Research found nothing? → Try data fix (training data format)
  → Data fix requires retrain? → Retrain from scratch
  → Retrain still stuck? → Increase model size
  → Still stuck? → More training data, different architecture, keep going
  → ALWAYS have a next action
```

Rules:
- **NEVER stop the loop** — no exceptions, no waiting, no alerts
- **Always have a next action** — if stuck, train more, research again, try something
- **Always revert** before trying next hypothesis
- **Max retries per hypothesis**: 1 (try once, then move on)
- **Stuck = stale pass rate for 3+ cycles** — escalate to next level, don't stop

### Branching: Try Different Approaches in Parallel

Use git branches to test hypotheses without breaking main:

```
main                    ← stable, always works
├── fix/autopep8        ← try autopep8 on this branch
├── fix/temperature     ← try lower temp on this branch
├── fix/data-eot        ← fix training data EOT format
├── fix/model-100m      ← try larger model
```

Rules:
- Create branch for each hypothesis: `git checkout -b fix/<name>`
- Test on branch. If fails, delete branch, try next.
- If works, merge to main.
- Never test on main directly.
- Log branch name in research-log.md.

---

## Architecture: 5-Agent Team

```
leetcode_model/
├── main.py                         # Entry point — runs the loop (train+eval+PM)
├── evaluate.py                     # Eval logic (project-specific)
├── agents/
│   ├── pm.md                       # PM persona prompt
│   ├── research.md                 # Research persona prompt
│   ├── code_change.md              # Code Change persona prompt
│   ├── review_agent.md             # Review persona prompt
│   ├── code_simplify.md            # Code Simplify persona prompt
│   └── watchdog.py                 # Watchdog — crash recovery
├── nanoGPT/                        # Training project (unchanged)
│   ├── train.py
│   ├── model.py
│   └── config/train_leetcode.py
├── flow.md                         # THE SPEC — loop lives here
├── research-state.md               # Trajectory
├── findings.md                     # Knowledge
└── research-log.md                 # Decisions
```

```
                    ┌─────────┐
                    │   PM    │  ← reads review, assigns tasks
                    └────┬────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐
    │ Research  │  │Code Change│  │  Review   │
    │ arxiv/git │  │ fixes     │  │ audits    │
    └───────────┘  └───────────┘  └───────────┘
                                       │
                                  ┌────▼──────┐
                                  │  Simplify │
                                  │ trim code │
                                  └───────────┘
```

### Agents

| Agent | Role | Persona | Trigger |
|-------|------|---------|---------|
| **PM** | Decides next action | Decisive, no-fluff. Reads review. Assigns task to right agent. | Every cycle after eval |
| **Research** | Finds solutions | Curious, evidence-based. Only uses arxiv + github. Cites URLs. | When results stale/worse |
| **Code Change** | Implements fixes | Precise, minimal. One change at a time. Explains what and why. | When research finds a fix |
| **Review** | Audits everything | Critical, thorough. Checks research quality, code quality, eval quality, progress. | Every cycle |
| **Code Simplify** | Removes bloat | Ruthless. Deletes unused code. Shorter is better. | When code_change adds too much |

### Flow

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

---

## Error Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Training crashes (OOM, NaN) | Exit code != 0 | Log error, retry next cycle |
| Eval crashes (CUDA error) | Exit code != 0 | Skip eval, continue training |
| Agent hangs (no heartbeat) | heartbeat.txt stale >60s | Kill and restart |
| Agent crashes (exception) | Process dies | Watchdog restarts |
| GPU OOM | CUDA OOM error | Log, retry next cycle |

Restart policy:
- Max 5 restarts per rolling hour
- After 5 → write `to_human/ALERT.md`
- On restart → resume from last checkpoint

---

## Files

### Agent CAN modify

| File | What |
|------|------|
| `research-state.md` | Project state, results trajectory |
| `findings.md` | Accumulated knowledge |
| `research-log.md` | Decision timeline with URLs |
| `nanoGPT/config/train_leetcode.py` | Model size, lr, epochs |
| `nanoGPT/out-leetcode/*` | Checkpoints, eval results |
| `evaluate.py` | Eval logic |

### Agent CAN read but NOT modify

| File | Why |
|------|-----|
| `agents/*.md` | Persona prompts — don't change unless improving team |
| `nanoGPT/train.py` | Core trainer |
| `nanoGPT/model.py` | GPT architecture |
| `nanoGPT/data/leetcode/prepare.py` | Data pipeline |

### Agent MUST NOT touch

| File | Why |
|------|-----|
| `~/work/data/newfacade_LeetCodeDataset/leetcode_train.jsonl` | Raw training data |
| `~/work/data/newfacade_LeetCodeDataset/leetcode_test.jsonl` | Raw test data |

---

## Training Details

- Framework: nanoGPT (Karpathy)
- Model: 32.87M params (6 layers, 6 heads, 384 embd)
- Tokenizer: GPT-2 BPE (tiktoken)
- Data: ~2M tokens from LeetCode CoT
- Resumes from `nanoGPT/out-leetcode/ckpt.pt` after first run

## Eval Details

228 real LeetCode problems. For each:
1. Take `problem_description` as prompt
2. Model generates code (temperature=0.8, top_k=200)
3. Compile check: `compile(code, '<gen>', 'exec')`
4. Run test: execute + `check(candidate)` with assertions
5. Timeout: 10 seconds per test

Results saved to `nanoGPT/out-leetcode/eval_results.json`.

## How to Run

PM opencode reads flow.md and runs the loop:

```
# PM spawns itself, follows flow.md
opencode --prompt "Read leetcode_model/flow.md. Run the loop: train → eval → review → decide. Spawn agents as needed."

# One cycle only
opencode --prompt "Read leetcode_model/flow.md. Run ONE cycle: train → eval → review."

# Just eval
opencode --prompt "Read leetcode_model/flow.md. Run evaluate.py and review results."
```
