# Flow

## Code Rules

- Keep code simple, clean, small
- No over-engineering, no unnecessary abstractions
- Each file should do one thing
- If you can write it in 20 lines, don't write 200
- **Always use `conda`** — never use bare `pip` or `python` without conda env

## Search Sources

When stuck or hitting errors, the agent MUST search before guessing.

| Priority | Source | Tool | Best for |
|----------|--------|------|----------|
| 1 | **arXiv** | `arxiv` python lib | Papers on training, code generation |
| 2 | **GitHub** | `websearch` | Issues, repos, error messages, solutions |
| 3 | **Stack Overflow** | `websearch` | Python errors, training bugs |
| 4 | **HuggingFace** | `websearch` | Model configs, dataset formats |
| 5 | **Any URL** | `webfetch` | Read docs, blog posts, papers |
| 6 | **Code examples** | `codesearch` | How to use a library/API |

Rule: Never guess. Search arXiv and GitHub first.

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

## Architecture: 2-Agent + Watchdog

```
leetcode_model/
├── main.py                         # Entry point — wires agents to project
├── evaluate.py                     # Eval logic (project-specific)
├── agents/                         # Generic agent framework
│   ├── run_loop.py                 # RunLoop class — ReAct executor
│   ├── hypothesis.py               # HypothesisAgent class — the guide
│   └── watchdog.py                 # Watchdog class — error recovery
├── nanoGPT/                        # Training project (unchanged)
│   ├── train.py
│   ├── model.py
│   └── config/train_leetcode.py
├── flow.md                         # THE SPEC
├── research-state.md               # Trajectory
├── findings.md                     # Knowledge
└── research-log.md                 # Decisions
```

```
┌──────────────────────────────────────┐
│         main.py                      │
│  Wires agents to this project.       │
│  python main.py → watchdog starts    │
└──────────┬───────────────────────────┘
           │
┌──────────▼───────────────────────────┐
│         watchdog.py (class)          │
│  Spawns run_loop. Monitors heartbeat.│
│  Auto-restarts on crash/hang.        │
└──────────┬───────────────────────────┘
           │
┌──────────▼───────────────────────────┐
│         run_loop.py (class)          │
│  ReAct: Reason → Train → Eval       │
│  Writes heartbeat. Every 5 → hypo.  │
└──────────┬───────────────────────────┘
           │ every 5 cycles
┌──────────▼───────────────────────────┐
│         hypothesis.py (class)        │
│  Reads results. Decides direction.   │
└──────────────────────────────────────┘
```

### agents/ — Generic, Reusable

The `agents/` folder is a **standalone framework**. Copy it to any project.

```python
# For a new project, just import and configure:
from agents.run_loop import RunLoop
from agents.hypothesis import HypothesisAgent
from agents.watchdog import Watchdog

loop = RunLoop(
    project_dir="/path/to/project",
    train_cmd=["python", "train.py"],
    eval_cmd=["python", "eval.py"],
    eval_results_path="/path/to/results.json",
)
loop.run()
```

### main.py — Project Entry Point

Wires the generic agents to this LeetCode project. This is the only project-specific file.

```python
python main.py              # watchdog + loop + hypothesis
python main.py --run        # loop only
python main.py --once       # one cycle
python main.py --eval       # just eval
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
| `research-log.md` | Decision timeline |
| `loop_state.json` | Current ReAct reasoning |
| `heartbeat.txt` | Watchdog heartbeat |
| `hypothesis_analysis.json` | Last hypothesis decision |
| `nanoGPT/config/train_leetcode.py` | Model size, lr, epochs |
| `nanoGPT/out-leetcode/*` | Checkpoints, eval results |
| `evaluate.py` | Eval logic |
| `main.py` | Project entry point |
| `autoresearch.log` | Main log |
| `watchdog.log` | Watchdog log |

### Agent CAN read but NOT modify

| File | Why |
|------|-----|
| `agents/` | Generic framework — don't change unless improving the framework itself |
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

```bash
python main.py              # full system (watchdog + loop + hypothesis)
python main.py --run        # loop only, no watchdog
python main.py --once       # one cycle only
python main.py --eval-only  # just eval
python main.py --hypothesis # just hypothesis review
```
