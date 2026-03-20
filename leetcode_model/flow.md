# Flow

## Code Rules

- Keep code simple, clean, small
- No over-engineering, no unnecessary abstractions
- Each file should do one thing
- If you can write it in 20 lines, don't write 200
- **Always use `conda`** — never use bare `pip` or `python` without conda env

## Search Sources

When stuck or hitting errors, the agent MUST search before guessing.

**Priority order:**

| Priority | Source | Tool | Best for |
|----------|--------|------|----------|
| 1 | **arXiv** | `arxiv` python lib | Papers on training, code generation |
| 2 | **GitHub** | `websearch` | Issues, repos, error messages, solutions |
| 3 | **Stack Overflow** | `websearch` | Python errors, training bugs |
| 4 | **HuggingFace** | `websearch` | Model configs, dataset formats |
| 5 | **Any URL** | `webfetch` | Read docs, blog posts, papers |
| 6 | **Code examples** | `codesearch` | How to use a library/API |

Rule: Never guess. Search arXiv and GitHub first.

## Paper Scout (auto-adapt papers to training)

Every 10 cycles, or when training plateaus, the agent MUST search for papers. Use whatever tools you feel are best:

- `arxiv` python lib — search by topic
- `websearch` — find papers, blog posts, GitHub repos
- `webfetch` — read any URL
- `codesearch` — find implementations
- `pymupdf` — read PDF papers

2. SCORE relevance (0-10):
   10 = code generation / LeetCode solving
   9  = small model training / data efficiency
   8  = pretraining strategies
   7  = attention / architecture changes
   5  = chain-of-thought / reasoning
   <5 = skip

3. READ the top paper (abstract + key findings)

4. ASK: can we adapt this to our pipeline?
   - Change model architecture? → modify nanoGPT/model.py
   - Change training strategy? → modify nanoGPT/config/train_leetcode.py
   - Change data format? → modify nanoGPT/data/leetcode/prepare.py
   - Change eval? → modify evaluate.py

5. IMPLEMENT if simple and high-impact
   - Keep it small (20 lines max per change)
   - Log what you changed in research-log.md
   - Log why in findings.md

6. REPEAT — train 5 min → eval → record
```

Relevant papers already found (log them here):
- Attention Residuals (Kimi, 2026) — replaces fixed residual connections with learned attention
- Mixture-of-Depths Attention (2026) — dynamic layer retrieval from all preceding layers
- The Finetuner's Fallacy (2026) — specialized pretraining beats finetuning-only
- Mamba-3 (2026) — improved sequence modeling with state space models

## Ultimate Goal

Model generates Python code for LeetCode problems that:
1. **Compiles** — no syntax errors
2. **Passes test cases** — correct output for all inputs in the test set

Success metric: `pass@1` on `leetcode_test.jsonl` (228 problems).
Current: 0%.
Target: as high as possible. >0% = learning. >10% = good. >30% = great.

## Hardware

- **GPU**: NVIDIA RTX 4070 Ti — 12GB VRAM (~11.6GB usable)
- **CPU RAM**: 93GB total (~84GB free)
- **OS**: Linux

Use GPU for training. CPU RAM is plenty for data loading, eval, anything.

## The Loop

```
LOOP FOREVER. Never stop unless human presses Ctrl+C. No max cycles. No goal check that stops the loop.

repeat forever:
    1. TRAIN  (~5 min)
       python nanoGPT/train.py config/train_leetcode.py

    2. EVAL   (run ALL 228 problems)
       python evaluate.py

    3. RECORD
       update research-state.md with results

    4. LOG
       if pass > 0%:  log "model is learning"
       if pass = 0% after 10 cycles: log "consider changing config"

    5. REPEAT — always repeat, never stop
```

Run the loop:
```bash
python run_loop.py          # automatic: train → eval → record → repeat
python run_loop.py --once   # one cycle only
python run_loop.py --eval   # skip training, just eval
```

## Files

### Agent CAN read, write, modify, delete

| File | What |
|------|------|
| `research-state.md` | Project state, results |
| `findings.md` | What we learned |
| `research-log.md` | Decisions |
| `nanoGPT/config/train_leetcode.py` | Model size, lr, epochs |
| `nanoGPT/out-leetcode/*` | Checkpoints, eval results, logs |
| `evaluate.py` | Eval logic |
| `run_loop.py` | The loop |
| `autoresearch.log` | Log file |
| `experiments/*` | Experiment dirs — create as needed |
| `data/*` | Processed data — create as needed |
| `src/*` | Shared code — create as needed |
| `to_human/*` | Reports — create as needed |
| Any new `.py`, `.md`, `.json`, `.txt`, `.yaml` file in project | New files freely |

### Agent CAN read but NOT modify

| File | Why |
|------|-----|
| `nanoGPT/train.py` | Core trainer — read to understand, don't change unless necessary |
| `nanoGPT/model.py` | GPT architecture — read to understand, don't change unless necessary |
| `nanoGPT/data/leetcode/prepare.py` | Data pipeline — read to understand, don't change unless necessary |
| `nanoGPT/sample.py` | Sampling code — read only |

### Agent MUST NOT touch — read or write

| File | Why |
|------|-----|
| `~/work/data/newfacade_LeetCodeDataset/leetcode_train.jsonl` | Raw training data — never modify |
| `~/work/data/newfacade_LeetCodeDataset/leetcode_test.jsonl` | Raw test data — never modify |
| `nanoGPT/.git/*` | Git internals |
| `nanoGPT/LICENSE` | License file |

## Training Details

- Framework: nanoGPT (Karpathy)
- Model: 50M params (6 layers, 6 heads, 384 embd)
- Tokenizer: GPT-2 BPE (tiktoken)
- Data: ~2M tokens from LeetCode CoT (train + test JSONL)
- First run: trains from scratch
- Later runs: resumes from `nanoGPT/out-leetcode/ckpt.pt`
- Config sets `init_from = 'resume'` automatically after first checkpoint

## Eval Details

Evaluates on REAL LeetCode problems from `leetcode_test.jsonl` (228 problems):

```
1.  [Medium] shortest-distance-after-road-addition-queries-i
2.  [Hard]   shortest-distance-after-road-addition-queries-ii
3.  [Medium] number-of-subsequences-with-odd-sum
4.  [Easy]   snake-in-matrix
5.  [Medium] count-the-number-of-good-nodes
6.  [Hard]   find-the-count-of-monotonic-pairs-i
7.  [Hard]   find-the-count-of-monotonic-pairs-ii
8.  [Medium] construct-string-with-minimum-cost-easy
9.  [Medium] find-the-power-of-k-size-subarrays-i
10. [Medium] find-the-power-of-k-size-subarrays-ii
... 228 total (Easy/Medium/Hard)
```

Each problem has:
- `problem_description` — the LeetCode problem statement
- `starter_code` — class/function skeleton
- `test` — `check(candidate)` function with assert statements
- `entry_point` — e.g. `Solution().twoSum`
- `prompt` — Python imports and helper classes (ListNode, TreeNode, etc.)

For each problem:
1. Take `problem_description` as prompt
2. Model generates code (temperature=0.2, greedy-ish)
3. Extract code from output (regex for ```python blocks)
4. Compile check: `compile(code, '<gen>', 'exec')`
5. Run test: execute code + `check(candidate)` function with all assertions
6. Timeout: 10 seconds per test

Results saved to `nanoGPT/out-leetcode/eval_results.json`.
