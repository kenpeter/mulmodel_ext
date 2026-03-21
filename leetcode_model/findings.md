# Research Findings — LeetCode Code Generation

## Research Question

Can a model trained from scratch on ~2M tokens of LeetCode CoT data generate code that compiles and passes test cases?

## Current Understanding

Model trained 29 cycles (~5 hours). Pass rate: 0%. Compile rate: ~7% (15-16/228). Model is not learning to generate correct code.

## Key Results

| Cycle | Pass | Compile | Notes |
|-------|------|---------|-------|
| 1-3   | 0%   | 7%      | First runs, baseline established |
| 4-10  | 0%   | 7%      | Stuck at 16/228 compile |
| 11-20 | 0%   | 6-7%    | Slight regression to 15/228 |
| 21-29 | 0%   | 3-7%    | Unstable, dropped to 8/228 at cycle 20 |

## Patterns and Constraints

### What's Working
- Training runs without errors
- Eval pipeline works
- Checkpoint saving/resuming works

### What's NOT Working (Critical Failures)

1. **`<|endoftext|>` at start**: Many outputs begin with `<|endoftext|>`. Model learns to emit EOT immediately because training data uses EOT as document separator. This is a fundamental data format problem.

2. **Wrong method names**: Compiling code has wrong method names:
   - `hashValue` instead of `stringHash`
   - `splitString` instead of `countGoodIntegers`
   - `maxXBip` instead of `maximumSubarrayXor`
   - Model doesn't learn to match entry points from problem description.

3. **Indentation errors**: ~60% of compile failures are indentation errors. Model doesn't understand Python indentation rules well.

4. **Temperature too high**: `evaluate.py` uses `temperature=0.8`. For code generation, this should be 0.2 or lower (greedy-ish).

5. **No method signature in prompt**: The eval prompt only includes `problem_description`, not the expected method signature. Model can't know what method name to use.

### Root Causes

- **Data format**: Problems separated by `<|endoftext|>`. Model learns to emit EOT early.
- **Prompt mismatch**: Eval prompt doesn't include `starter_code` or method signature.
- **Temperature**: 0.8 is too stochastic for code generation.
- **Model capacity**: 50M params on 2M tokens may be insufficient for learning code structure.

## Hypotheses for Next Cycle

| ID | Hypothesis | Status | Expected Impact |
|----|------------|--------|-----------------|
| H2 | Lower temperature to 0.2 in eval | FAILED — too deterministic for 32M model | — |
| H3 | Include `starter_code` in eval prompt | FAILED — model confused by format it wasn't trained on | — |
| H4 | Fix training data format (EOT at end, not between) | pending | HIGH |
| H5 | Include starter_code in training data | pending | HIGH |
| H6 | Increase model size (100M params) | pending | MEDIUM |
| H7 | Retrain from scratch with fixed data | pending | HIGH |

### What Does NOT Work (Tested)
- **autopep8 post-processing** — model generates structurally broken indentation, not PEP8 violations
- **Temperature 0.2** — too deterministic for a small model still learning
- **starter_code in eval prompt** — model wasn't trained with this format, adding at inference doesn't help

## Lessons and Constraints

- VRAM: 12GB RTX 4070 Ti — fits models up to ~350M params
- Training time: ~10 min per cycle (train + eval)
- Data: ~2M tokens from ~200 LeetCode problems
- GPT-2 tokenizer: 50,257 vocab size
- Model: nanoGPT, 6 layers, 6 heads, 384 embd (32.87M params)
- **Sage attention causes CUDA errors at inference** — disabled, using flash attention instead
- After disabling sage attention: compile rate 10% (23/228), still 0% pass
- Method names still wrong — model doesn't learn entry points from problem description

### Cycle 1 (2026-03-21 11:01)
- Pass: 0/228 (0.0%), Compile: 16/228
- Top error: indentation (166 cases)
- **Indents >50%** — tokenizer is the bottleneck
- **Wrong method names** — eval prompt needs starter_code

### Cycle 1 (2026-03-21 12:08)
- Pass: 0/30, Compile: 3/30
- Errors: 19 indent, 3 wrong method

### Cycle 1 (2026-03-21 12:15)
- Pass: 0/30, Compile: 1/30
- Errors: 25 indent, 1 wrong method

### Cycle 1 (2026-03-21 12:30) — Auto Research Session
- Tested: autopep8, temp=0.2, top_k=40, starter_code in prompt
- All changes made things worse or had no effect
- Baseline: 1/30 compile (3%), 0/30 pass
- **Key insight**: Inference-time fixes don't work for a 32M from-scratch model. Need DATA fixes.

### Cycle 2 (2026-03-21 14:30) — EOT Fix + Retrain
- Fixed prepare.py: encode_ordinary→encode, EOT at end of each document
- Retrained 5000 iters (block_size=512, no grad checkpointing)
- EOT issue: FIXED — no more `<|endoftext|>` at start of outputs
- Indentation issue: PERSISTS — model generates mixed 3/4/5 spaces
- Result: 1/30 compile (3%), 0/30 pass (same as before)
- **Key insight**: 32M params can't learn Python indentation rules

### 2026-03-21 22:37 — Cycle 1
- Compile: 4/30 (13%)
