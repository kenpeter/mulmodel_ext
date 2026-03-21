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

| ID | Hypothesis | Expected Impact |
|----|------------|-----------------|
| H2 | Lower temperature to 0.2 in eval | Reduce randomness, more deterministic code |
| H3 | Include `starter_code` in eval prompt | Model knows method signature to generate |
| H4 | Include method signature in training data | Model learns entry points |
| H5 | Increase model size (100M params) | More capacity for code patterns |
| H6 | Filter training data to remove EOT artifacts | Clean data |
| H7 | Add code-specific data augmentation | More diverse training examples |

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
