# Research Findings — LeetCode Code Generation

## Research Question

Can a model trained from scratch on ~2M tokens of LeetCode CoT data generate code that compiles and passes test cases?

## Current Understanding

Starting point. No experiments run yet.

## Key Results

(No results yet)

## Patterns and Constraints

- Data: ~2M tokens from ~200 LeetCode problems (train + test JSONL)
- VRAM: 12GB (RTX 4070 Ti) — fits models up to ~350M params
- Model: nanoGPT, 50M params, fresh random initialization
- Training framework: nanoGPT (Karpathy's implementation)
- Evaluation: compile check + test case execution on held-out problems
