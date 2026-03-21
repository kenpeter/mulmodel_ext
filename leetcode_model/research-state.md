# Research State

## Project

- **Title**: Fresh-train a LeetCode code generation model
- **Question**: Can a model trained from scratch on LeetCode CoT data generate correct code that compiles and passes test cases?
- **Status**: active — continuing training loop
- **Started**: 2026-03-20
- **Domain**: code generation, language model training

## Hypotheses

| ID | Statement | Status | Priority |
|----|-----------|--------|----------|
| H1 | A 32M-param model trained from scratch on 2M LeetCode tokens can learn to generate compilable Python code | active — compile rate 10% | high |
| H2 | Disabling sage attention fixes CUDA errors | confirmed — compile rate 0% → 10% | done |
| H3 | Model needs more data or larger size to learn code semantics | pending | medium |

## Experiments

- **Proxy metric**: pass@1 on test set (compile + pass test cases)
- **Baseline**: 0%
- **Best value**: 0.0% (compile: 23/228 = 10%)
- **Total runs**: 30

### Trajectory

| Run | Hypothesis | Pass% | Compile | Delta | Time (min) | Summary |
|-----|-----------|-------|---------|-------|------------|---------|
| 1-29 | H1 | 0.0% | 15-16/228 | 0 | 10 | Stuck with sage attention |
| 30 | H2 | 0.0% | 23/228 | +7 | 10 | Disabled sage attention |

## Outer Loop

- **Cycle**: 2
- **Last direction**: DEEPEN — compile rate improved, continue training
- **Last reflection**: 2026-03-21 — Disabling sage attention fixed CUDA errors. Compile rate 10%. Method names still wrong. Continue training loop.
