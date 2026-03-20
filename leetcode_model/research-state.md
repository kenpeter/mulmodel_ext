# Research State

## Project

- **Title**: Fresh-train a LeetCode code generation model
- **Question**: Can a model trained from scratch on LeetCode CoT data generate correct code that compiles and passes test cases?
- **Status**: active
- **Started**: 2026-03-20
- **Domain**: code generation, language model training

## Hypotheses

| ID | Statement | Status | Priority |
|----|-----------|--------|----------|
| H1 | A 50M-param model trained from scratch on 2M LeetCode tokens can learn to generate compilable Python code | active | high |

## Experiments

- **Proxy metric**: pass@1 on test set (compile + pass test cases)
- **Baseline**: 0
- **Best value**: 0.0%
- **Total runs**: 1

### Trajectory

| Run | Hypothesis | Metric | Delta | Time (min) | Summary |
|-----|-----------|--------|-------|------------|---------|

## Outer Loop

- **Cycle**: 0
- **Last direction**: null
- **Last reflection**: (none yet)
