# Research State

## Project

- **Title**: Fresh-train a LeetCode code generation model
- **Question**: Can a model trained from scratch on LeetCode CoT data generate correct code that compiles and passes test cases?
- **Status**: active — but fundamentally limited by model size
- **Started**: 2026-03-20
- **Domain**: code generation, language model training

## Hypotheses

| ID | Statement | Status | Priority |
|----|-----------|--------|----------|
| H1 | A 32M-param model trained from scratch can generate compilable Python code | SUPPORTED (83% compile) but WRONG semantics | high |
| H2 | Disabling sage attention fixes CUDA errors | confirmed | done |
| H3 | Indentation normalization improves compile rate | confirmed — 7% → 83% | done |
| H4 | Method name correction fixes entry point errors | FAILED — model generates broken code structure | done |
| H5 | 32M params insufficient for semantic code generation | confirmed — literature says 10B+ needed | done |

## Experiments

- **Proxy metric**: pass@1 on test set
- **Baseline**: 0%
- **Best value**: 0.0% pass, 83% compile (25/30)
- **Total runs**: 35+

### Trajectory (Recent)

| Run | Pass% | Compile | Delta | Summary |
|-----|-------|---------|-------|---------|
| 33 | 0% | 67% (20/30) | +34% | Indentation normalization added |
| 34 | 0% | 73% (22/30) | +6% | Method name extraction from entry_point |
| 35 | 0% | 83% (25/30) | +10% | Method name correction integrated |

## Outer Loop

- **Cycle**: 4
- **Last direction**: PIVOT — model capacity is the bottleneck, not code format
- **Last reflection**: 2026-03-22 — Compile rate improved to 83% with post-processing. BUT raw model output is completely broken (garbled syntax). 32M params cannot learn code generation. Literature confirms: Phi-4 14B needed for competitive programming. Next: either (1) increase model size significantly, (2) use pre-trained model, or (3) conclude this hypothesis is not viable.
