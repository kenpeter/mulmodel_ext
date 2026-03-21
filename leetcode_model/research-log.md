# Research Log

## 2026-03-20 — Bootstrap

- Set up workspace for LeetCode code generation
- Data: newfacade_LeetCodeDataset (train: 93MB, test: 7.5MB)
- Chose nanoGPT for fresh training (no pre-trained weights)
- Model: 50M params (6 layers, 6 heads, 384 embd)
- Configured QLoRA removed — going pure from-scratch

## 2026-03-21 — Outer Loop 1: PIVOT

- Ran 29 cycles over ~5 hours. Pass rate: 0%. Compile rate: ~7% (16/228).
- **Critical finding**: Model generates `<|endoftext|>` at start of many outputs. Training data uses EOT as document separator — model learns to emit it immediately.
- **Critical finding**: Compiling code has wrong method names. Eval prompt only includes `problem_description`, not `starter_code` or method signature.
- **Critical finding**: Temperature=0.8 is too high for code generation.
- **Decision**: PIVOT — fix eval prompt (add starter_code), lower temperature to 0.2. If that doesn't help, consider data format changes or model size increase.
- Next: implement H2 (lower temperature) + H3 (add starter_code to prompt)

## 2026-03-21 — Changes Applied (H2 + H3)

- **evaluate.py**: Lowered temperature from 0.8 to 0.2 (H2)
- **evaluate.py**: Lowered top_k from 200 to 40 (more focused sampling)
- **evaluate.py**: Added starter_code to eval prompt (H3) — model now sees method signature
- **evaluate.py**: Strip `<|endoftext|>` from generated output
- Rationale: Model was generating garbage due to high temperature, no method signature in prompt, and EOT tokens at start

## 2026-03-21 — Key Fix: Sage Attention Disabled

- **model.py**: Disabled sage attention at inference (line 62: `and False`)
- **Result**: Compile rate went from 0% to 10% (23/228)
- **Root cause**: Sage attention was causing CUDA device-side assert errors during inference
- **Lesson**: Sage attention is optimized for training but can cause numerical issues during inference with certain inputs
- Model still has 0% pass rate — method names are wrong, code logic is wrong
- Training loss: 10.7 → 0.37 (good convergence)
- Val loss: 1.88 (overfitting gap of ~1.5)
- Next: continue training loop, consider increasing model size or data augmentation

## 2026-03-21 11:01 — Cycle 1: PIVOT

- Majority indentation errors. Tokenizer is the bottleneck.
