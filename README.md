# mulmodel_ext — Knowledge Distillation: Qwen Teacher → Small Student

Training a ~172M parameter student model via knowledge distillation from a Qwen3.5-0.8B teacher, targeting LeetCode code generation.

---

## Architecture

### Student Model (~172M params)
| Component | Value |
|-----------|-------|
| Hidden dim | 512 |
| Layers | 12 |
| Attention heads | 8 (GQA, 2 KV heads) |
| Head dim | 64 |
| FFN dim | 2048 |
| Vocab | 248,320 |
| Attention | Kimi Delta Attention (KDA) |
| Extras | attn_residual, gradient checkpointing, weight-tied lm_head |

### Teacher Model
- `Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled`
- Loaded in bfloat16, frozen, used only for soft KDA targets

### Optimizer
- **MuonClip** — Muon with gradient clipping
- lr=2e-4, cosine decay to 2e-5 floor, warmup 200 steps, grad_accum=4

### Distillation Loss
- `loss = 0.5 * hard_CE + 0.5 * soft_KDA`
- Soft loss: top-64 sparse KL divergence between student and teacher logits

---

## Dataset
- `~/work/data/high_quality_leetcode/train.jsonl`
- 2,373 LeetCode problems with chain-of-thought reasoning
- Format: `{"problem": ..., "chain_of_thought": ..., "solution": ...}`

---

## Training Runs

### Run 1 — Initial KDA Training (from scratch)
- **Script**: `scripts/train_kda_muon.py`
- **Steps**: 5,936 (time-budget limited to ~06:00)
- **Duration**: ~118 min
- **Loss**: started ~27 volatile, ended ~28–35 (very volatile, no convergence)
- **Notes**: Teacher init copied 0 tensors (hidden 512 ≠ teacher hidden 1024). Student trained from random init with KDA soft targets only.

### Run 2 — Architecture Fix + Fresh Init
- **Changes**: Custom weight init with depth-scaling for MLP layers; 120M→179M param resize
- **Issue**: Logit explosion at lm_head (std=27.75 before fix, 0.45 after)
- **Fix**: `model/student.py` — `init_weights()` with `std = 0.02 / sqrt(2 * n_layers)`
- **Steps**: 15,000
- **Duration**: ~250 min (4.2h)
- **Loss trajectory**: 15.48 → 13.77 (stable descent, no divergence)
- **Hard loss final**: 10.44 | **Soft loss final**: 15.19
- **Inline eval (every 500 steps, 5 problems)**: 0/5 pass rate throughout
- **Syntax rate**: 1/5 from step ~1000 onward
- **Checkpoint**: `checkpoints/final.pt` (step_15000), 358 MB

### Run 3 — Resume from step_15000
- **Steps**: 5,893 more (20,893 cumulative)
- **Duration**: ~85 min
- **Loss final**: ~13.77 composite (h=10.50, s=15.22)
- **Note**: Time-budget bug — script estimated 12s/step but actual was 0.83s/step, so training stopped early. Fixed in script.

---

## Known Issues

| Issue | Status |
|-------|--------|
| Teacher init copies 0 tensors | Known — architecture mismatch (student 512 vs teacher 1024 hidden). KDA soft loss still active via teacher forward pass. |
| 0% LeetCode pass rate | Open — model generates plausible-looking Python but fails execution |
| Time-budget step estimate wrong | Fixed — changed `/ 3.0` to `/ 0.9`, `max_steps` uncapped |
| Optimizer state not saved in checkpoints | Known — cold-start on resume |
| Config defaults mismatch | `model/config.py` defaults (hidden=1024, layers=24) differ from training overrides (512, 12). Always pass explicit config. |

---

## Evaluation

```bash
# Quick eval — latest checkpoint, 10 problems
python eval_quick.py

# More problems
python eval_quick.py --n 30

# Full LeetCode eval (228 problems)
python eval/leetcode_eval.py --full

# Specific checkpoint
python eval_quick.py --checkpoint checkpoints/step_15000.pt
```

---

## Running Training

```bash
# Fresh run (time-limited to 06:00)
python scripts/train_kda_muon.py

# Resume from latest checkpoint
python scripts/train_kda_muon.py --resume

# Background (logs to train.log)
nohup python scripts/train_kda_muon.py --resume > train.log 2>&1 &
```

Checkpoints saved every 200 steps to `checkpoints/step_N.pt`. Final saved as `checkpoints/final.pt`.

---

## File Structure

```
scripts/train_kda_muon.py   — main training script
model/student.py            — student model (KDA transformer)
model/config.py             — StudentConfig (override defaults explicitly)
model/attention.py          — Kimi Delta Attention implementation
model/optimizer.py          — MuonClip optimizer
eval/leetcode_eval.py       — full LeetCode evaluation harness
eval_quick.py               — fast eval script
checkpoints/                — saved model weights
train.log                   — live training log
```

---

## Next Steps

- [ ] Investigate 0% pass rate — inspect generated code samples
- [ ] Try smaller teacher (0.5B) to reduce architecture gap
- [ ] Add greedy decode sampling during training eval
- [ ] Consider supervised fine-tuning warmup before KDA
