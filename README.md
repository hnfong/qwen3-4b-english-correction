# qwen-small-english-teacher

Fine-tune of **Qwen3-4B-Instruct-2507** (no VL) that corrects badly written English
sentences — trained with **tinygrad** on the words.hk correction data in `raw_data/`.

## Use it

```bash
.venv/bin/python correct.py "He go to school yesterday."
# -> He went to school yesterday.

.venv/bin/python correct.py "This thingy is cute." --cantonese "呢嚿嘢好可愛。"
# -> This little thing is cute.

.venv/bin/python correct.py "The quick brown fox jumps over the lazy dog."
# -> (unchanged — the model returns correct sentences as-is)

.venv/bin/python correct.py "..." --no-adapter      # base model, no fine-tune
```

~20 s per invocation (model load + kernel compile, disk-cached; generation ~3 s).
LoRA adapter used: `out/lora_r16a/best` (override with `--adapter`).

## How it was built

1. **Data** (`scripts/build_dataset.py` → `data/train.jsonl` + `data/val.jsonl`):
   - Pairs (bad English → corrected English) from the three teacher TSV groups
     (gemma3/llama3/plain, ~105k rows) + the 9 human-reviewed CSVs (gold).
   - Each `original` field = [sense letter] + Cantonese sentence + English
     translation; the Cantonese is extracted and used as an optional prompt input.
   - Teacher refusals/empties filtered; identity pairs (target = original) created
     when the original scored ≥4/5 by reviewers and no suggestion existed.
   - 66k train / 1.4k val pairs; ~15% of training examples omit the Cantonese line
     so the model works without it. See `data/sample_review.txt` for samples.
2. **Model** (`scripts/qwen3.py`): Qwen3 (GQA + per-head QK-norm + RoPE + SwiGLU +
   tied embeddings) implemented in tinygrad, **validated exactly against HF
   transformers** (tmp/validate_qwen3.py). LoRA r=16 α=32 on all attention + MLP
   projections. Jitted decode + chunked prefill for speed.
3. **Training** (`scripts/train_lora.py` + `tmp/train_driver.sh`): masked-loss SFT
   (completion only), AdamW, fp16 base weights. macOS GPU-watchdog mitigations
   required (see JOURNAL.md): per-kernel submission pacing, one micro-batch per
   step, restart-loop driver (~45 steps, val loss 2.0 → 0.88).
4. **Eval** (`scripts/eval_synth.py` → `out/eval_report.txt`): 14 synthetic cases —
   all pass, including identity behavior on already-correct input.

## Regenerate / retrain

```bash
.venv/bin/python scripts/build_dataset.py                     # rebuild data
tmp/train_driver.sh                                           # restart-loop training
.venv/bin/python scripts/eval_synth.py --adapter out/lora_r16a/best
```

Setup: `python3.12 -m venv .venv && .venv/bin/pip install tinygrad tokenizers numpy`
(weights: `models/qwen3-4b-instruct/`, downloaded from HF; AGENTS.md protocol
followed — artifacts in `out/`, scratch in `tmp/`).

## Known limitations

- Trained on ~200 examples × ~12 effective epochs (GPU-watchdog constrained);
  more driver cycles would improve style coverage.
- GGUF export skipped (optional per user decision Q3).
- The Cantonese reference is optional at inference; quality with it is slightly
  better on translationese cases.
