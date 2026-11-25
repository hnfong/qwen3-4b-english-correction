# PLAN 02 — MLX full fine-tune of Qwen3-1.7B + Qwen3-0.6B (per updated AGENTS.md)

Updated acceptance criteria (2025-11-25): models based on Qwen3-1.7B AND
Qwen3-0.6B, runnable (EN sentence + optional Cantonese → correction or original),
trained with ≥ half the data in raw_data/. Framework: **MLX** (port from
tinygrad, remove watchdog workarounds, no tinygrad references in final
pipeline). **Full fine-tuning** (no LoRA). Train with all valid data; two-stage
curriculum (machine TSV data → human-reviewed CSV data). LR: determine
empirically. Training runs ≤30 min until confident.

## Data accounting (criterion: ≥50% of raw_data)

- raw_data/: 105,101 rows (339 TSV + 9 CSV) → 35,750 unique dictionary entries
  → 66,167 SFT pairs (data/train.jsonl) + 1,374 val. Training on the full 66k
  pairs = trained on 100% of valid raw_data (criterion ≥50% met).
- Split for curriculum: stage 1 = machine-teacher pairs (gemma3/llama3/plain +
  identity), stage 2 = human-reviewed pairs (source=="human", ~2.2k).
- "Improve understanding of basic Cantonese": the Cantonese reference is part
  of every prompt; additionally ~15% of examples omit it (robustness). This
  also FIXES a README inaccuracy (the 15% omission was planned but never
  implemented in the tinygrad pipeline — implemented now).

## Phases

### M1 — MLX environment  [ ]
- [ ] pip install mlx, mlx-lm (into .venv; final pipeline must not import tinygrad)
- [ ] Verify mlx_lm.lora full-fine-tune support (--train-type full or equiv)
      + data format (prompt/completion jsonl, completion-only loss)

### M2 — Data prep  [ ]
- USER DIRECTIVE: new code must NOT handle think tokens at all (no stripping).
  Inference on base thinking models won't work — accepted. Fine-tuned models
  never emit them (SFT targets are direct answers).
- [ ] scripts/prep_mlx_data.py: train.jsonl → tmp/mlx_data/stage1.jsonl,
      stage2.jsonl, valid.jsonl (prompt = exact Qwen chat template w/ optional
      Cantonese [15% omitted deterministically], completion = target + <|im_end|>)

### M3 — Smoke + LR determination (0.6B, 30-min capped)  [ ]
- [ ] 30-min full-FT run on 0.6B: verify no watchdog issues, measure tok/s,
      memory, loss curve; pick LR (start 1e-5; adjust if flat/unstable)
- [ ] Checkpoint save/resume verified

### M4 — 0.6B full runs  [ ]
- [ ] Stage 1 (machine data, 1 epoch), stage 2 (human data, few epochs, low LR)
- [ ] eval_synth on fine-tuned model; correct.py rewritten on MLX

### M5 — 1.7B full runs  [ ]
- [ ] Same as M4 for 1.7B; eval

### M6 — Export + llama.cpp  [ ]
- [ ] MLX checkpoints → HF-format safetensors → GGUF f16/q8_0 → llama.cpp verify

### M7 — Publish + docs  [ ]
- [ ] HF: merged models + GGUFs for both sizes (same repo; 4B/LoRA artifacts stay)
- [ ] GitHub: MLX-only pipeline (tinygrad code remains in git history only),
      READMEs rewritten (framework, training, usage), JOURNAL
- [ ] QUESTIONS_FOR_USER.md: document Cantonese-interpretation decision + LR choice

## Key risks
- mlx_lm full-FT memory (1.7B: ~25 GB peak w/ AdamW fp32 moments — 96 GB OK)
- MLX + watchdog: expected fine (Apple's own framework); verify in M3
- convert_hf_to_gguf on MLX-saved checkpoints: MLX saves HF-compatible keys;
  verify config.json completeness (tie_word_embeddings etc.)
