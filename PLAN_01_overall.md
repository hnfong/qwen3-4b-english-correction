# PLAN 01 — Overall: Qwen3-4B English corrector (tinygrad)

Goal: fine-tune Qwen3-4B (no VL) with tinygrad on the words.hk correction data so that
`correct.py "English sentence" [--cantonese "..."]` prints a corrected sentence
(or the original if already fine). User decisions recorded in QUESTIONS_FOR_USER.md.

Status markers: [ ] todo · [~] in progress · [x] done

## Phases

### P0 — Data audit  [x]
- 339 TSVs (gemma3 ×166, llama3 ×83, plain ×90, ~105k rows) + 9 human-reviewed CSVs.
- Same underlying sentences across groups; `reviews` judge the ORIGINAL English
  (identical across groups); `suggested_translation` differs per teacher.
- `original` = [optional sense letter "A."–"J."] + Cantonese + space + English.
  ~98% contain CJK; ~0.4% English-only. Regex split OK for ~84% — needs robust
  last-CJK-boundary splitter.
- Quirks: teacher refusals ("Skipped due to inappropriate content."), empty
  suggestions (30% gemma3, 15% llama3), duplicate rows across files, review scores
  0–5 from two reviewer types (G./H.).
- Audit scripts: tmp/audit1.py, tmp/audit2.py (tmp/ is gitignored; findings here).

### P1 — Environment setup  [ ]
- [ ] venv (python3.12), tinygrad + tokenizers + numpy installed
- [ ] Clone tinygrad to ./tinygrad_ref for reference (examples/llama.py etc.)
- [ ] Verify tinygrad examples/llama.py Qwen3 support (arch + tokenizer)

### P2 — Dataset build  [ ]
- [ ] scripts/build_dataset.py → data/train.jsonl, data/val.jsonl
- [ ] Robust Cantonese/English splitter (last CJK→Latin boundary + validation)
- [ ] Target policy per QUESTIONS_FOR_USER.md (CSV gold > TSV suggestion >
      identity pair if original avg score >= 4)
- [ ] Prompt template: chat format, optional Cantonese line, fraction without it
- [ ] Stats + spot-check file for user review

### P3 — tinygrad inference of Qwen3-4B (baseline)  [ ]
- [ ] Download Qwen/Qwen3-4B (~8 GB) to models/
- [ ] Model port to tinygrad (QK-norm, tied embeddings, GQA), greedy + sampling
- [ ] Baseline run on eval sentences → out/baseline_eval.txt

### P4 — LoRA fine-tune in tinygrad  [ ]
- [ ] scripts/train_lora.py: LoRA on attention+MLP projections, AdamW, masked loss
      (completion-only), checkpointing, time-bounded steps (<=30 min/run)
- [ ] First training run + val loss curve

### P5 — Evaluation  [ ]
- [ ] scripts/eval_synth.py — ~12-14 synthetic sentences (typo/grammar/
      translationese/already-correct/with+without Cantonese)
- [ ] Comparison table baseline vs fine-tuned in out/

### P6 — Deliverable + docs  [ ]
- [ ] correct.py CLI (tinygrad inference, loads LoRA adapter if present)
- [ ] README.md (usage, training, data policy)
- [ ] JOURNAL.md kept current; commits per chunk
- [ ] GGUF export: optional only (Q3); skip unless time permits

## Key risks / fallbacks
- tinygrad Qwen3 inference speed on M2 → verify early; fallback: smaller context,
  bf16→int8 weights for inference if needed
- Training OOM/slow → LoRA rank 16, seq len 512, grad accumulation; 96 GB unified
  memory is plenty for 4B bf16 + adapters
- If LoRA quality poor → higher rank / more epochs / more data; full-FT last layers
  as last resort
