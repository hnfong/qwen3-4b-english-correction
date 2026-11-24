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

### P1 — Environment setup  [x]
- [x] venv (python3.12), tinygrad 0.14.0 + tokenizers + numpy installed
- [x] Clone tinygrad to ./tinygrad_ref for reference
- [x] tinygrad's own qwen3 paths unusable (llm module broken in pip release;
      extra/models/llama.py qk_norm not Qwen3-compatible) -> custom impl

### P2 — Dataset build  [x]
- [x] scripts/build_dataset.py → data/train.jsonl (66k), data/val.jsonl (1.4k)
- [x] Robust Cantonese/English splitter (last CJK→Latin boundary + validation)
- [x] Target policy per QUESTIONS_FOR_USER.md (CSV gold > TSV suggestion >
      identity pair if original avg score >= 4)
- [x] Qwen chat template; ~15% of examples without Cantonese line
- [x] data/sample_review.txt + data/stats.json for user review

### P3 — tinygrad inference of Qwen3-4B (baseline)  [x]
- [x] Downloaded Qwen/Qwen3-4B-Instruct-2507 to models/ (user chose no-VL variant)
- [x] Custom port in scripts/qwen3.py; validated EXACTLY vs HF transformers
- [x] Perf work: 0.3→9 tok/s decode (quadratic-recompute fix + TinyJit decode +
      bucketed prefill); baseline outputs in out/eval_report.txt (base column)

### P4 — LoRA fine-tune in tinygrad  [x]
- [x] scripts/train_lora.py + tmp/train_driver.sh (restart loop; macOS GPU
      watchdog mitigations: fp16 base, per-kernel pacing, 1 micro/step —
      see JOURNAL.md for the full saga)
- [x] Training done: ~45 steps, val loss 2.0 -> 0.88, adapter out/lora_r16a/best

### P5 — Evaluation  [x]
- [x] scripts/eval_synth.py — 14 synthetic sentences, with+without Cantonese
- [x] out/eval_report.txt: ALL cases pass (incl. identity on already-correct)

### P6 — Deliverable + docs  [x]
- [x] correct.py CLI (loads adapter, merges LoRA, greedy decode; --no-adapter for base)
- [x] README.md (usage, pipeline, limitations)
- [x] JOURNAL.md current; commits per chunk
- [x] GGUF export: skipped (optional per Q3)

## Key risks / fallbacks
- tinygrad Qwen3 inference speed on M2 → verify early; fallback: smaller context,
  bf16→int8 weights for inference if needed
- Training OOM/slow → LoRA rank 16, seq len 512, grad accumulation; 96 GB unified
  memory is plenty for 4B bf16 + adapters
- If LoRA quality poor → higher rank / more epochs / more data; full-FT last layers
  as last resort
