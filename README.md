# qwen3-4b-english-correction

(This is purely vibe coded, with sparse directions and steering from me. I do
not take any moral responsibility for anything in this repo.)

Fine-tunes of **Qwen3-1.7B** and **Qwen3-0.6B** that correct badly written English
sentences — the kind produced by Cantonese L1 speakers editing an English
dictionary (translationese, grammar slips, Chinglish collocations). Full
fine-tuning with **MLX** on an M2 Max Mac Studio. (The repo name and the 4B
artifacts are historical — see the Hugging Face repo; the 4B is not part of the
current acceptance criteria.)

Input: an English sentence, plus an *optional* Cantonese reference meaning.
Output: the corrected sentence — or the original, unchanged, if it is already fine.

```
EN: He go to school yesterday.            → He went to school yesterday.
EN: Open the light please.   ZH: 請開燈。  → Please turn on the light.
EN: The quick brown fox jumps over the lazy dog. → (unchanged)
```

Models are published on Hugging Face:
`sydneyfong/qwen3-4b-english-correction` → `qwen3-1.7b-full/`, `qwen3-0.6b-full/`
(HF format) and `qwen3-1.7b-english-correction-{f16,q8_0}.gguf`,
`qwen3-0.6b-english-correction-{f16,q8_0}.gguf` (llama.cpp).

## Use it (Makefile)

```bash
make interactive            # REPL corrector (downloads/trains everything needed)
make eval                   # 14 synthetic cases -> out/eval_report_qwen3-0.6b.txt
make train                  # 2-stage curriculum + fuse (resumes after crashes)

make train BASE_MODEL=Qwen/Qwen3-1.7B   # same pipeline, other size
```

`BASE_MODEL` is the only variable (any Qwen3 HF path). `make train` sets
`AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` (macOS watchdog mitigation, see JOURNAL.md)
and resumes from the last checkpoint if a stage crashes. To force a fresh
retrain: `rm -rf out/mlx-<name>-s1 out/mlx-<name>-s2 models/<name>-english-correction-mlx`.

Manual equivalents:

```bash
.venv/bin/python correct.py "He go to school yesterday."
.venv/bin/python correct.py "This thingy is cute." --cantonese "呢嚿嘢好可愛。"
```

## Pipeline

1. **Data** (`scripts/build_dataset.py` → `data/train.jsonl` 66,167 pairs +
   `data/val.jsonl` 1,374): pairs (bad English → corrected English) from the
   three teacher TSV groups + the 9 human-reviewed CSVs (gold). Each entry
   carries the Cantonese reference meaning. 105,101 raw rows → 35,750 unique
   entries → 66,167 pairs (100% of valid raw_data; acceptance criterion is ≥50%).
2. **Data prep** (`scripts/prep_mlx_data.py` → `tmp/mlx_data/`): two-stage
   curriculum — stage 1: 63,951 machine-teacher pairs; stage 2: 2,216
   human-reviewed pairs. Rendered as raw Qwen-format **text** (no
   apply_chat_template: the stock Qwen3 template inserts empty <arg_key>…</tool_call> blocks
   into assistant turns, which this project deliberately avoids — think tokens
   are treated as nonexistent). ~15% of examples omit the Cantonese line
   (deterministic) so the model also works without it.
3. **Training** (MLX `mlx_lm lora --fine-tune-type full`, i.e. full fine-tune):
   - stage 1: 8,000 iters, batch 8, lr 1e-5 (1 epoch over stage 1)
   - stage 2: 830 iters, lr 5e-6 (3 epochs over stage 2 human data)
   - LR chosen empirically (1e-5 stable and converging for both sizes; 5e-6
     polish for the small human-data stage).
   - 0.6B: val loss ~1.0 (base ~2.2); 1.7B: val loss ~0.87 (base ~3.2).
   - No watchdog workarounds needed for stage 1; stage 2 ran with
     `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` (see JOURNAL.md — the
     kIOGPUCommandBufferCallbackErrorImpactingInteractivity issue is a macOS
     display-compositing problem, not a framework bug; display-off is the
     reliable mitigation).
4. **Export**: `mlx_lm fuse` → HF-format dir (with a think-free
   `chat_template.jinja`) → llama.cpp `convert_hf_to_gguf.py` → q8_0 via
   `llama-quantize`. Verified greedy-identical behavior across MLX, HF
   transformers+peft (4B legacy) and llama.cpp.

## Eval

14 hand-written synthetic cases (grammar, tense, agreement, articles, Chinglish
collocations, translationese, run-ons, already-correct inputs; with and without
Cantonese reference): **all pass for both 1.7B and 0.6B** (`out/eval_report_*.txt`).
Think tokens: not handled anywhere; the fine-tuned models do not emit them.

## Retrain

```bash
make            # sets up venv, downloads weights, builds data, trains, fuses
.venv/bin/python scripts/prep_mlx_data.py
# stage 1 (per model):
.venv/bin/python -m mlx_lm lora --model models/qwen3-1.7b --train \
  --data tmp/mlx_data/stage1 --fine-tune-type full --batch-size 8 \
  --max-seq-length 512 --learning-rate 1e-5 --iters 8000 \
  --adapter-path out/mlx17_s1
# stage 2 (human data):
.venv/bin/python -m mlx_lm lora --model models/qwen3-1.7b --train \
  --data tmp/mlx_data/stage2 --fine-tune-type full --batch-size 8 \
  --max-seq-length 512 --learning-rate 5e-6 --iters 830 \
  --resume-adapter-file out/mlx17_s1/adapters.safetensors \
  --adapter-path out/mlx17_s2
.venv/bin/python -m mlx_lm fuse --model models/qwen3-1.7b \
  --adapter-path out/mlx17_s2 --save-path models/qwen3-1.7b-english-correction-mlx
```

## Known limitations

- The 1.7B/0.6B base models are hybrid-thinking variants; the fine-tuned models
  answer directly (think behavior is trained out), but *inference on the base
  models with this code will produce think blocks* — by design, not handled.
- Single-sentence correction only; no document-level editing.
- 1 epoch over the data; more epochs / data passes would likely improve style
  coverage further.
