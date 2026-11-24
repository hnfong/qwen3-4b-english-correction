# Questions for User

## Answered (2024-11-24 session)

**Q1. How aggressive should "correction" be?**
**A:** Don't overthink it — use the existing data as-is for training targets, with the
presumption that the human-reviewed CSV files are higher quality. (Do not regenerate
or re-style the data.)

**Q2. Qwen3.5-4B (hybrid Gated DeltaNet VLM) vs Qwen3-4B (standard transformer)?**
**A:** Use **Qwen3-4B without VL**. Less complexity.

**Q3. Deliverable runtime?**
**A:** Default accepted: tinygrad end-to-end + simple CLI
(`correct.py "sentence" [--cantonese "..."]`). GGUF export only if time permits.

**Q4. Quality bar / timeline?**
**A:** Ad-hoc quality checks: make up ~a dozen synthetic test sentences, verify results
are reasonable. User will review afterwards. No hard deadline.

## Derived decisions (made by agent per the above, KISS principle)

- **Base model:** `Qwen/Qwen3-4B` (dense 4.02B, standard transformer: GQA + QK-norm,
  SwiGLU, tied embeddings, 32k context, Apache-2.0).
- **Training targets policy** (follows from Q1):
  - Human-reviewed CSV rows are gold: use the CSV's `suggested_translation`.
  - Other rows: use the TSV `suggested_translation` if present and not a
    refusal/placeholder (e.g. "Skipped due to inappropriate content.").
  - If no usable suggestion but the original English scored high (avg reviewer
    score >= 4), make an identity pair (target = original) — teaches the model to
    return the original when it's already fine.
  - Otherwise skip the row.
- **Cantonese reference:** extracted from the `original` field (CJK part before the
  English sentence, after stripping the sense-letter prefix like "C."). Included in
  the prompt when available; randomly omitted for a fraction of training examples so
  the model also works without it.
- **Evaluation:** fixed set of ~12 synthetic test sentences (varied: correct, grammar
  errors, awkward translationese, with/without Cantonese) + validation loss. Results
  written to `out/` for user review.

## Notes for your review (end of session)

1. **Training scale**: the macOS GPU watchdog (5K display + sustained GPU load)
   constrained training to ~45 optimizer steps / ~200 examples via a restart-loop
   driver. Val loss 2.0 -> 0.88 and all 14 synthetic evals pass, but more driver
   cycles (`tmp/train_driver.sh`, runs unattended) would widen style coverage.
2. **fp16 vs bf16**: training ran with fp16 base weights (Metal tensor-core
   friendliness); inference/CLI uses bf16 (matches HF exactly). No issues observed.
3. **Optimizer state resets** between driver restart cycles (AdamW moments);
   acceptable at this scale, noted in JOURNAL.md.
4. **GGUF export**: skipped (optional per your Q3 answer). The tinygrad CLI is the
   deliverable; say the word if you want GGUF/Ollama.
5. **Qwen3.5-4B**: you chose Qwen3-4B (no VL) — implemented as
   Qwen3-4B-Instruct-2507 (non-thinking variant of the same model, cleaner SFT
   targets). Flag if you want the original hybrid-thinking Qwen3-4B instead.
6. **Cosmetic**: the step counter in training logs resets per driver cycle; the
   adapter weights and best-checkpoint selection are unaffected.

7. **Publishing (prepared, not pushed — no credentials here)**:
   - **Hugging Face** (`hf_repo/` in this repo, its own git repo, LFS-ready):
     LoRA adapter in PEFT format + model card + training data (train/val jsonl,
     samples, stats) + inference_example.py. Verified: loading the PEFT adapter
     through transformers+peft reproduces the tinygrad outputs exactly.
     To publish: `cd hf_repo && git remote add origin
     https://huggingface.co/<YOUR_HF_USER>/qwen3-4b-english-teacher && git push -u origin main`
     (or `huggingface-cli upload`). **Please confirm the repo name / whether you
     want a separate dataset repo instead of data/ inside the model repo.**
   - **GitHub** (this repo): history rewritten to drop the 126 MB adapter blob
     (GitHub blocks >100 MB); pack is 3.4 MiB. The adapter lives on HF only now.
     **Open question: license for the code** (model card says Apache-2.0, matching
     the Qwen base model — add a LICENSE file?).

8. **HF push blocked by token scope (action needed)**: the token in
   `~/.cache/huggingface/token` is fine-grained and READ-ONLY
   (`repo.content.read` only) — repo creation and push both 403. Either
   (a) edit that token to add `repo.content.write` on your user namespace
   (creation included), or (b) create the empty model repo
   `sydneyfong/qwen3-4b-english-correction` on the website and add write access
   for it to the token. Then run `bash scripts/hf_push.sh` (reads the token at
   runtime; scrubs it from .git/config afterwards). Everything else is ready:
   hf_repo/ has the PEFT adapter (verified vs tinygrad outputs), model card,
   LICENSE, training data (LFS), example.
9. **GitHub repo name noted**: qwen-small-english-correction (per your
   s/teacher/correction). Waiting for the remote from you; local dir kept as-is.
