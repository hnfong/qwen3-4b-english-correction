# JOURNAL

Distilled context for any successor agent. Read this + QUESTIONS_FOR_USER.md +
PLAN_01_overall.md first. Newest entries at the top.

## 2024-11-24 (evening) — P4/P5: training works, eval passes

**Done**
- Fixed two dataset bugs found during training smoke tests:
  1. Label off-by-one: labels must be NEXT-token aligned (labels[i] = tokens[i+1],
     -100 elsewhere). Before: loss ~20 (nonsense); after: base CE ~2-3.8. ✓
  2. `build_batches` tuple-unpack bug: `for l, _ in batch` grabbed TOKENS as
     labels (tuples are (toks, labels)). One-line fix.
- Metal training fragility saga (the big time sink; see below).
- Training runs completed via a restart-loop driver (tmp/train_driver.sh):
  ~45 optimizer steps, 960-example subset, LoRA r=16 a=32 lr 1.5e-4, seqlen 160,
  batch 4. Val loss 0.88 (base ~2.0+). Adapter: out/lora_r16a/best.
- **Synthetic eval: ALL 14 cases pass** (out/eval_report.txt): grammar, tense,
  agreement, articles, Chinglish collocations ("Open the light" → "turn on"),
  translationese, run-ons — AND identity behavior on already-correct sentences
  (returns them unchanged). Works with and without Cantonese reference.

**The macOS GPU watchdog problem (critical knowledge)**
- Symptom: `RuntimeError: Impacting Interactivity
  (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)` killing training
  processes after ~1-4 steps. This Mac drives a 5K display; sustained GPU
  saturation (>~60-90 s continuous) makes macOS kill command buffers.
- Root causes stacked: (1) tinygrad 0.14 autograd's backward kernels for this
  model run at ~1.6 TFLOPS (vs 20+ forward) — a (4,256) backward is ~40 s of
  continuous GPU; (2) multi-micro accumulation steps push stretches >90 s.
- Mitigations that WORK (all in scripts/train_lora.py):
  - fp16 base weights (TRAIN_FP16=1 default)
  - 5 ms sleep per Metal kernel submission (PACE_S env, monkeypatches
    MetalProgram.__call__)
  - ONE micro-batch per optimizer step (accum=1), seqlen 160, batch 4
  - full Device synchronize after each micro-batch and each opt.step
  - **restart-loop driver**: each process trains ~4-5 steps then exits; adapter
    resumed from out/lora_RUN/last. Fresh process every ~4 min dodges the
    cumulative kill. Compiled kernels ARE disk-cached across processes, so
    restarts cost ~10 s not ~100 s.
- What did NOT work: BEAM=2 (search itself saturates GPU → killed), pacing
  alone without restarts, GPU grad accumulation (any 2-micro step died),
  DEBUG=2 (luck, not a fix).
- Optimizer state does NOT survive restarts (AdamW moments reset) — acceptable.

**Quirk**: step counter resets per driver process (cosmetic; adapter weights and
best-checkpoint selection by val loss are correct).

**Next**: finish eval report rerun (scripts/eval_synth.py report writer fixed),
P6 CLI deliverable (correct.py drafted), README, final commits.

## 2024-11-24 (night) — Clean-clone reproducibility verified

- Added scripts/setup.sh (venv + deps + HF weight download + dataset build) and
  scripts/download_model.sh; committed the trained adapter (out/lora_r16a/best,
  126 MB, force-added past the out/ ignore).
- Tested an actual `git clone` into tmp/clean_clone: setup.sh ran end-to-end
  (downloaded 7.5 GB from HF), correct.py produced the right correction (21.9 s),
  and rebuilding the dataset from raw_data gave a BYTE-IDENTICAL train.jsonl
  (md5 6cebde93...). So: runtime + data pipeline fully reproducible; retraining
  is seeded but not bit-reproducible (driver restart timing shifts the shuffle
  sequence) — quality-wise it converges to the same behavior.
