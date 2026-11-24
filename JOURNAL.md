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

## 2024-11-25 — Publishing prep (HF + GitHub), correct.py REPL

- correct.py: user added an interactive REPL (no sentence arg -> loop until
  empty-empty); wired argparse to nargs="?" + EOF guard; committed.
- **HF repo at hf_repo/** (own git repo, LFS): LoRA adapter CONVERTED to PEFT
  format (scripts/convert_to_peft.py; 504 tensors, standard
  base_model.model.model.layers.*.{self_attn,mlp}.*_proj.lora_{A,B}.weight keys)
  + model card + training data (train/val jsonl in LFS, samples, stats) +
  inference_example.py. VERIFIED: transformers+peft+adapter reproduces tinygrad
  outputs exactly (4/4 test sentences).
- **GitHub repo (this one)**: history rewritten with filter-branch to drop the
  126 MB adapter blob (GitHub blocks >100 MB); pack 3.4 MiB. Also untracked
  data/*.jsonl (they live in hf_repo now) + a stray .pyc.
- **FOOTGUN**: filter-branch's final read-tree -u DELETED the working-tree
  adapter (it became untracked). Recovered from tmp/clean_clone's checkout
  (identical file). Lesson: before filter-branch, back up working-tree files
  that the index-filter removes.
- tmp/clean_clone is stale (pre-rewrite history); tmp/gh_check is the current
  GitHub-clone smoke test (passes: setup.sh + correct.py with adapter supplied).
- Push instructions + open questions (license, HF repo name) in
  QUESTIONS_FOR_USER.md #7.

## 2024-11-25 (later) — license, rename, HF push blocked on token scope

- LICENSE: canonical Apache-2.0 text downloaded from apache.org into repo root
  + hf_repo/. User confirmed license choice.
- Rename per user: qwen3-4b-english-correction (HF), qwen-small-english-correction
  (GitHub, remote pending). Model card + README updated.
- correct.py: user's interactive-REPL change committed with argparse wiring
  (nargs="?") + EOF guard.
- **HF push attempted, blocked**: token is fine-grained READ-ONLY
  (repo.content.read). create_repo -> 403; git ls-remote -> "Repository not
  found". User must add repo.content.write to the token (or pre-create the repo
  + grant write), then `bash scripts/hf_push.sh` (runtime-token push, scrubs
  token from .git/config after).
- PEFT conversion verified again end-to-end; hf_repo commit history clean (2
  commits, LFS objects stored).
