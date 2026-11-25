#!/bin/bash
# Finish the 1.7B pipeline with the display asleep (watchdog mitigation,
# see JOURNAL + ml-explore/mlx#3267): stage2 -> fuse -> eval -> GGUF -> upload.
set -e
cd /Volumes/StudioData/projects/qwen-small-english-teacher
echo "=== stage2 (human data, 3 epochs) ==="
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 .venv/bin/python -m mlx_lm lora --model models/qwen3-1.7b --train \
  --data tmp/mlx_data/stage2 --fine-tune-type full --batch-size 8 \
  --max-seq-length 512 --learning-rate 5e-6 --iters 830 \
  --steps-per-report 100 --steps-per-eval 200 --save-every 200 \
  --resume-adapter-file out/mlx17_s1/adapters.safetensors \
  --adapter-path out/mlx17_s2
echo "=== fuse ==="
rm -rf models/qwen3-1.7b-english-correction-mlx
.venv/bin/python -m mlx_lm fuse --model models/qwen3-1.7b \
  --adapter-path out/mlx17_s2 --save-path models/qwen3-1.7b-english-correction-mlx
echo "=== eval ==="
.venv/bin/python scripts/eval_synth.py --model models/qwen3-1.7b-english-correction-mlx \
  --out out/eval_report_17b_mlx_full
echo "=== gguf ==="
tmp/torch_venv/bin/python tmp/llama.cpp/convert_hf_to_gguf.py \
  models/qwen3-1.7b-english-correction-mlx \
  --outfile models/qwen3-1.7b-english-correction-f16.gguf --outtype f16
~/projects/llama.gguf/llama-quantize models/qwen3-1.7b-english-correction-f16.gguf \
  models/qwen3-1.7b-english-correction-q8_0.gguf q8_0
echo "=== llama.cpp verify ==="
SYST='You are an English editor. Fix grammar, word choice and phrasing of the given English sentence. Reply with the corrected sentence only, with no explanations. If the sentence is already correct, reply with it unchanged.'
~/projects/llama.gguf/llama-completion -m models/qwen3-1.7b-english-correction-q8_0.gguf \
  -e --temp 0 -n 60 --no-display-prompt \
  -p "<|im_start|>system\n$SYST<|im_end|>\n<|im_start|>user\nEnglish sentence: She don't like coffee.<|im_end|>\n<|im_start|>assistant\n" | head -1
echo "ALL_DONE"
