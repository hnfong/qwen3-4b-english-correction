#!/bin/bash
# Download Qwen/Qwen3-4B-Instruct-2507 (~8 GB) from Hugging Face into models/.
set -e
DIR=models/qwen3-4b-instruct
mkdir -p $DIR && cd $DIR
BASE=https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main
for f in config.json generation_config.json merges.txt model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json; do
  [ -s "$f" ] || curl -sL -o "$f" "$BASE/$f"
done
for i in 1 2 3; do
  f="model-0000$i-of-00003.safetensors"
  [ -s "$f" ] || curl -sL --retry 3 -C - -o "$f" "$BASE/$f"
done
echo "model ready: $(du -sh . | cut -f1)"
