# Qwen3 english-correction — MLX full fine-tune pipeline
#
# One variable: BASE_MODEL (any Qwen3 HF path). Targets: train | eval | interactive.
#
#   make train        # 2-stage curriculum (machine data -> human data) + fuse
#   make eval         # 14 synthetic cases -> out/eval_report_<model>.txt
#   make interactive  # REPL corrector (omit the sentence arg)
#
# Notes:
# - eval/interactive build the model first if it does not exist yet.
# - train sets AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 (macOS watchdog mitigation for
#   sustained Metal load, see JOURNAL.md / ml-explore/mlx#3267). If a stage
#   crashes, rerun `make train` — stages resume from their last checkpoint.
# - To force a fresh retrain: rm -rf out/mlx-$(NAME)-s1 out/mlx-$(NAME)-s2 $(FUSED)

BASE_MODEL ?= Qwen/Qwen3-4B

NAME      := $(shell echo $(notdir $(BASE_MODEL)) | tr 'A-Z' 'a-z')
MODEL_DIR := models/$(NAME)
FUSED     := models/$(NAME)-english-correction-mlx
ADAPTER   := out/mlx-$(NAME)
PY        := .venv/bin/python
WATCHDOG  := AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1

.DEFAULT_GOAL := train
.PHONY: train eval interactive

$(PY):
	python3.12 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q mlx mlx-lm

$(MODEL_DIR)/config.json: $(PY)
	.venv/bin/hf download $(BASE_MODEL) --local-dir $(MODEL_DIR)

data/train.jsonl: $(wildcard raw_data/*.tsv) $(wildcard raw_data/*.csv) scripts/build_dataset.py $(PY)
	$(PY) scripts/build_dataset.py

tmp/mlx_data/stage1/train.jsonl: data/train.jsonl scripts/prep_mlx_data.py
	$(PY) scripts/prep_mlx_data.py

$(ADAPTER)-s1/.done: tmp/mlx_data/stage1/train.jsonl $(MODEL_DIR)/config.json
	@if [ -f $(ADAPTER)-s1/adapters.safetensors ]; then \
	  RESUME="--resume-adapter-file $(ADAPTER)-s1/adapters.safetensors"; else RESUME=; fi; \
	$(WATCHDOG) $(PY) -m mlx_lm lora --model $(MODEL_DIR) --train \
	  --data tmp/mlx_data/stage1 --fine-tune-type full --batch-size 8 \
	  --max-seq-length 512 --learning-rate 1e-5 --iters 8000 \
	  --steps-per-report 100 --steps-per-eval 250 --save-every 500 \
	  --adapter-path $(ADAPTER)-s1 $$RESUME
	@touch $@

$(ADAPTER)-s2/.done: $(ADAPTER)-s1/.done tmp/mlx_data/stage2/train.jsonl
	$(WATCHDOG) $(PY) -m mlx_lm lora --model $(MODEL_DIR) --train \
	  --data tmp/mlx_data/stage2 --fine-tune-type full --batch-size 8 \
	  --max-seq-length 512 --learning-rate 5e-6 --iters 830 \
	  --steps-per-report 100 --steps-per-eval 200 --save-every 200 \
	  --resume-adapter-file $(ADAPTER)-s1/adapters.safetensors \
	  --adapter-path $(ADAPTER)-s2
	@touch $@

$(FUSED)/.done: $(ADAPTER)-s2/.done scripts/chat_template.jinja
	$(PY) -m mlx_lm fuse --model $(MODEL_DIR) --adapter-path $(ADAPTER)-s2 --save-path $(FUSED)
	cp scripts/chat_template.jinja $(FUSED)/chat_template.jinja
	@touch $@

train: $(FUSED)/.done
	@echo "model ready: $(FUSED)"

eval: $(FUSED)/.done
	$(PY) scripts/eval_synth.py --model $(FUSED) --out out/eval_report_$(NAME)

interactive: $(FUSED)/.done
	$(PY) correct.py --model $(FUSED)

# llama.cpp binaries (for quantize + smoke test)
LLAMA_DIR ?= $(HOME)/projects/llama.gguf

tmp/llama.cpp/.git:
	git clone -q --depth 1 https://github.com/ggml-org/llama.cpp $@

tmp/torch_venv/bin/python:
	python3.12 -m venv tmp/torch_venv
	tmp/torch_venv/bin/pip install -q torch transformers gguf sentencepiece

models/$(NAME)-english-correction-f16.gguf: $(FUSED)/.done tmp/llama.cpp/.git tmp/torch_venv/bin/python
	tmp/torch_venv/bin/python tmp/llama.cpp/convert_hf_to_gguf.py $(FUSED) --outfile $@ --outtype f16

# quantize + smoke test: expect a corrected sentence, no think blocks
models/$(NAME)-english-correction-q8_0.gguf: models/$(NAME)-english-correction-f16.gguf
	$(LLAMA_DIR)/llama-quantize $< $@ q8_0
	@echo "--- smoke test ---"
	@$(LLAMA_DIR)/llama-completion -m $@ -f scripts/smoke_prompt.txt --temp 0 -n 60 --no-display-prompt < /dev/null 2>/dev/null

gguf: models/$(NAME)-english-correction-q8_0.gguf models/$(NAME)-english-correction-f16.gguf
