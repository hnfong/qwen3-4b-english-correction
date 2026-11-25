---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: mlx
tags:
- full-finetune
- grammar-correction
- english
- cantonese
- hong-kong
- qwen3
- mlx
- llama.cpp
language:
- en
- yue
---

# Qwen3 english-correction (0.6B + 1.7B, full fine-tune)

**Qwen3-0.6B** and **Qwen3-1.7B**, fully fine-tuned (no LoRA) with **MLX** to
correct badly written English sentences — the kind produced by Cantonese L1
speakers editing an English dictionary (translationese, grammar slips, Chinglish
collocations). Trained on the full derived dataset: **66,167 pairs** from the
words.hk Cantonese-English dictionary (105,101 raw teacher/human rows → 35,750
unique entries), including 2,216 human-reviewed pairs as a final curriculum stage.

Input: an English sentence, plus an *optional* Cantonese reference meaning.
Output: the corrected sentence — or the original, unchanged, if it is already fine.
The base models are hybrid-thinking variants; the fine-tuned models answer
directly (think behavior is trained out; this pipeline does not use think tokens).

```
EN: He go to school yesterday.            → He went to school yesterday.
EN: Open the light please.   ZH: 請開燈。  → Please turn on the light.
EN: The quick brown fox jumps over the lazy dog. → (unchanged)
```

## Files

| path | what |
|---|---|
| `qwen3-0.6b-full/`, `qwen3-1.7b-full/` | merged full fine-tunes, HF/MLX format (config + weights + tokenizer + think-free chat template) |
| `qwen3-0.6b-english-correction-{f16,q8_0}.gguf` | llama.cpp GGUFs (0.6B) |
| `qwen3-1.7b-english-correction-{f16,q8_0}.gguf` | llama.cpp GGUFs (1.7B) |
| `data/` | the training data (66,167 train / 1,374 val pairs, samples, stats) |

## Usage — MLX

```python
from mlx_lm import load, generate
model, tok = load("qwen3-1.7b-full")  # this repo, subfolder
SYSTEM = ("You are an English editor. Fix grammar, word choice and phrasing of the "
          "given English sentence. Reply with the corrected sentence only, with no "
          "explanations. If the sentence is already correct, reply with it unchanged.")
prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
          f"<|im_start|>user\nEnglish sentence: He go to school yesterday.<|im_end|>\n"
          f"<|im_start|>assistant\n")
print(generate(model, tok, prompt=prompt, max_tokens=80, sampler=make_sampler(temp=0)))
```

## Usage — llama.cpp

```bash
SYST='You are an English editor. Fix grammar, word choice and phrasing of the given English sentence. Reply with the corrected sentence only, with no explanations. If the sentence is already correct, reply with it unchanged.'
llama-completion -m qwen3-1.7b-english-correction-q8_0.gguf -e --temp 0 -n 60 \
  -p "<|im_start|>system\n$SYST<|im_end|>\n<|im_start|>user\nEnglish sentence: He go to school yesterday.<|im_end|>\n<|im_start|>assistant\n"
# -> He went to school yesterday.
```

## Training (MLX, full fine-tune, two-stage curriculum)

- stage 1: 63,951 machine-teacher pairs, 1 epoch (8,000 iters, batch 8, lr 1e-5)
- stage 2: 2,216 human-reviewed pairs, 3 epochs (830 iters, lr 5e-6)
- loss on the full sequence (prompt tokens included — the Cantonese reference is
  part of every trainable prompt, serving the Cantonese-understanding goal);
  15.2% of examples omit the Cantonese line (deterministic) for robustness.
- Data rendering is raw text with **no think tokens** (the stock Qwen3 chat
  template inserts empty think blocks into assistant turns — avoided entirely).

## Results (14 hand-written synthetic cases; all pass for both sizes)

Grammar, tense, agreement, articles, Chinglish collocations ("Open the light" →
"turn on the light"), translationese, run-ons — and identity behavior on
already-correct sentences.

## Limitations

- Single-sentence correction; no document-level editing.
- Inference on the *base* (pre-fine-tune) thinking models with this prompt code
  will produce think blocks — by design, not handled.
- 1 training epoch; more passes would likely widen style coverage.
