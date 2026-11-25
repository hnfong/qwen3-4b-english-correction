#!/usr/bin/env python3
"""Prepare MLX training data (raw 'text' format) from data/train.jsonl.

Two-stage curriculum per updated AGENTS.md:
  stage1: machine-teacher pairs (tsv-derived: gemma3/llama3/plain + identity)
  stage2: human-reviewed pairs (csv-derived, source == 'human')

text = our own Qwen-format rendering (system/user/assistant, im_start/im_end),
NO apply_chat_template — the Qwen3 template inserts an empty <arg_key>...</tool_call> block
into assistant turns, which we must not have (user directive: think tokens do
not exist). Loss on the full sequence (TextDataset, no masking) — prompt tokens
include the Cantonese reference, directly serving the 'improve basic Cantonese
understanding' goal.

~15% of examples (deterministic, hash-based) omit the Cantonese line for
robustness without it.

Output: tmp/mlx_data/stage1/{train,valid}.jsonl, tmp/mlx_data/stage2/{train,valid}.jsonl
"""
import hashlib, json, pathlib

SYSTEM = ("You are an English editor. Fix grammar, word choice and phrasing of the "
          "given English sentence. Reply with the corrected sentence only, with no "
          "explanations. If the sentence is already correct, reply with it unchanged.")

def user_content(en, zh):
    return (f"Cantonese reference: {zh}\n" if zh else "") + f"English sentence: {en}"

def omit_zh(en):
    return int(hashlib.md5(en.encode()).hexdigest(), 16) % 100 < 15

IM_END = "<|" + "im_end" + "|>"

def render(ex):
    zh = None if (ex.get("zh") is None or omit_zh(ex["en"])) else ex["zh"]
    text = (f"<|im_start|>system\n{SYSTEM}{IM_END}\n"
            f"<|im_start|>user\n{user_content(ex['en'], zh)}{IM_END}\n"
            f"<|im_start|>assistant\n{ex['target']}{IM_END}")
    return {"text": text}

def main():
    out = pathlib.Path("tmp/mlx_data")
    (out / "stage1").mkdir(parents=True, exist_ok=True)
    (out / "stage2").mkdir(parents=True, exist_ok=True)
    train = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
    valid = [json.loads(l) for l in open("data/val.jsonl", encoding="utf-8")]
    s1 = [render(e) for e in train if e["source"] != "human"]
    s2 = [render(e) for e in train if e["source"] == "human"]
    for name, rows in [("stage1/train", s1), ("stage2/train", s2), ("stage1/valid", [render(e) for e in valid]), ("stage2/valid", [render(e) for e in valid])]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}.jsonl: {len(rows)}")
    n_zh1 = sum(1 for r in s1 if "Cantonese reference" in r["text"])
    print(f"stage1 with-Cantonese: {n_zh1}/{len(s1)} ({n_zh1/len(s1):.1%})")

if __name__ == "__main__":
    main()
