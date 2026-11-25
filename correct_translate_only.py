#!/usr/bin/env python3
"""Correct an English sentence (optionally with a Cantonese reference meaning).

MLX pipeline. Default model: the full fine-tuned Qwen3-0.6B
(models/qwen3-0.6b-english-correction-mlx, produced by mlx_lm fuse).

Usage:
  .venv/bin/python correct.py "He go to school yesterday."
  .venv/bin/python correct.py "This thingy is cute." --cantonese "呢嚿嘢好可愛。"
  .venv/bin/python correct.py                    # interactive mode
"""
import argparse, pathlib, sys

from mlx_lm.sample_utils import make_sampler

SYSTEM = ("You are an English editor. Fix grammar, word choice and phrasing of the "
          "given English sentence. Reply with the corrected sentence only, with no "
          "explanations. If the sentence is already correct, reply with it unchanged.")

def build_prompt(en: str, zh: str | None) -> str:
    user = (f"Cantonese reference: {zh}\n" if zh else "") + f"English sentence: {en}"
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")

def main():
    ap = argparse.ArgumentParser(description="English sentence corrector (MLX)")
    ap.add_argument("sentence", nargs="?", default=None,
                    help="English sentence to correct; omit to enter interactive mode")
    ap.add_argument("--cantonese", default=None, help="optional Cantonese reference meaning")
    ap.add_argument("--model", default="models/qwen3-0.6b-english-correction-mlx")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    from mlx_lm import load, generate
    model, tok = load(args.model)

    def correct(en: str, zh: str | None, temp: float) -> str:
        text = generate(model, tok, prompt=build_prompt(en, zh),
                        max_tokens=args.max_tokens,
                        sampler=make_sampler(temp=temp))
        return text.strip().split("\n")[0].strip()

    def ask(p):
        try: return input(p)
        except EOFError: return None

    while True:
        zh = ask("Cantonese: ")
        en = zh
        if not zh: break
        result = correct(en, zh, args.temperature)
        print("=>", result)

        if args.temperature > 0:
            # Do it again, sometimes it makes the result better.
            print("=>", correct(result, zh, args.temperature))
            print("=>", correct(result, zh, args.temperature))
            print("=>", correct(result, zh, args.temperature))

if __name__ == "__main__":
    main()
