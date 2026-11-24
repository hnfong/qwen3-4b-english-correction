#!/usr/bin/env python3
"""Correct an English sentence (optionally with a Cantonese reference meaning).

Usage:
  .venv/bin/python correct.py "He go to school yesterday."
  .venv/bin/python correct.py "This thingy is cute." --cantonese "呢嚿嘢好可愛。"
  .venv/bin/python correct.py "..." --no-adapter          # base model
  .venv/bin/python correct.py "..." --adapter out/lora_r16a/best
"""
import argparse, pathlib, re, sys
THINK_RE = re.compile(r"^<think>.*?</think>\s*", re.S)  # hybrid-thinking variants (e.g. Qwen3-1.7B)
sys.path.insert(0, str(pathlib.Path(__file__).parent / "scripts"))
from tinygrad import Tensor, dtypes
from qwen3 import Qwen3, Config, Tokenizer, build_prompt, user_content

def main():
    ap = argparse.ArgumentParser(description="English sentence corrector (Qwen3-4B + LoRA, tinygrad)")
    ap.add_argument("sentence", nargs="?", default=None,
                    help="English sentence to correct; omit to enter interactive mode")
    ap.add_argument("--cantonese", default=None, help="optional Cantonese reference meaning")
    ap.add_argument("--model", default="models/qwen3-4b-instruct")
    ap.add_argument("--adapter", default="out/lora_r16a/best")
    ap.add_argument("--no-adapter", action="store_true", help="use the base model")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    tok = Tokenizer(args.model)
    cfg = Config.from_hf(args.model)
    lora_r = 0 if args.no_adapter else 16
    model = Qwen3(cfg, lora_r=lora_r)
    model.load_hf(args.model)
    if not args.no_adapter:
        from tinygrad.nn.state import safe_load, load_state_dict
        sd = {k: v.to("METAL") for k, v in safe_load(f"{args.adapter}/adapter.safetensors").items()}
        load_state_dict(model, sd, strict=False)
        model.merge_loras()

    if args.sentence:
        prompt = build_prompt(user_content(args.sentence, args.cantonese))
        ids = tok.encode(prompt)
        out = model.generate(ids, max_new=args.max_new, temperature=args.temperature)
        text = THINK_RE.sub("", tok.decode(out), count=1).split("\n")[0].strip()
        print(text)
    else:
        def ask(p):
            try: return input(p)
            except EOFError: return None
        while True:
            sentence = ask("English: ")
            if sentence is None: break
            cantonese = ask("Cantonese: ")
            if cantonese is None: break
            if not sentence and not cantonese: break
            prompt = build_prompt(user_content(sentence, cantonese))
            ids = tok.encode(prompt)
            out = model.generate(ids, max_new=args.max_new, temperature=args.temperature)
            text = THINK_RE.sub("", tok.decode(out), count=1).split("\n")[0].strip()
            print("=>", text)



if __name__ == "__main__":
    main()
