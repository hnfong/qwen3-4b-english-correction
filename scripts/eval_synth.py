#!/usr/bin/env python3
"""Synthetic eval: 14 hand-made sentences through an MLX fine-tuned model.

Usage:
  .venv/bin/python scripts/eval_synth.py --model models/qwen3-0.6b-english-correction-mlx
Writes out/<out>.txt (+ .json).
"""
import argparse, json, pathlib, sys

from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from correct import build_prompt

CASES = [
    ("He go to school yesterday.", "佢琴日返學。", "tense+agreement"),
    ("She don't like coffee.", None, "agreement"),
    ("I have saw that movie last week.", None, "tense"),
    ("I bought a umbrella because it was raining.", None, "article"),
    ("I am living in Hong Kong since 2010.", "我二零一零年開始住喺香港。", "tense+preposition"),
    ("This thingy is cute.", "呢嚿嘢好可愛。", "translationese (from data)"),
    ("Open the light please.", "請開燈。", "Chinglish collocation"),
    ("I very like this song.", None, "Chinglish intensifier"),
    ("There is many people in the room.", None, "there is/are"),
    ("I recieve your letter yesterday.", None, "spelling+tense"),
    ("I went home I ate dinner and then I sleep.", None, "run-on"),
    ("The quick brown fox jumps over the lazy dog.", None, "already correct"),
    ("He teaches me very patiently.", "佢好耐心噉教我。", "already correct (from data)"),
    ("Hong Kong is a nice city and the food is also very nice and the people are nice too.", None, "repetition"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--out", default="out/eval_report")
    ap.add_argument("--max-tokens", type=int, default=80)
    args = ap.parse_args()

    from mlx_lm import load, generate
    model, tok = load(args.model, adapter_path=args.adapter_path)

    def gen(en, zh):
        text = generate(model, tok, prompt=build_prompt(en, zh),
                        max_tokens=args.max_tokens, sampler=make_sampler(temp=0.0))
        return text.strip().split("\n")[0].strip()

    results = []
    for en, zh, note in CASES:
        out_zh = gen(en, zh) if zh else None
        out_no = gen(en, None)
        results.append({"case": note, "en": en, "zh": zh, "out_with_zh": out_zh,
                        "out_without_zh": out_no})
        print(f"[ft] {en} -> {out_no}", flush=True)

    pathlib.Path("out").mkdir(exist_ok=True)
    out_json = pathlib.Path(str(args.out) + ".json")
    out_json.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    with open(pathlib.Path(str(args.out) + ".txt"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"=== {r['case']} ===\nEN : {r['en']}\n")
            if r["zh"]: f.write(f"ZH : {r['zh']}\n")
            if r["out_with_zh"] is not None: f.write(f"ft w/ ZH : {r['out_with_zh']}\n")
            f.write(f"ft w/o ZH: {r['out_without_zh']}\n\n")
    print("wrote", out_json)

if __name__ == "__main__":
    main()
