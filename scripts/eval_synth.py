#!/usr/bin/env python3
"""Synthetic eval: ~14 hand-made sentences, baseline vs fine-tuned model.

Usage:
  .venv/bin/python scripts/eval_synth.py --adapter out/lora_r16a/best
Writes out/eval_report.txt (+ .json).
"""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from tinygrad import Tensor, dtypes
from qwen3 import Qwen3, Config, Tokenizer, build_prompt, user_content

# (en, zh or None, note)
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

def generate(model, tok, en, zh, max_new=60):
    prompt = build_prompt(user_content(en, zh))
    ids = tok.encode(prompt)
    out = model.generate(ids, max_new=max_new, temperature=0.0)
    text = tok.decode(out)
    text = text.split("\n")[0].strip()  # single-sentence replies only
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-4b-instruct")
    ap.add_argument("--adapter", default=None, help="LoRA dir with adapter.safetensors")
    ap.add_argument("--out", default="out/eval_report")
    args = ap.parse_args()

    tok = Tokenizer(args.model)
    cfg = Config.from_hf(args.model)
    results = []

    for tag, lora_r in [("baseline", 0), ("finetuned", 16)]:
        if tag == "finetuned" and not args.adapter:
            print("no adapter given, skipping finetuned"); continue
        model = Qwen3(cfg, lora_r=lora_r)
        model.load_hf(args.model)
        if tag == "finetuned":
            from tinygrad.nn.state import safe_load, load_state_dict
            sd = {k: v.to("METAL") for k, v in safe_load(f"{args.adapter}/adapter.safetensors").items()}
            load_state_dict(model, sd, strict=False)
            model.merge_loras()
            print("merged LoRA adapter")
        for en, zh, note in CASES:
            t0 = time.time()
            out_zh = generate(model, tok, en, zh) if zh else None
            out_no = generate(model, tok, en, None)
            results.append({"case": note, "en": en, "zh": zh, "model": tag,
                            "out_with_zh": out_zh, "out_without_zh": out_no,
                            "sec": round(time.time() - t0, 1)})
            print(f"[{tag}] {en} -> {out_no}", flush=True)

    pathlib.Path("out").mkdir(exist_ok=True)
    out_json = pathlib.Path(str(args.out) + ".json")
    out_json.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    by_case = {}
    for r in results:
        by_case.setdefault(r["case"], {})[r["model"]] = r
    with open(pathlib.Path(str(args.out) + ".txt"), "w", encoding="utf-8") as f:
        for note, models in by_case.items():
            b, ft = models.get("baseline"), models.get("finetuned")
            f.write(f"=== {note} ===\nEN : {b['en'] if b else ft['en']}\n")
            if b and b["zh"]: f.write(f"ZH : {b['zh']}\n")
            if b:
                f.write(f"base          : {b['out_without_zh']}\n")
            if ft:
                if ft["out_with_zh"] is not None:
                    f.write(f"finetuned w/ ZH: {ft['out_with_zh']}\n")
                f.write(f"finetuned w/o ZH: {ft['out_without_zh']}\n")
            f.write("\n")
    print("wrote", out_json)

if __name__ == "__main__":
    main()
