#!/usr/bin/env python3
"""Merge the LoRA adapter into the base model (tinygrad) and export as an
HF-format directory (config/tokenizer/safetensors) ready for llama.cpp's
convert_hf_to_gguf.py.

Usage: .venv/bin/python scripts/export_merged.py \
         --adapter out/lora_r16a/best --out models/qwen3-4b-english-correction-hf
"""
import argparse, json, pathlib, shutil, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from tinygrad import Tensor, nn, Device
from tinygrad.nn.state import safe_load, load_state_dict, safe_save
from qwen3 import Qwen3, Config

# tinygrad attr path -> HF safetensors key
def hf_key(attr: str) -> str:
    parts = attr.split(".")
    if parts[0] == "emb": return "model.embed_tokens.weight"
    if parts[0] == "norm": return "model.norm.weight"
    # blk.{i}.attn.{q,k,v,o}.w | blk.{i}.attn.{q,k}_norm.weight | blk.{i}.mlp.{gate,up,down}.w | blk.{i}.norm{1,2}.weight
    i, kind, mod = parts[1], parts[2], parts[3]
    if kind == "attn":
        if mod in ("q", "k", "v", "o"): return f"model.layers.{i}.self_attn.{mod}_proj.weight"
        if mod in ("q_norm", "k_norm"): return f"model.layers.{i}.self_attn.{mod}.weight"
        raise ValueError(attr)
    if kind == "mlp": return f"model.layers.{i}.mlp.{mod}_proj.weight"
    if kind == "norm1": return f"model.layers.{i}.input_layernorm.weight"
    if kind == "norm2": return f"model.layers.{i}.post_attention_layernorm.weight"
    raise ValueError(attr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-4b-instruct")
    ap.add_argument("--adapter", default="out/lora_r16a/best")
    ap.add_argument("--out", default="models/qwen3-4b-english-correction-hf")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading base + adapter + merging...", flush=True)
    model = Qwen3(Config.from_hf(args.model), lora_r=16)
    model.load_hf(args.model)
    sd = {k: v.to(Device.DEFAULT) for k, v in safe_load(f"{args.adapter}/adapter.safetensors").items()}
    load_state_dict(model, sd, strict=False)
    model.merge_loras()  # folds scale*B@A into base weights, in place

    print("exporting HF-format state dict...", flush=True)
    from tinygrad.nn.state import get_state_dict
    hf = {hf_key(k): t.cast("bfloat16").contiguous() for k, t in get_state_dict(model).items()
          if "lora" not in k and k not in ("cos", "sin", "caches")}
    assert len(hf) == model.cfg.n_layers * 11 + 2, f"unexpected tensor count {len(hf)}"  # 11/layer + emb + norm
    safe_save(hf, out / "model.safetensors")

    for f in ["config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json", "merges.txt"]:
        shutil.copy(pathlib.Path(args.model) / f, out / f)
    meta = {"base": args.model, "adapter": args.adapter,
            "merge": "w += (alpha/r) * B@A folded in fp32, cast bf16", "framework": "tinygrad"}
    (out / "merge_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"merged model at {out}")

if __name__ == "__main__":
    main()
