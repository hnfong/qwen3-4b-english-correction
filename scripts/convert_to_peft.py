#!/usr/bin/env python3
"""Convert the tinygrad LoRA adapter to HF PEFT format.

tinygrad names:  blk.{i}.attn.{q,k,v,o}.lora_a/lora_b, blk.{i}.mlp.{gate,up,down}.lora_a/lora_b
  shapes: lora_a (r, in), lora_b (out, r)  — same as PEFT's lora_A/lora_B weights.
PEFT names:     base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight
                base_model.model.model.layers.{i}.mlp.{gate,up,down}_proj.lora_{A,B}.weight

Usage: .venv/bin/python scripts/convert_to_peft.py --adapter out/lora_r16a/best --out hf_repo
"""
import argparse, json, pathlib
from tinygrad.nn.state import safe_load

MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
       "gate": "gate_proj", "up": "up_proj", "down": "down_proj"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="out/lora_r16a/best")
    ap.add_argument("--out", default="hf_repo")
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()

    sd = safe_load(f"{args.adapter}/adapter.safetensors")
    out = {}
    for k, t in sd.items():
        # k like: blk.12.attn.q.lora_a  /  blk.3.mlp.up.lora_b
        parts = k.split(".")
        i, kind, mod, ab = parts[1], parts[2], parts[3], parts[4]
        hf_mod = MAP[mod]
        sub = "self_attn" if kind == "attn" else "mlp"
        AB = "A" if ab == "lora_a" else "B"
        out[f"base_model.model.model.layers.{i}.{sub}.{hf_mod}.lora_{AB}.weight"] = t

    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)
    from tinygrad.nn.state import safe_save
    safe_save(out, outdir / "adapter_model.safetensors")
    cfg = {
        "alpha_pattern": {}, "auto_mapping": None,
        "base_model_name_or_path": args.base,
        "bias": "none", "fan_in_fan_out": False, "inference_mode": True,
        "init_lora_weights": True, "layer_replication": None,
        "layers_pattern": None, "layers_to_transform": None, "loftq_config": {},
        "lora_alpha": 32.0, "lora_dropout": 0.0, "megatron_config": None,
        "megatron_core": "megatron.core", "modules_to_save": None,
        "peft_type": "LORA", "r": 16, "rank_pattern": {}, "revision": None,
        "target_modules": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
        "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False,
    }
    (outdir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {len(out)} tensors -> {outdir}/adapter_model.safetensors")

if __name__ == "__main__":
    main()
