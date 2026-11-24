#!/usr/bin/env python3
"""LoRA fine-tune Qwen3 on the correction dataset (tinygrad).

Data model (v2, no legacy compat):
  - tmp/train_shuffled.jsonl is THE canonical training order (generated once by
    scripts/shuffle_train.py, deterministic).
  - Training consumes it SEQUENTIALLY from --offset, `--batch` examples per
    optimizer step, up to --window examples per process, then saves
    cursor = offset + consumed into the checkpoint meta. The driver restarts
    processes with --offset = last cursor, so restarts never replay data.
  - No in-process shuffling, no example caps, no padding tricks: every batch
    pads to exactly seqlen (one graph shape per run).

Usage: .venv/bin/python scripts/train_lora.py --run R [--offset N] [--window 256]
"""
import argparse, json, os, pathlib, random, sys, time

from tinygrad import Tensor, dtypes, nn, Device
from tinygrad.helpers import Context, TRAINING
from tinygrad.nn.state import safe_save, get_state_dict, safe_load, load_state_dict
import tinygrad.runtime.ops_metal as _om
# macOS GPU watchdog mitigation: with a 5K display attached, sustained GPU
# saturation makes macOS kill command buffers
# (kIOGPUCommandBufferCallbackErrorImpactingInteractivity). A tiny sleep per
# kernel submission creates gaps that keep the compositor responsive.
PACE_S = float(os.environ.get('PACE_S', '0.005'))
FP16 = os.environ.get('TRAIN_FP16', '1') == '1'
if PACE_S > 0:
    _orig_metal_call = _om.MetalProgram.__call__
    def _paced_call(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, **kw):
        r = _orig_metal_call(self, *bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, **kw)
        time.sleep(PACE_S)
        return r
    _om.MetalProgram.__call__ = _paced_call

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qwen3 import Qwen3, Config, Tokenizer, encode_example

def encode_stream(data, tok, seqlen):
    """Tokenize examples; drop those over seqlen. Returns list of (toks, labels)."""
    exs, dropped = [], 0
    for ex in data:
        toks, labels = encode_example(tok, ex)
        if len(toks) > seqlen:
            dropped += 1; continue
        exs.append((toks, labels))
    return exs, dropped

def forward_loss(model, toks_t, labs_t):
    logits = model.forward_train(toks_t)  # (B,T,V) fp32
    return logits.sparse_categorical_crossentropy(labs_t, ignore_index=-100)

def val_loss(model, batches, max_batches=8):
    tot, n = 0.0, 0
    for toks, labs in batches[:max_batches]:
        l = forward_loss(model, Tensor(toks, dtype=dtypes.int32), Tensor(labs, dtype=dtypes.int32)).item()
        tot += l; n += 1
    return tot / max(n, 1)

def save_lora(model, path: pathlib.Path, meta: dict):
    path.mkdir(parents=True, exist_ok=True)
    sd = {k: v for k, v in get_state_dict(model).items() if "lora" in k}
    safe_save(sd, path / "adapter.safetensors")
    (path / "meta.json").write_text(json.dumps(meta, indent=1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b-instruct")
    p.add_argument("--data", default="tmp/train_shuffled.jsonl",
                   help="canonical SHUFFLED stream (scripts/shuffle_train.py)")
    p.add_argument("--val", default="data/val.jsonl")
    p.add_argument("--run", required=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=160)
    p.add_argument("--offset", type=int, default=0, help="cursor into the shuffled stream")
    p.add_argument("--window", type=int, default=256, help="max examples per process")
    p.add_argument("--max-minutes", type=float, default=4.0)
    p.add_argument("--val-every", type=int, default=4)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="LoRA init seed only")
    p.add_argument("--init-adapter", default=None)
    args = p.parse_args()

    out = pathlib.Path("out") / f"lora_{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")

    stream = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    offset = args.offset
    if offset >= len(stream):
        print(f"cursor {offset} >= stream len {len(stream)}; wrapping to 0", flush=True)
        offset = 0
    window = stream[offset:offset + args.window]
    print(f"stream cursor {offset}..{offset + len(window)} of {len(stream)}", flush=True)

    print("loading data...", flush=True)
    val_raw = [json.loads(l) for l in open(args.val, encoding="utf-8")][:100]  # fixed subset
    tok = Tokenizer(args.model)
    exs, dropped = encode_stream(window, tok, args.seqlen)
    val_exs, _ = encode_stream(val_raw, tok, args.seqlen)
    val_batches = []
    for j in range(0, len(val_exs) - args.batch + 1, args.batch):
        chunk = val_exs[j:j + args.batch]
        toks = [t + [151643] * (args.seqlen - len(t)) for t, _ in chunk]
        labs = [l + [-100] * (args.seqlen - len(l)) for _, l in chunk]
        val_batches.append((toks, labs))
    print(f"{len(exs)} examples in window ({dropped} dropped > {args.seqlen} toks), val {len(val_batches)} batches", flush=True)

    print("loading model...", flush=True)
    cfg = Config.from_hf(args.model)
    model = Qwen3(cfg, lora_r=args.lora_r)
    model.load_hf(args.model)
    if FP16:  # fp16 base weights: Metal tensor-core friendly, halves watchdog pressure
        for k, t in get_state_dict(model).items():
            if "lora" not in k and t.dtype == dtypes.bfloat16: t.replace(t.cast(dtypes.float16))
    if args.init_adapter and (pathlib.Path(args.init_adapter) / "adapter.safetensors").exists():
        asd = {k: v.to(Device.DEFAULT) for k, v in safe_load(f"{args.init_adapter}/adapter.safetensors").items()}
        load_state_dict(model, asd, strict=False)
        print(f"resumed adapter from {args.init_adapter}", flush=True)
    params = model.lora_params()
    print(f"LoRA r={args.lora_r} alpha={args.alpha}: {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)
    opt = nn.optim.AdamW(params, lr=args.lr, weight_decay=0.0, b1=0.9, b2=0.95)

    Tensor.manual_seed(args.seed)
    t_start = time.time()
    step, best_val, hist = 0, float("inf"), []
    consumed = 0
    print("training...", flush=True)
    with Context(TRAINING=1):
        dev = Device[Device.DEFAULT]
        done = False
        while not done and consumed < len(exs) and time.time() - t_start < args.max_minutes * 60:
            t0 = time.time()
            # NOTE: backward PER micro-batch + full sync each micro: giant fused
            # graphs trip the macOS GPU watchdog (see JOURNAL). One micro = one step.
            chunk = exs[consumed:consumed + args.batch]
            if len(chunk) < args.batch: break
            toks = [t + [151643] * (args.seqlen - len(t)) for t, _ in chunk]
            labs = [l + [-100] * (args.seqlen - len(l)) for _, l in chunk]
            l = forward_loss(model, Tensor(toks, dtype=dtypes.int32), Tensor(labs, dtype=dtypes.int32))
            lv = float(l.item())  # sync per micro-batch
            grads = l.gradient(*params)
            for pp, g in zip(params, grads): pp.grad = g
            Tensor.realize(*[pp.grad for pp in params])
            dev.synchronize()
            opt.step()
            for pp in params: pp.grad = None
            dev.synchronize()
            consumed += args.batch
            step += 1
            hist.append(lv)
            if step % 5 == 0:
                msg = {"step": step, "cursor": offset + consumed, "loss": round(lv, 4),
                       "sec": round(time.time() - t0, 2), "elapsed_min": round((time.time() - t_start) / 60, 1)}
                print(json.dumps(msg), flush=True)
                log.write(json.dumps(msg) + "\n"); log.flush()
            if step % args.val_every == 0:
                vl = val_loss(model, val_batches, args.val_batches)
                msg = {"step": step, "cursor": offset + consumed, "val_loss": round(vl, 4)}
                print(json.dumps(msg), flush=True)
                log.write(json.dumps(msg) + "\n"); log.flush()
                if vl < best_val:
                    best_val = vl
                    save_lora(model, out / "best", {"step": step, "cursor": offset + consumed,
                                                    "val_loss": vl, "args": vars(args)})
            save_lora(model, out / "last", {"step": step, "cursor": offset + consumed,
                                            "loss": lv, "args": vars(args)})

    save_lora(model, out / "last", {"step": step, "cursor": offset + consumed, "args": vars(args)})
    (out / "history.json").write_text(json.dumps({"loss": hist[::5], "best_val": best_val, "steps": step}))
    print(f"done: {step} steps, consumed {consumed} examples, cursor={offset + consumed}, best_val={best_val:.4f}", flush=True)

if __name__ == "__main__":
    main()
