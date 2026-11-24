#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-4B on the correction dataset (tinygrad).

Time-bounded runs (<= --max-minutes) with periodic checkpoints + val loss.
Usage: .venv/bin/python scripts/train_lora.py --run r16 [--max-examples 4000]
"""
import argparse, json, os, pathlib, random, sys, time
# macOS GPU watchdog mitigation: with a 5K display attached, sustained GPU
# saturation makes macOS kill command buffers
# (kIOGPUCommandBufferCallbackErrorImpactingInteractivity). A tiny sleep per
# kernel submission creates gaps that keep the compositor responsive.
PACE_S = float(os.environ.get('PACE_S', '0.005'))
FP16 = os.environ.get('TRAIN_FP16', '1') == '1'

from tinygrad import Tensor, dtypes, nn, Device
from tinygrad.helpers import Context, TRAINING
from tinygrad.nn.state import safe_save, get_state_dict, safe_load, load_state_dict
import tinygrad.runtime.ops_metal as _om
if PACE_S > 0:
    _orig_metal_call = _om.MetalProgram.__call__
    def _paced_call(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kw):
        r = _orig_metal_call(self, *bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, **kw)
        time.sleep(PACE_S)
        return r
    _om.MetalProgram.__call__ = _paced_call
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qwen3 import Qwen3, Config, Tokenizer, encode_example

def build_batches(data, tok, seqlen, bs, seed, shuffle=True):
    """Tokenize + bucket by length; returns list of (tokens, labels) int lists."""
    exs, dropped = [], 0
    for ex in data:
        toks, labels = encode_example(tok, ex)
        if len(toks) > seqlen:
            dropped += 1; continue
        exs.append((toks, labels))
    rng = random.Random(seed)
    if shuffle: rng.shuffle(exs)
    out = []
    for i in range(0, len(exs), bs * 64):  # sort within mega-chunks to limit padding
        chunk = sorted(exs[i:i + bs * 64], key=lambda x: len(x[0]))
        for j in range(0, len(chunk), bs):
            batch = chunk[j:j + bs]
            if not batch: continue
            # NOTE: pad EVERYTHING to exactly seqlen -> one graph shape for the whole
            # run. Variable batch-max shapes recompile the train graph per batch
            # (minutes each on Metal). Padding waste is far cheaper than recompiles.
            toks = [t + [151643] * (seqlen - len(t)) for t, _ in batch]
            labs = [l + [-100] * (seqlen - len(l)) for _, l in batch]  # tuples are (toks, labels)
            out.append((toks, labs))
    return out, dropped

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
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--val", default="data/val.jsonl")
    p.add_argument("--run", default="r16")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=1, help="keep 1: multi-micro steps trip the macOS GPU watchdog")
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--max-examples", type=int, default=4000)
    p.add_argument("--max-minutes", type=float, default=28.0)
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-adapter", default=None, help="load adapter.safetensors before training (resume)")
    p.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0 = time-bound only)")
    args = p.parse_args()

    out = pathlib.Path("out") / f"lora_{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")

    print("loading data...", flush=True)
    train_raw = [json.loads(l) for l in open(args.train)][: args.max_examples]
    val_raw = [json.loads(l) for l in open(args.val)]
    tok = Tokenizer(args.model)
    batches, dropped = build_batches(train_raw, tok, args.seqlen, args.batch, args.seed)
    val_batches, _ = build_batches(val_raw, tok, args.seqlen, 4, 0, shuffle=False)
    print(f"{len(batches)} micro-batches ({dropped} dropped > {args.seqlen} toks), val {len(val_batches)}", flush=True)

    print("loading model...", flush=True)
    cfg = Config.from_hf(args.model)
    model = Qwen3(cfg, lora_r=args.lora_r)
    model.load_hf(args.model)
    if FP16:  # fp16 base weights: Metal tensor-core friendly, halves watchdog pressure
        for k, t in get_state_dict(model).items():
            if 'lora' not in k and t.dtype == dtypes.bfloat16: t.replace(t.cast(dtypes.float16))
    if args.init_adapter and (pathlib.Path(args.init_adapter) / "adapter.safetensors").exists():
        sd = {k: v.to(Device.DEFAULT) for k, v in safe_load(f"{args.init_adapter}/adapter.safetensors").items()}
        load_state_dict(model, sd, strict=False)
        print(f"resumed adapter from {args.init_adapter}", flush=True)
    params = model.lora_params()
    n_lora = sum(p.numel() for p in params)
    print(f"LoRA r={args.lora_r} alpha={args.alpha}: {n_lora/1e6:.1f}M params", flush=True)
    opt = nn.optim.AdamW(params, lr=args.lr, weight_decay=0.0, b1=0.9, b2=0.95)

    Tensor.manual_seed(args.seed)
    t_start = time.time()
    step, best_val, hist = 0, float("inf"), []
    print("training...", flush=True)
    with Context(TRAINING=1):
        done = False
        while not done and time.time() - t_start < args.max_minutes * 60:
            epoch_shuf = batches[:]
            random.Random(args.seed + 1000 + step).shuffle(epoch_shuf)
            for i in range(0, len(epoch_shuf) - args.accum + 1, args.accum):
                if time.time() - t_start > args.max_minutes * 60:
                    done = True; break
                t0 = time.time()
                # NOTE: backward PER micro-batch (not one graph over accum forwards):
                # a single backward over accum forwards makes giant fused GPU command
                # buffers that trip macOS's GPU watchdog (kIOGPUCommandBufferCallback
                # ErrorImpactingInteractivity). Small per-micro graphs + a sync per
                # micro-batch (loss.item()) keep every command buffer short.
                pgrads, loss_vals = None, []
                for toks, labs in epoch_shuf[i:i + args.accum]:
                    l = forward_loss(model, Tensor(toks, dtype=dtypes.int32), Tensor(labs, dtype=dtypes.int32))
                    loss_vals.append(float(l.item()))  # sync per micro-batch
                    grads = l.gradient(*params)
                    if pgrads is None: pgrads = grads
                    else: pgrads = [a + b for a, b in zip(pgrads, grads)]
                    Tensor.realize(*[g for g in pgrads if g is not None])
                    Device[Device.DEFAULT].synchronize()  # full sync per micro (watchdog)
                for pp, g in zip(params, pgrads): pp.grad = g * (1.0 / args.accum)
                opt.step()
                for pp in params: pp.grad = None
                Device[Device.DEFAULT].synchronize()
                step += 1
                lv = sum(loss_vals) / len(loss_vals)
                hist.append(lv)
                if step % 5 == 0:
                    msg = {"step": step, "loss": round(lv, 4), "sec": round(time.time() - t0, 2),
                           "elapsed_min": round((time.time() - t_start) / 60, 1)}
                    print(json.dumps(msg), flush=True)
                    log.write(json.dumps(msg) + "\n"); log.flush()
                if step % args.val_every == 0:
                    vl = val_loss(model, val_batches)
                    msg = {"step": step, "val_loss": round(vl, 4)}
                    print(json.dumps(msg), flush=True)
                    log.write(json.dumps(msg) + "\n"); log.flush()
                    if vl < best_val:
                        best_val = vl
                        save_lora(model, out / "best", {"step": step, "val_loss": vl, "args": vars(args)})
                if step % args.save_every == 0:
                    save_lora(model, out / f"step{step}", {"step": step, "args": vars(args)})
                save_lora(model, out / "last", {"step": step, "args": vars(args)})
                if args.max_steps and step >= args.max_steps:
                    done = True; break

    save_lora(model, out / "last", {"step": step, "args": vars(args)})
    (out / "history.json").write_text(json.dumps({"loss": hist[::5], "best_val": best_val, "steps": step}))
    print(f"done: {step} steps, best_val={best_val:.4f}", flush=True)

if __name__ == "__main__":
    main()
