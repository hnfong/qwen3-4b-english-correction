#!/usr/bin/env python3
"""Qwen3 in tinygrad: model + LoRA + HF-safetensors loader + train/infer.

Custom implementation (tinygrad 0.14 has no Qwen3 training path; its llm module
is inference-only and broken in the pip release). Validated against HF
transformers in tmp/validate_qwen3.py (see JOURNAL).

Conventions: weights stored (out,in), forward = x @ W.T. Base weights are plain
Tensors (frozen); only LoRA A/B get gradients (loss.gradient(*lora_params)).
"""
import json, pathlib
from dataclasses import dataclass

from tinygrad import Tensor, nn, dtypes, Device
from tinygrad.nn.state import safe_load

DEFAULT_SYSTEM = "You are an English editor. Fix grammar, word choice and phrasing of the given English sentence. Reply with the corrected sentence only, with no explanations. If the sentence is already correct, reply with it unchanged."
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
EOS_ID = 151645  # <|im_end|>

@dataclass
class Config:
  dim: int; hidden_dim: int; n_heads: int; n_kv_heads: int; n_layers: int
  vocab_size: int; head_dim: int = 128; norm_eps: float = 1e-6; rope_theta: float = 5e6
  max_context: int = 1536

  @staticmethod
  def from_hf(path):
    c = json.loads((pathlib.Path(path) / "config.json").read_text())
    return Config(dim=c["hidden_size"], hidden_dim=c["intermediate_size"],
                  n_heads=c["num_attention_heads"], n_kv_heads=c["num_key_value_heads"],
                  n_layers=c["num_hidden_layers"], vocab_size=c["vocab_size"],
                  head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
                  norm_eps=c["rms_norm_eps"], rope_theta=c["rope_theta"])

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.weight = Tensor.ones(dim); self.eps = eps
  def __call__(self, x):
    out_dtype = x.dtype
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + self.eps).rsqrt()).cast(out_dtype) * self.weight.cast(out_dtype)

class LoRALinear:
  """Frozen W (out,in) + optional LoRA: y = x@W.T + scale*(x@A.T@B.T). r=0 -> plain."""
  def __init__(self, w: Tensor, r: int = 0, alpha: float = 16.0):
    self.w = w
    self.lora_a = self.lora_b = None
    if r > 0:
      out_f, in_f = w.shape
      self.lora_a = (Tensor.randn(r, in_f) * (in_f ** -0.5)).float().is_param_()
      self.lora_b = Tensor.zeros(out_f, r, dtype=dtypes.float32).is_param_()
      self.scale = alpha / r
  def __call__(self, x: Tensor) -> Tensor:
    y = x @ self.w.T
    if self.lora_a is not None:
      y = y + self.scale * ((x @ self.lora_a.T) @ self.lora_b.T).cast(y.dtype)
    return y
  def params(self):
    return [p for p in (self.lora_a, self.lora_b) if p is not None]

def rope_tables(max_context, head_dim, theta):
  inv = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2, dtype=dtypes.float32) / head_dim))
  ang = Tensor.arange(max_context, dtype=dtypes.float32)[:, None] * inv[None, :]
  return ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]  # (1,T,1,D/2)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, start_pos: int = 0):
  # x: (B,T,H,D); rotate_half convention, tables sliced then broadcast
  T = x.shape[1]
  c = cos[:, start_pos:start_pos + T].cast(x.dtype)
  s = sin[:, start_pos:start_pos + T].cast(x.dtype)
  half = x.shape[-1] // 2
  x1, x2 = x[..., :half], x[..., half:]
  return (x1 * c - x2 * s).cat(x2 * c + x1 * s, dim=-1)

def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  # (B,T,KVH,D) -> (B,T,H,D)
  if n_rep == 1: return x
  B, T, KVH, D = x.shape
  return x.unsqueeze(3).expand(B, T, KVH, n_rep, D).reshape(B, T, KVH * n_rep, D)

class Attention:
  def __init__(self, cfg: Config, lora_r: int = 0):
    hd, d = cfg.head_dim, cfg.dim
    self.cfg = cfg
    wq = Tensor.empty(cfg.n_heads * hd, d, dtype=dtypes.bfloat16)
    wk = Tensor.empty(cfg.n_kv_heads * hd, d, dtype=dtypes.bfloat16)
    wv = Tensor.empty(cfg.n_kv_heads * hd, d, dtype=dtypes.bfloat16)
    wo = Tensor.empty(d, cfg.n_heads * hd, dtype=dtypes.bfloat16)
    self.q, self.k, self.v, self.o = (LoRALinear(w, lora_r) for w in (wq, wk, wv, wo))
    self.q_norm, self.k_norm = RMSNorm(hd, cfg.norm_eps), RMSNorm(hd, cfg.norm_eps)
  def linears(self):
    return [self.q, self.k, self.v, self.o]
  def __call__(self, x: Tensor, cos, sin, start_pos: int = 0, cache: Tensor | None = None):
    B, T, _ = x.shape
    cfg, n_rep = self.cfg, self.cfg.n_heads // self.cfg.n_kv_heads
    xq = self.q(x).reshape(B, T, cfg.n_heads, cfg.head_dim)
    xk = self.k(x).reshape(B, T, cfg.n_kv_heads, cfg.head_dim)
    xv = self.v(x).reshape(B, T, cfg.n_kv_heads, cfg.head_dim)
    xq, xk = self.q_norm(xq), self.k_norm(xk)
    xq, xk = apply_rope(xq, cos, sin, start_pos), apply_rope(xk, cos, sin, start_pos)
    q_ = xq.transpose(1, 2)  # (B,H,T,D)
    if cache is not None:  # inference; cache: (2,B,ctx,KVH,D)
      cache[:, :, start_pos:start_pos + T].assign(Tensor.stack(xk, xv)).realize()
      end = start_pos + T
      keys = repeat_kv(cache[0, :, :end], n_rep).transpose(1, 2)      # (B,H,S,D)
      values = repeat_kv(cache[1, :, :end], n_rep).transpose(1, 2)
      if T > 1:
        mask = Tensor.full((1, 1, T, end), float("-inf"), dtype=x.dtype).triu(start_pos + 1)
        attn = q_.scaled_dot_product_attention(keys, values, mask)
      else:
        attn = q_.scaled_dot_product_attention(keys, values)
    else:  # training: no cache, GQA handled by sdpa
      keys, values = xk.transpose(1, 2), xv.transpose(1, 2)           # (B,KVH,T,D)
      attn = q_.scaled_dot_product_attention(keys, values, is_causal=True, enable_gqa=True)
    return self.o(attn.transpose(1, 2).reshape(B, T, -1))

class MLP:
  def __init__(self, cfg: Config, lora_r: int = 0):
    wg = Tensor.empty(cfg.hidden_dim, cfg.dim, dtype=dtypes.bfloat16)
    wu = Tensor.empty(cfg.hidden_dim, cfg.dim, dtype=dtypes.bfloat16)
    wd = Tensor.empty(cfg.dim, cfg.hidden_dim, dtype=dtypes.bfloat16)
    self.gate, self.up, self.down = (LoRALinear(w, lora_r) for w in (wg, wu, wd))
  def linears(self):
    return [self.gate, self.up, self.down]
  def __call__(self, x: Tensor) -> Tensor:
    return self.down(self.gate(x).silu() * self.up(x))

class Block:
  def __init__(self, cfg: Config, lora_r: int = 0):
    self.attn = Attention(cfg, lora_r)
    self.mlp = MLP(cfg, lora_r)
    self.norm1, self.norm2 = RMSNorm(cfg.dim, cfg.norm_eps), RMSNorm(cfg.dim, cfg.norm_eps)
  def __call__(self, x, cos, sin, start_pos=0, cache=None):
    h = x + self.attn(self.norm1(x), cos, sin, start_pos, cache)
    return h + self.mlp(self.norm2(h))

class Qwen3:
  eos_id = EOS_ID
  def __init__(self, cfg: Config, lora_r: int = 0):
    self.cfg = cfg
    self.emb = Tensor.empty(cfg.vocab_size, cfg.dim, dtype=dtypes.bfloat16)
    self.blk = [Block(cfg, lora_r) for _ in range(cfg.n_layers)]
    self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
    self.cos, self.sin = rope_tables(cfg.max_context, cfg.head_dim, cfg.rope_theta)
    self.caches = None
    self._decode_jit = None
  # ---- weights
  def load_hf(self, path: str):
    path = pathlib.Path(path)
    idx = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {s: safe_load(path / s) for s in sorted(set(idx.values()))}
    hf = {k: shards[s][k].to(Device.DEFAULT) for k, s in idx.items()}
    self.emb.replace(hf["model.embed_tokens.weight"]); self.norm.weight.replace(hf["model.norm.weight"])
    for i, blk in enumerate(self.blk):
      p = f"model.layers.{i}."
      a, m = blk.attn, blk.mlp
      a.q.w.replace(hf[p + "self_attn.q_proj.weight"]); a.k.w.replace(hf[p + "self_attn.k_proj.weight"])
      a.v.w.replace(hf[p + "self_attn.v_proj.weight"]); a.o.w.replace(hf[p + "self_attn.o_proj.weight"])
      a.q_norm.weight.replace(hf[p + "self_attn.q_norm.weight"]); a.k_norm.weight.replace(hf[p + "self_attn.k_norm.weight"])
      blk.norm1.weight.replace(hf[p + "input_layernorm.weight"]); blk.norm2.weight.replace(hf[p + "post_attention_layernorm.weight"])
      m.gate.w.replace(hf[p + "mlp.gate_proj.weight"]); m.up.w.replace(hf[p + "mlp.up_proj.weight"]); m.down.w.replace(hf[p + "mlp.down_proj.weight"])
    Tensor.realize(self.emb, self.norm.weight)
    return self
  # ---- LoRA
  def lora_params(self):
    return [p for blk in self.blk for lin in (*blk.attn.linears(), *blk.mlp.linears()) for p in lin.params()]
  def merge_loras(self):
    """Fold LoRA deltas into base weights (in place)."""
    for blk in self.blk:
      for lin in (*blk.attn.linears(), *blk.mlp.linears()):
        if lin.lora_a is not None:
          delta = (lin.lora_b @ lin.lora_a) * lin.scale  # (out,in) fp32
          lin.w.replace((lin.w.float() + delta).cast(lin.w.dtype))
          lin.lora_a = lin.lora_b = None
    Tensor.realize(*[lin.w for blk in self.blk for lin in (*blk.attn.linears(), *blk.mlp.linears())])
  # ---- forward
  def forward_train(self, tokens: Tensor) -> Tensor:
    h = self.emb[tokens].contiguous()
    for blk in self.blk: h = blk(h, self.cos, self.sin)
    # no per-block realize here: single scheduling run, CSE handles everything
    return (self.norm(h).cast(self.emb.dtype) @ self.emb.T).float()  # (B,T,V) tied head
  # ---- inference
  def init_cache(self, batch: int = 1):
    if self.caches is not None: return
    c = self.cfg
    self.caches = [Tensor.zeros(2, batch, c.max_context, c.n_kv_heads, c.head_dim, dtype=dtypes.bfloat16).contiguous().realize()
                   for _ in range(c.n_layers)]
  def _forward_cache(self, tokens: Tensor, start_pos, last_idx: int | None = None) -> Tensor:
    """Cached forward. start_pos may be an int (prefill) or bound Variable (jitted decode).
    Returns logits at last_idx (default: last position)."""
    assert self.caches is not None, "call init_cache first"
    h = self.emb[tokens].contiguous()
    for blk, cache in zip(self.blk, self.caches):
      # NOTE: realize h per block. The in-block cache assign.realize() splits scheduling;
      # without realizing h, each block would recompute the entire lazy prefix (quadratic).
      h = blk(h, self.cos, self.sin, start_pos, cache).realize()
    if last_idx is not None: h = h[:, last_idx:last_idx + 1]
    else: h = h[:, -1:]
    return (self.norm(h).cast(self.emb.dtype) @ self.emb.T).float()
  def _prefill(self, ids: list[int], chunk: int = 512) -> Tensor:
    """Right-padded chunked prefill with fixed shapes (stable kernel cache).
    Pad positions beyond len(ids) are never read afterwards."""
    import math
    n = len(ids)
    n_chunks = max(1, math.ceil(n / chunk))
    logits = None
    for ci in range(n_chunks):
      seg = ids[ci * chunk:(ci + 1) * chunk]
      seg = seg + [0] * (chunk - len(seg))
      logits = self._forward_cache(Tensor([seg], dtype=dtypes.int32), ci * chunk,
                                   last_idx=(n - 1 - ci * chunk) if ci == n_chunks - 1 else None)
    return logits[0]
  def _decode_step(self, tokens: Tensor, start_pos) -> Tensor:
    return self._forward_cache(tokens, start_pos)[:, 0]
  def generate(self, prompt_ids: list[int], max_new: int = 120, temperature: float = 0.0) -> list[int]:
    from tinygrad import Variable, TinyJit
    c = self.cfg
    assert len(prompt_ids) + max_new <= c.max_context, "prompt too long"
    self.init_cache()
    if self._decode_jit is None:
      self._decode_jit = TinyJit(self._decode_step, prune=True)
    logits = self._prefill(prompt_ids).realize()
    sp = len(prompt_ids)
    v = Variable("start_pos", 1, c.max_context - 1)
    out = []
    for _ in range(max_new):
      if temperature <= 0:
        nxt = int(logits.argmax().item())
      else:
        probs = (logits / temperature).softmax().numpy()
        nxt = int(__import__("numpy").random.choice(len(probs), p=probs))
      out.append(nxt)
      if nxt == self.eos_id or sp + 1 >= c.max_context: break
      logits = self._decode_jit(Tensor([[nxt]], dtype=dtypes.int32), v.bind(sp)).realize()[0]
      sp += 1
    return out

# ---- chat template / example encoding ----
def build_prompt(user_content: str, system: str = DEFAULT_SYSTEM) -> str:
  return f"{IM_START}system\n{system}{IM_END}\n{IM_START}user\n{user_content}{IM_END}\n{IM_START}assistant\n"

def user_content(en: str, zh: str | None) -> str:
  return (f"Cantonese reference: {zh}\n" if zh else "") + f"English sentence: {en}"

def encode_example(tok, ex) -> tuple[list[int], list[int]]:
  """(tokens, labels) with NEXT-token alignment: labels[i] = token the model should
  predict at position i (i.e. tokens[i+1]); -100 where nothing to predict/train."""
  p_ids = tok.encode(build_prompt(user_content(ex["en"], ex["zh"])))
  t_ids = tok.encode(ex["target"]) + [EOS_ID]
  toks = p_ids + t_ids
  labels = [-100] * len(toks)
  for i in range(len(p_ids) - 1, len(toks) - 1):  # positions whose next token is target/EOS
    labels[i] = toks[i + 1]
  return toks, labels

class Tokenizer:
  def __init__(self, path: str):
    from tokenizers import Tokenizer as HFTokenizer
    self.tok = HFTokenizer.from_file(str(pathlib.Path(path) / "tokenizer.json"))
  def encode(self, text: str) -> list[int]:
    return self.tok.encode(text, add_special_tokens=False).ids
  def decode(self, ids: list[int]) -> str:
    return self.tok.decode(ids, skip_special_tokens=True)
