#!/usr/bin/env python3
"""Upload pipeline artifacts to the Hugging Face repo.

Uploads (per fine-tuned model found in models/):
  models/<name>-english-correction-mlx/  ->  <name>-full/          (merged model)
  models/<name>-english-correction-{f16,q8_0}.gguf ->  root         (llama.cpp)
Plus repo furniture: hf_card.md -> README.md, LICENSE, data/.

Token: --token, or $HF_TOKEN, or ~/.cache/huggingface/token (in that order).
The token is never printed or stored anywhere else.

Usage:
  .venv/bin/python scripts/upload_hf.py --dry-run     # show the plan
  .venv/bin/python scripts/upload_hf.py               # upload everything found
  .venv/bin/python scripts/upload_hf.py --model qwen3-0.6b
"""
import argparse, os, pathlib, sys

DEFAULT_REPO = "sydneyfong/qwen3-4b-english-correction"
TOKEN_PATH = pathlib.Path.home() / ".cache" / "huggingface" / "token"
ROOT = pathlib.Path(__file__).resolve().parent.parent

def get_token(args):
    if args.token: return args.token.strip()
    if os.environ.get("HF_TOKEN"): return os.environ["HF_TOKEN"].strip()
    if TOKEN_PATH.exists(): return TOKEN_PATH.read_text().strip()
    sys.exit("no token: --token, $HF_TOKEN, or " + str(TOKEN_PATH))

def find_models():
    """Local fused-model dirs: models/<name>-english-correction-mlx"""
    return sorted({p.name.replace("-english-correction-mlx", "")
                   for p in (ROOT / "models").glob("*-english-correction-mlx")})

def build_plan(models, repo_files):
    plan = []  # (local path, repo path)
    for f, remote in repo_files:
        if pathlib.Path(ROOT / f).exists():
            plan.append((ROOT / f, remote))
        else:
            print(f"  skip (missing): {f}")
    for name in models:
        fused = ROOT / "models" / f"{name}-english-correction-mlx"
        if fused.exists():
            for f in sorted(fused.iterdir()):
                if f.name in (".done", "README.md") or not f.is_file(): continue
                plan.append((f, f"{name}-full/{f.name}"))
        for q in ("f16", "q8_0"):
            g = ROOT / "models" / f"{name}-english-correction-{q}.gguf"
            if g.exists(): plan.append((g, g.name))
            else: print(f"  skip (missing): {g.name}")
    return plan

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--model", action="append", default=None,
                    help="model name (e.g. qwen3-0.6b); repeatable; default: all found")
    ap.add_argument("--no-data", action="store_true", help="skip card/LICENSE/data")
    ap.add_argument("--token", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = args.model or find_models()
    repo_files = []
    if not args.no_data:
        repo_files += [("hf_card.md", "README.md"), ("LICENSE", "LICENSE"),
                       ("data/train.jsonl", "data/train.jsonl"),
                       ("data/val.jsonl", "data/val.jsonl"),
                       ("data/sample_review.txt", "data/sample_review.txt"),
                       ("data/stats.json", "data/stats.json")]
    plan = build_plan(models, repo_files)

    total = sum(f.stat().st_size for f, _ in plan)
    print(f"repo: {args.repo} | {len(plan)} files, {total/1e9:.2f} GB")
    for f, remote in plan:
        print(f"  {remote:52s} <- {f.relative_to(ROOT)} ({f.stat().st_size:>13,})")
    if args.dry_run:
        print("dry run — nothing uploaded"); return

    from huggingface_hub import HfApi
    api = HfApi(token=get_token(args))
    for f, remote in plan:
        print(f"uploading {remote} ...", flush=True)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=remote,
                        repo_id=args.repo, commit_message=f"upload {remote}")
    print(f"done: https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
