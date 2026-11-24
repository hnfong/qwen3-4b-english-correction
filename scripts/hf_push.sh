#!/bin/bash
# Push hf_repo/ to https://huggingface.co/sydneyfong/qwen3-4b-english-correction
# Reads the HF token from ~/.cache/huggingface/token at runtime (never stored).
# Requires a token WITH write access (fine-grained: repo.content.write on the
# user namespace, or a classic "write" token).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/hf_repo"
TOKEN=$(cat ~/.cache/huggingface/token)
REPO=sydneyfong/qwen3-4b-english-correction

"$ROOT/.venv/bin/python" - <<EOF
from huggingface_hub import HfApi
import pathlib
api = HfApi(token=pathlib.Path.home().joinpath(".cache/huggingface/token").read_text().strip())
print("repo:", api.create_repo("$REPO", exist_ok=True, private=False))
EOF

git remote remove origin 2>/dev/null || true
git remote add origin "https://sydneyfong:${TOKEN}@huggingface.co/${REPO}.git"
git push -u origin main
git remote set-url origin "https://huggingface.co/${REPO}.git"   # strip token from .git/config
echo "pushed; remote url scrubbed"
