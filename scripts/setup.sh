#!/bin/bash
# One-time setup from a clean clone: python env, model weights, dataset.
# raw_data/ is NOT in git (it is the user's local data) — dataset rebuild
# requires it; skip below if absent.
set -e
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  PY=$(command -v python3.12 || command -v python3)
  "$PY" -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q tinygrad tokenizers numpy
echo "-- python env ready"

bash scripts/download_model.sh

if [ -d raw_data ]; then
  .venv/bin/python scripts/build_dataset.py
  echo "-- dataset built"
else
  echo "-- NOTE: raw_data/ not present; skipping dataset build (needed only for retraining)"
fi
echo "setup complete — try: .venv/bin/python correct.py \"He go to school yesterday.\""
