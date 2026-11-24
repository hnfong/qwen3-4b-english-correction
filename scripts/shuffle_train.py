#!/usr/bin/env python3
"""Shuffle data/train.jsonl ONCE, deterministically, into tmp/train_shuffled.jsonl.

Training consumes that file SEQUENTIALLY via a cursor (offset) stored in the
checkpoint; restarts resume where they left off. The shuffle is seeded, so a
deleted file regenerates in the same order. Run: .venv/bin/python scripts/shuffle_train.py
"""
import json, random, pathlib

src = pathlib.Path("data/train.jsonl")
dst = pathlib.Path("tmp/train_shuffled.jsonl")
if dst.exists():
    n = sum(1 for _ in open(dst, encoding="utf-8"))
    print(f"{dst} already exists ({n} lines) — leaving it alone (delete to reshuffle)")
else:
    rows = open(src, encoding="utf-8").readlines()
    random.Random(0).shuffle(rows)
    dst.parent.mkdir(exist_ok=True)
    dst.write_text("".join(rows), encoding="utf-8")
    print(f"wrote {len(rows)} shuffled rows -> {dst}")
