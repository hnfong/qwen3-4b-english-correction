#!/usr/bin/env python3
"""Build SFT dataset from raw_data/ TSVs + human-reviewed CSVs.

Output: data/train.jsonl, data/val.jsonl, data/stats.json, data/sample_review.txt
Each example: {"en": str, "zh": str|null, "target": str, "source": str,
               "kind": "correction"|"identity", "avg_score": float|null}
"""
import csv, glob, json, re, sys, hashlib
from collections import defaultdict
from pathlib import Path

RAW = Path("raw_data")
OUT = Path("data")
OUT.mkdir(exist_ok=True)

CJK_IDEO = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF]")
CJK_ANY = re.compile(r"[\u2E80-\u9FFF\uF900-\uFAFF\u3000-\u303F\uFF01-\uFF60\u2018\u201C]")
SENSE_LETTER = re.compile(r"^([A-J])\.\s+(?=[^\x00-\x7F])")
SCORE_RE = re.compile(r"[GH]\. Score:\s*(\d+)")
REFUSAL_RE = re.compile(r"^(?:[A-Z]\.\s*)?(skipped|skip\b|n/?a$|none$|-+$)", re.I)
REFUSAL_SUBSTR = ("inappropriate content", "cannot assist", "cannot fulfill", "i cannot")
# greedy: boundary = CJK char/punct run + whitespace + Latin/quote/paren start
ZH_EN_SPLIT = re.compile(
    r"^(.*[\u2E80-\u9FFF\uF900-\uFAFF\u3000-\u303F\uFF01-\uFF60][\u3000-\u303F\uFF01-\uFF60\]\)]*\s+)"
    r"([A-Za-z\"(\u2018\u201C].*)$", re.S)
EN_OK = re.compile(r"^[A-Za-z\"(\u2018\u201C]")

def parse_scores(reviews):
    vals = [int(m.group(1)) for r in reviews for m in [SCORE_RE.search(r)] if m]
    return sum(vals) / len(vals) if vals else None

def is_refusal(s):
    return bool(REFUSAL_RE.match(s)) or any(sub in s.lower() for sub in REFUSAL_SUBSTR)

def split_zh_en(original):
    """Return (zh, en). zh may be None (English-only or unparseable)."""
    o = original.strip()
    o = SENSE_LETTER.sub("", o)
    m = ZH_EN_SPLIT.match(o)  # greedy first group -> LAST boundary
    if m:
        zh, en = m.group(1).strip(), m.group(2).strip()
        if CJK_IDEO.search(zh) and en and EN_OK.match(en):
            return zh, en
    if not CJK_IDEO.search(o):  # English-only entry
        return None, o
    # fallback: after last CJK char/punct
    idxs = [m.end() - 1 for m in CJK_ANY.finditer(o)]
    if idxs:
        en = o[idxs[-1] + 1:].strip()
        if en and EN_OK.match(en) and len(en) > 10:
            zh = o[:idxs[-1] + 1].strip()
            if CJK_IDEO.search(zh):
                return zh, en
    return None, o  # unparseable: treat all as English, zh=None

def read_tsv(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            rows.append(line.split("\t"))
    return rows

def group_of(name):
    if name.startswith("gemma3-"): return "gemma3"
    if name.startswith("llama3-"): return "llama3"
    return "plain"

# ---- collect rows: key=(word_id, original) -> {reviews, suggestions: {group: sugg}, human: sugg|None}
entries = {}

def add(word_id, original, reviews, suggestion, group, human):
    key = (word_id.strip(), original.strip())
    e = entries.setdefault(key, {"reviews": [], "sugg": {}, "human": None})
    if reviews and not e["reviews"]:
        e["reviews"] = reviews
    if suggestion and suggestion.strip():
        s = suggestion.strip()
        if human:
            if not e["human"] and not is_refusal(s): e["human"] = s
        else:
            e["sugg"].setdefault(group, s)

for path in sorted(glob.glob(str(RAW / "*.tsv"))):
    g = group_of(Path(path).name)
    for r in read_tsv(path)[1:]:
        if len(r) < 4: continue
        add(r[0], r[2], [c for c in r[4:] if c.strip()], r[3], g, human=False)

for path in sorted(glob.glob(str(RAW / "*.csv"))):
    for r in list(csv.reader(open(path, newline="", encoding="utf-8")))[1:]:
        if len(r) < 4: continue
        add(r[0], r[2], [c for c in r[4:] if c.strip()], r[3], group_of(Path(path).name), human=True)

# ---- build examples
examples, stats = [], defaultdict(int)
for (word_id, original), e in entries.items():
    stats["n_entries"] += 1
    zh, en = split_zh_en(original)
    if not en or not en.strip():
        stats["drop_no_english"] += 1; continue
    if len(en) < 4:
        stats["drop_too_short"] += 1; continue
    if CJK_IDEO.search(en):
        stats["drop_en_has_cjk"] += 1; continue
    avg = parse_scores(e["reviews"])

    candidates = []  # (source, target)
    if e["human"]:
        candidates.append(("human", e["human"]))
    for g, s in e["sugg"].items():
        candidates.append((g, s))

    targets, seen = [], set()
    for src, t in candidates:
        t = t.strip()
        if not t or is_refusal(t):
            stats[f"refusal_{src}"] += 1; continue
        if CJK_IDEO.search(t):
            stats[f"target_cjk_{src}"] += 1; continue
        if t.lower() == en.lower():
            stats["target_same_as_original"] += 1; continue
        if t not in seen:
            seen.add(t); targets.append((src, t))

    if targets:
        for src, t in targets:
            examples.append({"en": en, "zh": zh, "target": t, "source": src,
                             "kind": "correction", "avg_score": avg})
            stats[f"pair_{src}"] += 1
    elif avg is not None and avg >= 4.0:
        examples.append({"en": en, "zh": zh, "target": en, "source": "identity",
                         "kind": "identity", "avg_score": avg})
        stats["identity_pairs"] += 1
    else:
        stats["drop_no_target"] += 1

# ---- val split by input (no leakage of same sentence across train/val)
def input_hash(ex): return hashlib.md5((json.dumps([ex["en"], ex["zh"]])).encode()).hexdigest()
val, train = [], []
for ex in examples:
    (val if int(input_hash(ex), 16) % 50 == 0 else train).append(ex)

with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
    for ex in train: f.write(json.dumps(ex, ensure_ascii=False) + "\n")
with open(OUT / "val.jsonl", "w", encoding="utf-8") as f:
    for ex in val: f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# ---- human-reviewable sample
with open(OUT / "sample_review.txt", "w", encoding="utf-8") as f:
    by_kind = defaultdict(list)
    for ex in examples:
        by_kind["identity" if ex["kind"] == "identity" else f"correction_{ex['source']}"].append(ex)
    for kind, xs in sorted(by_kind.items()):
        f.write(f"\n{'='*80}\n{kind}  (n={len(xs)})\n{'='*80}\n")
        for ex in xs[:25]:
            f.write(f"ZH: {ex['zh'] or '(none)'}\nEN: {ex['en']}\nTGT: {ex['target']}\n--\n")

stats_out = {
    "train": len(train), "val": len(val),
    "n_unique_inputs": len({input_hash(ex) for ex in examples}),
    **dict(sorted(stats.items())),
}
(OUT / "stats.json").write_text(json.dumps(stats_out, indent=1, ensure_ascii=False))
print(json.dumps(stats_out, indent=1, ensure_ascii=False))
