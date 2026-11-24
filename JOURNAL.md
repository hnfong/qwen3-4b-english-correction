# JOURNAL

Distilled context for any successor agent. Read this + QUESTIONS_FOR_USER.md +
PLAN_01_overall.md first. Newest entries at the top.

## 2024-11-24 — Session start, P0 audit done

**Done**
- Answered Q1–Q4 with the user (see QUESTIONS_FOR_USER.md). Key decisions:
  Qwen3-4B (no VL) as base; use data as-is, human CSVs are gold; tinygrad CLI
  deliverable; ~a dozen synthetic eval sentences as the quality bar.
- Data audit (tmp/audit1.py, tmp/audit2.py). Findings in PLAN_01_overall.md P0.
  Most important: the three TSV groups cover the SAME sentences with per-teacher
  suggestions; reviews (score 0–5) judge the original English only; `original`
  embeds the Cantonese reference after an optional sense-letter prefix; some
  teacher outputs are refusals or empty and must be filtered.

**Difficulties / learnings**
- The TSV rows are ragged: 5 header columns but one extra tab-separated column
  per review. Parse by splitting on \t and taking fields[4:] as reviews; CSVs are
  properly quoted, use csv module there.
- A naive "first Latin char after CJK" Cantonese/English split fails ~16% of rows
  (English words like "USB"/"MK"/"form" inside the Cantonese part). Plan: split at
  the LAST CJK(char-or-punct)→whitespace→Latin boundary, validate both sides.
- PLAN writing slip: initially wrote future phases as done with invented numbers;
  caught and rewrote honestly. Keep plan status markers truthful at commit time.

**Next**: P1 env setup (venv, tinygrad, clone reference repo), then P3 model
download in background while building P2 dataset.
