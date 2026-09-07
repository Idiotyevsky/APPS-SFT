#!/usr/bin/env python3
"""Generate ToolAPPS's own fixed evaluation set, following RLEF-Code's
published selection procedure on our local pinned APPS test copy.

Procedure (mirrors tarunbeerelli/RLEF-Code prepare_openrlhf_data.py):
  * pool per difficulty, ordered by numeric problem id (their dir-name sort);
  * random.seed(42); random.shuffle(pool); take the first 250 per bucket
    (introductory, interview, competition).

IMPORTANT: RLEF-Code's eval set is built from the official Hendrycks APPS
directory data (which carries reference solutions and an execution-verified
pool, hence competition=213). Our local codeparrot/apps test.jsonl has no
solutions and passes every record at loader level, so this artifact is
procedure-compatible but NOT per-problem identical to RLEF's 713. Numbers from
the two sets must not be compared item-by-item.

Outputs:
  data/eval/toolapps_eval.jsonl            rows (public + private io)
  data/eval/toolapps_eval_manifest.json    counts + procedure + hashes
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/raw/apps/test.jsonl"
OUT_DIR = ROOT / "data/eval"
OUT_JSONL = OUT_DIR / "toolapps_eval.jsonl"
OUT_MANIFEST = OUT_DIR / "toolapps_eval_manifest.json"

DIFF_ORDER = ["introductory", "interview", "competition"]
PER_BUCKET = 250
SEED = 42


def main() -> int:
    by_diff: dict[str, list[dict]] = {d: [] for d in DIFF_ORDER}
    for line in open(TEST, encoding="utf-8"):
        row = json.loads(line)
        difficulty = row.get("difficulty")
        if difficulty not in by_diff:
            continue
        by_diff[difficulty].append(row)

    for pool in by_diff.values():
        pool.sort(key=lambda r: int(r.get("id", -1)))

    random.seed(SEED)
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for difficulty in DIFF_ORDER:
        pool = list(by_diff[difficulty])
        random.shuffle(pool)
        take = pool[:PER_BUCKET]
        counts[difficulty] = len(take)
        selected.extend(take)

    selected.sort(key=lambda r: int(r["id"]))
    rows = []
    for raw in selected:
        io = json.loads(raw.get("input_output", "{}"))
        fn = io.get("fn_name")
        rows.append({
            "problem_id": str(int(raw["id"])),
            "difficulty": raw.get("difficulty"),
            "question": raw.get("question"),
            "starter_code": raw.get("starter_code") or "",
            "io_mode": "call" if fn else "stdin",
            "fn_name": fn,
            "inputs": io.get("inputs", []),
            "outputs": io.get("outputs", []),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    OUT_JSONL.write_text(text, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    manifest = {
        "name": "toolapps-eval",
        "rows": len(rows),
        "per_bucket": PER_BUCKET,
        "seed": SEED,
        "source": str(TEST),
        "source_revision": "21e74ddf8de1a21436da12e3e653065c5213e9d1",
        "by_difficulty": counts,
        "procedure": (
            "difficulty-bucketed pools sorted by numeric id; "
            "random.seed(42); shuffle; take first 250 (RLEF-Code procedure)."
        ),
        "comparability": (
            "procedure-compatible with RLEF-Code's fixed eval set, but built "
            "on codeparrot/apps test.jsonl without reference solutions; NOT "
            "per-problem identical to RLEF's 713. Do not compare item-by-item."
        ),
        "sha256": digest,
        "problem_ids_by_difficulty": {
            d: [r["problem_id"] for r in rows if r["difficulty"] == d]
            for d in DIFF_ORDER
        },
    }
    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows), "by_difficulty": counts,
        "jsonl": str(OUT_JSONL), "manifest": str(OUT_MANIFEST),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
