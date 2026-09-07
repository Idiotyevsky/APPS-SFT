#!/usr/bin/env python3
"""Align our eval set with RLEF-Code's fixed 713-problem held-out set.

RLEF eval set (repo: tarunbeerelli/RLEF-Code) is generated deterministically
(APPS test loader + seed 42, per-difficulty shuffle, cap 250; competition
capped by its own verification at 213). We cannot bit-exactly regenerate it
without their processed directory dataset, so this script consumes THEIR
artifact directly:

  * --apps-eval  <rlef data/apps_eval.jsonl>   (rows with problem_id/difficulty)
  * --ids-file   <json list or jsonl of {"problem_id": ...}> (any of the above)

and emits OUR canonical eval artifact under data/eval/:
  rlef_713.jsonl                 rows sourced from our local APPS test.jsonl
  rlef_713_problem_ids.json      per-difficulty id manifest + checks
Inputs are matched by numeric id against data/raw/apps/test.jsonl (same pinned
revision). Any RLEF id missing locally is reported loudly.

Outputs are only written when every provided id resolves locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from synthesis.io_utils import atomic_write_json, atomic_write_jsonl  # noqa: E402

TEST_PATH = ROOT_DIR / "data/raw/apps/test.jsonl"
OUT_DIR = ROOT_DIR / "data/eval"
OUT_JSONL = OUT_DIR / "rlef_713.jsonl"
OUT_MANIFEST = OUT_DIR / "rlef_713_problem_ids.json"


def _norm(value) -> str:
    try:
        return str(int(str(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _collect_ids(apps_eval: Path | None, ids_file: Path | None) -> list[str]:
    if apps_eval is not None:
        rows = [json.loads(line) for line in apps_eval.open(encoding="utf-8")]
        ids = []
        for row in rows:
            key = row.get("problem_id", row.get("id"))
            if key is not None:
                ids.append(_norm(key))
        return ids
    if ids_file is not None:
        text = ids_file.read_text(encoding="utf-8").strip()
        value = json.loads(text)
        if isinstance(value, list):
            return [_norm(v) for v in value]
        if isinstance(value, dict):
            value = value.get("problem_ids", value.get("ids", []))
            return [_norm(v) for v in value]
        return []
    raise ValueError("provide --apps-eval or --ids-file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps-eval", type=Path, default=None,
                        help="RLEF data/apps_eval.jsonl path")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="json list / jsonl of problem ids")
    parser.add_argument("--expected", default="250,250,213",
                        help="expected per-difficulty counts (intro,interview,comp)")
    args = parser.parse_args()

    ids = _collect_ids(args.apps_eval, args.ids_file)
    if not ids:
        raise SystemExit("no ids collected; check input format")
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise SystemExit(f"input contains {dup} duplicate ids")

    local: dict[str, dict] = {}
    for line in open(TEST_PATH, encoding="utf-8"):
        row = json.loads(line)
        local[_norm(row.get("id", row.get("problem_id")))] = row

    missing = [pid for pid in ids if pid not in local]
    if missing:
        raise SystemExit(
            f"{len(missing)} ids missing from local test.jsonl: {missing[:10]}"
        )

    rows = []
    for pid in ids:
        raw = local[pid]
        io = json.loads(raw.get("input_output", "{}"))
        rows.append({
            "problem_id": pid,
            "difficulty": raw.get("difficulty"),
            "question": raw.get("question"),
            "starter_code": raw.get("starter_code") or "",
            "io_mode": "call" if io.get("fn_name") else "stdin",
            "fn_name": io.get("fn_name"),
            "inputs": io.get("inputs", []),
            "outputs": io.get("outputs", []),
        })

    by_diff = Counter(str(r["difficulty"]) for r in rows)
    expected = [int(v) for v in args.expected.split(",")]
    diff_names = ["introductory", "interview", "competition"]
    wanted = dict(zip(diff_names, expected))
    warnings = [
        f"{d}: expected {wanted[d]}, got {by_diff.get(d, 0)}"
        for d in diff_names if by_diff.get(d, 0) != wanted[d]
    ]
    for warning in warnings:
        print("WARN:", warning, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(OUT_JSONL, rows)
    manifest = {
        "aligns_with": "tarunbeerelli/RLEF-Code fixed 713 held-out set",
        "source": "data/raw/apps/test.jsonl (pinned revision 21e74dd…)",
        "input_artifact": str(args.apps_eval or args.ids_file),
        "total": len(rows),
        "by_difficulty": dict(by_diff),
        "problem_ids_by_difficulty": {
            d: [r["problem_id"] for r in rows if r["difficulty"] == d]
            for d in diff_names
        },
        "warnings": warnings,
    }
    atomic_write_json(OUT_MANIFEST, manifest)
    print(json.dumps({
        "rows": len(rows), "by_difficulty": dict(by_diff),
        "jsonl": str(OUT_JSONL), "manifest": str(OUT_MANIFEST),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
