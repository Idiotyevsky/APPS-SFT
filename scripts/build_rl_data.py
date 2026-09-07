#!/usr/bin/env python3
"""Build a standalone RL data folder from raw APPS train.

Produces:
  data/rl/
    splits.json          problem-level pool assignment (sft/rl/behavior_dev)
    README.md            schema + usage + information boundary
    manifest.json        counts by difficulty / io
    public/problems.jsonl      id, difficulty, io_mode, fn_name,
                               input_format_note, question, starter_code, url
    private/tests.jsonl        id, fn_name, inputs, outputs   (reward-only)
    private/references.jsonl   id, solutions                  (reward/audit-only)

Policy environments read ONLY public/; rewards and audits read private/.
No APPS test split is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from synthesis.apps_loader import parse_apps_record_strict, stream_apps_jsonl  # noqa: E402
from synthesis.io_utils import atomic_write_json, atomic_write_jsonl  # noqa: E402
from synthesis.splitter import difficulty_stratified_split  # noqa: E402

APPS = ROOT_DIR / "data/raw/apps/train.jsonl"
OUT = ROOT_DIR / "data/rl"


def main() -> int:
    records = [raw for _, raw in stream_apps_jsonl(APPS)]
    ids = []
    for raw in records:
        identifier = raw.get("id", raw.get("problem_id"))
        ids.append({
            "id": str(identifier),
            "difficulty": raw.get("difficulty") or "unknown",
        })
    splits = difficulty_stratified_split(ids, seed=42)
    rl_ids = set(splits["rl"])
    atomic_write_json(OUT / "splits.json", splits)

    public_rows, private_tests, private_refs = [], [], []
    count_by_diff = {}
    for raw in records:
        identifier = raw.get("id", raw.get("problem_id"))
        sid = str(identifier)
        if sid not in rl_ids:
            continue
        try:
            rec = parse_apps_record_strict(raw)
        except ValueError:
            continue
        spec = rec["input_output"]
        fn = spec.get("fn_name")
        io_mode = "call" if fn else "stdin"
        note = (
            "Pass one complete stdin string to run_candidate(input)."
            if io_mode == "stdin"
            else "Pass a canonical JSON argument list to run_candidate(input)."
        )
        public_rows.append({
            "id": sid,
            "difficulty": rec["difficulty"],
            "io_mode": io_mode,
            "fn_name": fn,
            "input_format_note": note,
            "question": rec["question"],
            "starter_code": rec.get("starter_code") or "",
            "url": rec.get("url"),
        })
        private_tests.append({
            "id": sid, "fn_name": fn,
            "io_mode": io_mode,
            "inputs": spec["inputs"],
            "outputs": spec["outputs"],
        })
        private_refs.append({
            "id": sid, "solutions": rec["solutions"],
        })
        count_by_diff[sid] = rec["difficulty"]

    atomic_write_jsonl(OUT / "public" / "problems.jsonl", public_rows)
    atomic_write_jsonl(OUT / "private" / "tests.jsonl", private_tests)
    atomic_write_jsonl(OUT / "private" / "references.jsonl", private_refs)

    from collections import Counter
    by_diff = Counter(count_by_diff.values())
    by_io = Counter(row["io_mode"] for row in public_rows)
    manifest = {
        "dataset": "codeparrot/apps",
        "split": "train",
        "pool": "rl",
        "problems": len(public_rows),
        "by_difficulty": dict(by_diff),
        "by_io": dict(by_io),
        "public_fields": [
            "id", "difficulty", "io_mode", "fn_name",
            "input_format_note", "question", "starter_code", "url",
        ],
        "private_fields": ["inputs", "outputs", "solutions"],
        "boundary": (
            "policy may only read public/problems.jsonl; private tests and "
            "references are reward/audit-only and must never enter context"
        ),
    }
    atomic_write_json(OUT / "manifest.json", manifest)
    (OUT / "README.md").write_text(
        "\n".join([
            "# RL data folder (from APPS train, rl pool)",
            "",
            f"- problems: {len(public_rows)} (difficulty {dict(by_diff)}, "
            f"io {dict(by_io)})",
            "- `public/problems.jsonl`: model-visible fields only.",
            "- `private/tests.jsonl` + `private/references.jsonl`: for the",
            "  reward/grader and audits only; never shown to the policy.",
            "- Tool contract is the same as SFT:",
            "  `run_candidate(input)` / `submit(code)`.",
            "- Reward signal: private grader pass rate / accepted over",
            "  `private/tests.jsonl`; keep a held-out portion for evaluation",
            "  to limit reward overfitting.",
            "- APPS test split is never used.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
