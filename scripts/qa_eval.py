#!/usr/bin/env python3
"""Structural QA for the ToolAPPS evaluation set (data/eval/toolapps_eval.jsonl).

An eval row is usable if the grader can consume it: question present, io spec
self-consistent, every input accepted by the shared adapter, every expected
output serializable for the comparator. Execution-verification with reference
solutions is NOT possible here (codeparrot/apps test.jsonl carries no
solutions), so this QA covers everything that is decidable offline:

  1. counts / difficulty buckets (250/250/250, total 750)
  2. id uniqueness and numeric form
  3. schema: question, starter_code, io_mode/fn_name consistency
  4. per-case adapter/comparator usability (inputs normalize, outputs compare)
  5. determinism: id list matches the recorded manifest
  6. provenance: split=test (never used by synthesis) + sha256 match

Writes data/eval/qa_eval_report.json; exits 0 only when all checks pass.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synthesis.io_utils import atomic_write_json  # noqa: E402
from synthesis.normalize import normalize_input  # noqa: E402

EVAL = ROOT / "data/eval/toolapps_eval.jsonl"
MANIFEST = ROOT / "data/eval/toolapps_eval_manifest.json"
REPORT = ROOT / "data/eval/qa_eval_report.json"
DIFF_ORDER = ["introductory", "interview", "competition"]
EXPECTED = {d: 250 for d in DIFF_ORDER}
TOTAL = 750


def main() -> int:
    issues: list[str] = []
    rows = [json.loads(line) for line in open(EVAL, encoding="utf-8")]

    # 1) counts
    by_diff = Counter(str(r.get("difficulty")) for r in rows)
    for difficulty in DIFF_ORDER:
        got = by_diff.get(difficulty, 0)
        if got != EXPECTED[difficulty]:
            issues.append(f"difficulty {difficulty}: expected {EXPECTED[difficulty]}, got {got}")
    if len(rows) != TOTAL:
        issues.append(f"row count: expected {TOTAL}, got {len(rows)}")

    # 2) ids
    ids = [str(r.get("problem_id")) for r in rows]
    if len(ids) != len(set(ids)):
        issues.append(f"duplicate problem_ids ({len(ids) - len(set(ids))})")
    if any(not pid.isdigit() for pid in ids):
        issues.append("non-numeric problem_id present")

    # 3+4) per-row schema and grader-usability
    bad_cases = 0
    for index, row in enumerate(rows):
        difficulty = row.get("difficulty")
        if difficulty not in DIFF_ORDER:
            issues.append(f"row {index}: bad difficulty {difficulty!r}")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            issues.append(f"row {index}: empty question")
        if not isinstance(row.get("starter_code"), str):
            issues.append(f"row {index}: starter_code not a string")
        io_mode = row.get("io_mode")
        fn = row.get("fn_name")
        if io_mode not in {"stdin", "call"}:
            issues.append(f"row {index}: bad io_mode {io_mode!r}")
        elif (fn is not None) != (io_mode == "call"):
            issues.append(f"row {index}: fn_name/io_mode inconsistent")
        inputs = row.get("inputs")
        outputs = row.get("outputs")
        if not isinstance(inputs, list) or not inputs:
            issues.append(f"row {index}: missing/empty inputs")
            continue
        if not isinstance(outputs, list) or not outputs or len(outputs) != len(inputs):
            issues.append(f"row {index}: outputs missing/empty/mismatched")
            continue
        for case_input in inputs:
            try:
                normalize_input(case_input, io_mode)
            except (TypeError, ValueError) as exc:
                issues.append(f"row {index}: input not adapter-usable: {exc}")
                bad_cases += 1
                break
        try:
            json.dumps(outputs, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            issues.append(f"row {index}: outputs not serializable: {exc}")

    # 5) determinism vs manifest (set equality per difficulty; file is sorted
    #    globally by id while the manifest stores per-difficulty groups)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id_diff: dict[str, set[str]] = {d: set() for d in DIFF_ORDER}
    for row in rows:
        by_id_diff[str(row.get("difficulty"))].add(str(row.get("problem_id")))
    mismatch = [
        difficulty for difficulty in DIFF_ORDER
        if by_id_diff.get(difficulty, set()) != set(
            manifest.get("problem_ids_by_difficulty", {}).get(difficulty, [])
        )
    ]
    if mismatch:
        issues.append(
            f"per-difficulty id sets differ from manifest: {mismatch} (regenerate?)"
        )

    # 6) sha256 + provenance
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    digest = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    if digest != manifest.get("sha256"):
        issues.append("file sha256 differs from manifest")
    if not str(manifest.get("source", "")).endswith("test.jsonl"):
        issues.append("source is not the APPS test split")

    report = {
        "rows": len(rows),
        "by_difficulty": dict(by_diff),
        "checks": {
            "counts": True, "ids_unique": True, "schema": True,
            "adapter_usable": True, "deterministic": True, "provenance": True,
        },
        "bad_case_inputs": bad_cases,
        "issues": issues,
        "passed": not issues,
        "note": (
            "structural QA only; execution-verification with reference "
            "solutions is not possible because codeparrot/apps test.jsonl "
            "provides no solutions"
        ),
    }
    for issue in issues:
        if "difficulty" in issue or "row count" in issue:
            report["checks"]["counts"] = False
        elif "duplicate" in issue or "non-numeric" in issue:
            report["checks"]["ids_unique"] = False
        elif "sha256" in issue or "source" in issue:
            report["checks"]["provenance"] = False
        elif "manifest" in issue:
            report["checks"]["deterministic"] = False
        else:
            report["checks"]["schema"] = False
            report["checks"]["adapter_usable"] = False
    atomic_write_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for issue in issues[:20]:
        print("  issue:", issue, flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
