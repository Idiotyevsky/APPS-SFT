#!/usr/bin/env python3
"""Re-grade protocol-v2 repair pairs and summarize target quality.

This is an offline audit only.  It does not alter the dataset or grader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synthesis.apps_loader import parse_apps_record_strict  # noqa: E402
from synthesis.grader import PrivateGrader  # noqa: E402
from synthesis.normalize import normalized_code_hash  # noqa: E402
from synthesis.sandbox import SandboxConfig, SandboxedExecutor  # noqa: E402

REPAIR_STATES = {
    "first_failure_direct_repair",
    "post_run_repair",
    "multiround_final_submit",
}
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "data/coding_sft_protocol_v2")
    parser.add_argument("--apps", type=Path,
                        default=ROOT / "data/raw/apps/train.jsonl")
    parser.add_argument("--episodes", type=Path,
                        default=ROOT / "data/sft_v5_final/episodes.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "sft/outputs/repair_quality_audit_v2")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def tool_call(item: dict) -> dict | None:
    if item.get("from") != "function_call":
        return None
    value = json.loads(item["value"])
    return value if isinstance(value, dict) else None


def changed_lines(old: str, new: str) -> int:
    matcher = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines())
    count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            count += max(i2 - i1, j2 - j1)
    return count


def token_edit_ratio(old: str, new: str) -> float:
    a, b = TOKEN_RE.findall(old), TOKEN_RE.findall(new)
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(a=a, b=b).ratio()


def load_pairs(dataset_dir: Path) -> list[dict]:
    rows = json.loads((dataset_dir / "train.json").read_text(encoding="utf-8"))
    manifest = {
        row["sample_id"]: row
        for row in map(json.loads, (dataset_dir / "protocol_warmup_manifest.jsonl")
                       .read_text(encoding="utf-8").splitlines())
    }
    pairs = []
    for row in rows:
        meta = manifest[row["sample_id"]]
        if meta["state_type"] not in REPAIR_STATES:
            continue
        submits = []
        for item in row["conversations"]:
            call = tool_call(item)
            if call and call.get("name") == "submit":
                submits.append(call["arguments"]["code"])
        if len(submits) < 2:
            raise RuntimeError(f"{row['sample_id']}: repair has fewer than 2 submits")
        old, new = submits[-2], submits[-1]
        pairs.append({
            "sample_id": row["sample_id"],
            "problem_id": str(meta["problem_id"]),
            "state_type": meta["state_type"],
            "difficulty": meta.get("difficulty"),
            "old_code": old,
            "new_code": new,
            "exact_duplicate": old == new,
            "changed_lines": changed_lines(old, new),
            "token_edit_ratio": token_edit_ratio(old, new),
        })
    if len(pairs) != 170:
        raise RuntimeError(f"expected 170 repair pairs, got {len(pairs)}")
    return pairs


def load_problems(path: Path, ids: set[str]) -> dict[str, dict]:
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            identifier = str(raw.get("id", raw.get("problem_id")))
            if identifier in ids:
                found[identifier] = parse_apps_record_strict(raw)
    missing = sorted(ids - found.keys())
    if missing:
        raise RuntimeError(f"missing raw APPS problems: {missing[:10]}")
    return found


def load_episode_metadata(path: Path, ids: set[str]) -> dict[str, dict]:
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            identifier = str(episode.get("id"))
            if identifier in ids:
                found[identifier] = episode["metadata"]
    missing = sorted(ids - found.keys())
    if missing:
        raise RuntimeError(f"missing source episodes: {missing[:10]}")
    return found


def grade_job(payload: tuple[str, str, dict]) -> tuple[str, dict]:
    key, code, problem = payload
    grader = PrivateGrader(SandboxedExecutor(
        SandboxConfig(timeout_sec=3, memory_mb=512, backend="local")))
    spec = problem["input_output"]
    mode = "call" if spec.get("fn_name") else "stdin"
    result = grader.grade(code, spec["inputs"], spec["outputs"], mode,
                          spec.get("fn_name"))
    return key, {
        "status": result.status,
        "passed": result.passed,
        "total": result.total,
        "pass_rate": result.pass_rate,
        "accepted": result.accepted,
        "failing_input": result.failing_input,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    den = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / den if den else None


def summarize(rows: list[dict]) -> dict:
    delta = [row["delta_pass_rate"] for row in rows]
    lines = [row["changed_lines"] for row in rows]
    ratios = [row["token_edit_ratio"] for row in rows]
    result = {
        "n": len(rows),
        "improved_nonaccepted": sum(r["outcome"] == "improved" for r in rows),
        "unchanged": sum(r["outcome"] == "unchanged" for r in rows),
        "regressed": sum(r["outcome"] == "regressed" for r in rows),
        "accepted": sum(r["new_grade"]["accepted"] for r in rows),
        "improved_including_accepted": sum(r["delta_pass_rate"] > 0 for r in rows),
        "exact_duplicates": sum(r["exact_duplicate"] for r in rows),
        "mean_delta_pass_rate": statistics.mean(delta) if delta else None,
        "median_delta_pass_rate": statistics.median(delta) if delta else None,
        "median_changed_lines": statistics.median(lines) if lines else None,
        "changed_lines_p95": percentile(lines, .95),
        "median_token_edit_ratio": statistics.median(ratios) if ratios else None,
        "token_edit_ratio_p95": percentile(ratios, .95),
        "pearson_delta_vs_changed_lines": pearson(delta, lines),
        "pearson_delta_vs_token_edit_ratio": pearson(delta, ratios),
        "old_mean_pass_rate": statistics.mean(r["old_grade"]["pass_rate"] for r in rows),
        "new_mean_pass_rate": statistics.mean(r["new_grade"]["pass_rate"] for r in rows),
    }
    return result


def main() -> int:
    args = parse_args()
    pairs = load_pairs(args.dataset_dir)
    problems = load_problems(args.apps, {p["problem_id"] for p in pairs})
    episodes = load_episode_metadata(
        args.episodes, {p["problem_id"] for p in pairs})

    unique = {}
    for pair in pairs:
        for side in ("old", "new"):
            code = pair[f"{side}_code"]
            digest = hashlib.sha256(code.encode()).hexdigest()
            key = f"{pair['problem_id']}:{digest}"
            unique[key] = (key, code, problems[pair["problem_id"]])

    grades = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(grade_job, payload) for payload in unique.values()]
        for index, future in enumerate(as_completed(futures), 1):
            key, grade = future.result()
            grades[key] = grade
            if index % 25 == 0 or index == len(futures):
                print(f"graded {index}/{len(futures)} unique problem-code pairs", flush=True)

    details = []
    for pair in pairs:
        row = dict(pair)
        for side in ("old", "new"):
            digest = hashlib.sha256(pair[f"{side}_code"].encode()).hexdigest()
            row[f"{side}_grade"] = grades[f"{pair['problem_id']}:{digest}"]
        row["delta_pass_rate"] = row["new_grade"]["pass_rate"] - row["old_grade"]["pass_rate"]
        new_hash = normalized_code_hash(row["new_code"])
        source = episodes[row["problem_id"]]
        row["new_is_verified_reference"] = (
            new_hash == source["reference"]["code_hash"])
        row["new_is_source_final_code"] = new_hash == source["final_code_hash"]
        if row["new_grade"]["accepted"]:
            row["outcome"] = "accepted"
        elif row["delta_pass_rate"] > 0:
            row["outcome"] = "improved"
        elif row["delta_pass_rate"] < 0:
            row["outcome"] = "regressed"
        else:
            row["outcome"] = "unchanged"
        details.append(row)

    by_state = {
        state: summarize([row for row in details if row["state_type"] == state])
        for state in sorted(REPAIR_STATES)
    }
    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "apps_path": str(args.apps.resolve()),
        "repair_pair_count": len(details),
        "distinct_problems": len({r["problem_id"] for r in details}),
        "unique_grade_jobs": len(unique),
        "new_is_verified_reference": sum(
            row["new_is_verified_reference"] for row in details),
        "new_is_source_final_code": sum(
            row["new_is_source_final_code"] for row in details),
        "old_status": dict(Counter(row["old_grade"]["status"] for row in details)),
        "test_unit_count": dict(sorted(Counter(
            row["old_grade"]["total"] for row in details).items())),
        "single_test_unit_rows": sum(
            row["old_grade"]["total"] == 1 for row in details),
        "outcome_definition": (
            "accepted is a mutually-exclusive outcome; improved_nonaccepted excludes accepted, "
            "while improved_including_accepted is delta_pass_rate > 0"
        ),
        "diff_definition": (
            "changed_lines sums max(old_span,new_span) for non-equal line opcodes; "
            "token_edit_ratio is 1 - SequenceMatcher ratio over regex lexical tokens"
        ),
        "overall": summarize(details),
        "by_state": by_state,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "details.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
        encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pct(value):
        return "n/a" if value is None else f"{100*value:.2f}%"

    lines = [
        "# Repair Quality Audit — protocol v2",
        "",
        "Every prior candidate and repair target was re-executed against the existing",
        "private grader and the original APPS train tests. No stored pass claim was trusted.",
        "",
        "| state | N | Δpass > 0 | unchanged | regressed | accepted | mean Δpass | exact dup | median changed lines | median token edit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for state in ["first_failure_direct_repair", "post_run_repair", "multiround_final_submit"]:
        s = by_state[state]
        lines.append(
            f"| {state} | {s['n']} | {s['improved_including_accepted']} | {s['unchanged']} | "
            f"{s['regressed']} | {s['accepted']} | {s['mean_delta_pass_rate']:.4f} | "
            f"{s['exact_duplicates']} | {s['median_changed_lines']:.1f} | "
            f"{pct(s['median_token_edit_ratio'])} |")
    s = summary["overall"]
    lines.extend([
        f"| **overall** | **{s['n']}** | **{s['improved_including_accepted']}** | **{s['unchanged']}** | "
        f"**{s['regressed']}** | **{s['accepted']}** | **{s['mean_delta_pass_rate']:.4f}** | "
        f"**{s['exact_duplicates']}** | **{s['median_changed_lines']:.1f}** | "
        f"**{pct(s['median_token_edit_ratio'])}** |",
        "",
        "`accepted` overlaps with `Δpass > 0`; all accepted targets improved on their prior candidate.",
        "",
        "## Additional diagnostics",
        "",
        f"- Distinct problems: {summary['distinct_problems']}",
        f"- Unique grader jobs after exact cache deduplication: {summary['unique_grade_jobs']}",
        f"- Targets equal verified reference (normalized hash): {summary['new_is_verified_reference']}/{len(details)}",
        f"- Prior candidate statuses: {summary['old_status']}",
        f"- Rows with one APPS test unit: {summary['single_test_unit_rows']}/{len(details)}",
        f"- Old mean pass rate: {s['old_mean_pass_rate']:.4f}",
        f"- New mean pass rate: {s['new_mean_pass_rate']:.4f}",
        f"- Δpass vs changed-lines Pearson r: {s['pearson_delta_vs_changed_lines']}",
        f"- Δpass vs token-edit-ratio Pearson r: {s['pearson_delta_vs_token_edit_ratio']}",
        "",
        "Definitions are recorded in summary.json.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
