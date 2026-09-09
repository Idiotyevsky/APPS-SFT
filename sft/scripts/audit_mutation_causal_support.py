#!/usr/bin/env python3
"""Audit whether each repair target is causally supported by its current input.

For every synthetic-multi repair prefix, recover the exact AST mutation set by
re-enumerating edits on the verified reference and matching the current
candidate hash.  Then undo every non-empty subset and execute it on the same
failing/probe input visible immediately before the repair target.

This script is read-only with respect to datasets and uses the existing local
grader/runtime unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synthesis.apps_loader import parse_apps_record_strict  # noqa: E402
from synthesis.grader import PrivateGrader  # noqa: E402
from synthesis.mutation import (  # noqa: E402
    apply_edits, enumerate_supported_single_edits,
)
from synthesis.normalize import (  # noqa: E402
    canonical_json, compare_output, normalize_stdout, normalized_code_hash,
)
from synthesis.sandbox import SandboxConfig, SandboxedExecutor  # noqa: E402

REPAIR_STATES = {
    "first_failure_direct_repair",
    "post_run_repair",
    "multiround_final_submit",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "data/coding_sft_protocol_v2")
    parser.add_argument("--episodes", type=Path,
                        default=ROOT / "data/sft_v5_final/episodes.jsonl")
    parser.add_argument("--apps", type=Path,
                        default=ROOT / "data/raw/apps/train.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "sft/outputs/mutation_causal_support_audit_v2")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def decode_call(item: dict) -> dict | None:
    if item.get("from") != "function_call":
        return None
    value = json.loads(item["value"])
    return value if isinstance(value, dict) else None


def record_key(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def read_inputs(dataset_dir: Path, episodes_path: Path,
                apps_path: Path) -> list[dict]:
    train = json.loads((dataset_dir / "train.json").read_text(encoding="utf-8"))
    manifest = {
        row["sample_id"]: row
        for row in map(json.loads, (dataset_dir / "protocol_warmup_manifest.jsonl")
                       .read_text(encoding="utf-8").splitlines())
    }
    selected = []
    ids = set()
    for row in train:
        meta = manifest[row["sample_id"]]
        if meta["state_type"] not in REPAIR_STATES:
            continue
        pid = str(meta["problem_id"])
        ids.add(pid)
        selected.append((row, meta, pid))

    episodes = {}
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            pid = str(episode.get("id"))
            if pid in ids:
                episodes[pid] = episode

    multi_ids = {
        pid for pid, episode in episodes.items()
        if episode["metadata"]["candidate"]["origin"] == "synthetic_multi"
    }
    problems = {}
    with apps_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            pid = str(raw.get("id", raw.get("problem_id")))
            if pid in multi_ids:
                problems[pid] = parse_apps_record_strict(raw)
    if multi_ids - problems.keys():
        raise RuntimeError(f"missing raw problems: {sorted(multi_ids-problems.keys())[:10]}")

    jobs = []
    for row, meta, pid in selected:
        episode = episodes[pid]
        if episode["metadata"]["candidate"]["origin"] != "synthetic_multi":
            continue
        conversations = row["conversations"]
        target = decode_call(conversations[-1])
        if not target or target.get("name") != "submit":
            raise RuntimeError(f"{row['sample_id']}: repair target is not submit")
        reference = target["arguments"]["code"]

        previous_submit = None
        paired_call = None
        paired_observation = None
        pending = None
        for item in conversations[:-1]:
            call = decode_call(item)
            if call:
                pending = call
                if call.get("name") == "submit":
                    previous_submit = call["arguments"]["code"]
            elif item.get("from") == "observation" and pending is not None:
                paired_call = pending
                paired_observation = json.loads(item["value"])
                pending = None
        if previous_submit is None or paired_call is None:
            raise RuntimeError(f"{row['sample_id']}: incomplete repair history")
        if paired_call["name"] == "run_candidate":
            visible_input = paired_call["arguments"]["input"]
        elif paired_call["name"] == "submit":
            visible_input = paired_observation.get("failing_input")
        else:
            visible_input = None
        if not isinstance(visible_input, str):
            raise RuntimeError(f"{row['sample_id']}: no visible failing/probe input")

        mutation = episode["metadata"].get("mutation") or {}
        jobs.append({
            "sample_id": row["sample_id"],
            "problem_id": pid,
            "state_type": meta["state_type"],
            "difficulty": meta.get("difficulty"),
            "reference": reference,
            "candidate": previous_submit,
            "visible_input": visible_input,
            "visible_action": paired_call["name"],
            "visible_observation_status": paired_observation.get("status"),
            "original_bug_count": mutation.get("bug_count"),
            "metadata_records": mutation.get("edits") or [],
            "problem": problems[pid],
        })
    if len(jobs) != 112:
        raise RuntimeError(f"expected 112 synthetic-multi repairs, got {len(jobs)}")
    return jobs


def recovery_candidates(job: dict) -> tuple[list[list[Any]], dict]:
    reference = job["reference"]
    expected_hash = normalized_code_hash(job["candidate"])
    metadata_counts = Counter(record_key(record) for record in job["metadata_records"])
    enumerated = enumerate_supported_single_edits(reference)
    filtered = [
        edit for edit in enumerated
        if record_key(edit.record.to_dict()) in metadata_counts
    ]
    expected_current_count = int(job["original_bug_count"])
    if job["state_type"] == "multiround_final_submit":
        expected_current_count -= 1
    matches = []
    considered = 0
    for edits in itertools.combinations(filtered, expected_current_count):
        counts = Counter(record_key(edit.record.to_dict()) for edit in edits)
        if any(count > metadata_counts[key] for key, count in counts.items()):
            continue
        considered += 1
        try:
            code, _ = apply_edits(reference, edits)
        except (SyntaxError, ValueError, TypeError):
            continue
        if normalized_code_hash(code) == expected_hash:
            matches.append(list(edits))
    # Stable keys distinguish different AST locations even when records render
    # identically.  Do not silently select one of multiple reconstructions.
    unique = {}
    for edits in matches:
        key = tuple(sorted(edit.stable_key for edit in edits))
        unique[key] = edits
    return list(unique.values()), {
        "enumerated_edit_count": len(enumerated),
        "record_filtered_edit_count": len(filtered),
        "combination_count_considered": considered,
        "expected_current_bug_count": expected_current_count,
    }


def public_run(grader: PrivateGrader, code: str, input_text: str,
               mode: str, fn_name: str | None) -> tuple[dict, Any]:
    result, actual = grader.run_candidate(code, input_text, mode, fn_name)
    return {
        "status": result.status,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "exit_code": result.exit_code,
        "truncated": result.truncated,
    }, actual


def positional_agreement(left: list[Any], right: list[Any]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return sum(a == b for a, b in zip(left, right)) / denominator


def structured_leaves(value: Any, path: tuple = ()) -> dict[tuple, str]:
    if isinstance(value, dict):
        leaves = {}
        for key in sorted(value, key=str):
            leaves.update(structured_leaves(value[key], path + (("key", str(key)),)))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(structured_leaves(item, path + (("index", index),)))
        return leaves
    return {path: canonical_json(value)}


def partial_agreement(actual: Any, reference: Any, mode: str,
                      status: str) -> dict[str, float]:
    if mode == "stdin":
        actual_text = str(actual or "")
        reference_text = str(reference or "")
        line_score = positional_agreement(
            normalize_stdout(actual_text), normalize_stdout(reference_text))
        token_score = positional_agreement(
            actual_text.split(), reference_text.split())
        return {
            "primary": token_score,
            "normalized_line_agreement": line_score,
            "normalized_token_agreement": token_score,
        }
    if status != "ok" or actual is None:
        return {"primary": 0.0, "structured_leaf_agreement": 0.0}
    left, right = structured_leaves(actual), structured_leaves(reference)
    keys = set(left) | set(right)
    score = (sum(left.get(key) == right.get(key) for key in keys) / len(keys)
             if keys else 1.0)
    return {"primary": score, "structured_leaf_agreement": score}


def audit_one(job: dict) -> dict:
    row = {key: value for key, value in job.items()
           if key not in {"problem", "reference", "candidate", "visible_input"}}
    row["candidate_hash"] = normalized_code_hash(job["candidate"])
    row["reference_hash"] = normalized_code_hash(job["reference"])
    row["private_test_unit_count"] = len(
        job["problem"]["input_output"]["inputs"])
    row["visible_input_sha256"] = hashlib.sha256(
        job["visible_input"].encode()).hexdigest()
    matches, recovery = recovery_candidates(job)
    row["recovery"] = recovery
    row["reconstruction_count"] = len(matches)
    if not matches:
        row.update({
            "category": "no_reconstructed_subset_explains_X",
            "exclusion_reason": "reconstruction_failed",
            "minimal_supported_undo_size": None,
            "target_overreach": None,
            "minimal_supported_undo_sets": [],
        })
        return row
    if len(matches) > 1:
        row.update({
            "category": "ambiguous_reconstruction",
            "exclusion_reason": "multiple_edit_sets_match_candidate_hash",
            "minimal_supported_undo_size": None,
            "target_overreach": None,
            "minimal_supported_undo_sets": [],
        })
        return row

    edits = matches[0]
    current_k = len(edits)
    row["current_bug_count"] = current_k
    row["recovered_records"] = [edit.record.to_dict() for edit in edits]
    metadata_counts = Counter(record_key(r) for r in job["metadata_records"])
    recovered_counts = Counter(record_key(e.record.to_dict()) for e in edits)
    row["records_bijective_or_subset"] = all(
        count <= metadata_counts[key] for key, count in recovered_counts.items())
    expected_k = recovery["expected_current_bug_count"]
    row["bug_count_sanity_passed"] = current_k == expected_k
    if not row["records_bijective_or_subset"] or not row["bug_count_sanity_passed"]:
        row.update({
            "category": "reconstruction_sanity_failure",
            "exclusion_reason": "record_or_bug_count_mismatch",
            "minimal_supported_undo_size": None,
            "target_overreach": None,
            "minimal_supported_undo_sets": [],
        })
        return row

    spec = job["problem"]["input_output"]
    mode = "call" if spec.get("fn_name") else "stdin"
    fn_name = spec.get("fn_name")
    grader = PrivateGrader(SandboxedExecutor(
        SandboxConfig(timeout_sec=3, memory_mb=512, backend="local")))
    reference_run, reference_actual = public_run(
        grader, job["reference"], job["visible_input"], mode, fn_name)
    candidate_run, candidate_actual = public_run(
        grader, job["candidate"], job["visible_input"], mode, fn_name)
    row["reference_run"] = reference_run
    row["candidate_run"] = candidate_run
    reference_ok = reference_run["status"] == "ok"
    baseline_matches = (
        reference_ok and candidate_run["status"] == "ok"
        and compare_output(candidate_actual, reference_actual, mode)
    )
    row["reference_run_ok"] = reference_ok
    row["baseline_matches_reference"] = baseline_matches
    row["baseline_partial_score"] = partial_agreement(
        candidate_actual, reference_actual, mode, candidate_run["status"])
    if not reference_ok:
        row.update({
            "category": "no_reconstructed_subset_explains_X",
            "exclusion_reason": "reference_execution_failed",
            "minimal_supported_undo_size": None,
            "target_overreach": None,
            "minimal_supported_undo_sets": [],
        })
        return row
    if baseline_matches:
        row.update({
            "category": "no_reconstructed_subset_explains_X",
            "exclusion_reason": "baseline_already_matches_reference",
            "minimal_supported_undo_size": None,
            "target_overreach": None,
            "minimal_supported_undo_sets": [],
        })
        return row

    subset_rows = []
    indices = tuple(range(current_k))
    for size in range(1, current_k + 1):
        for undone in itertools.combinations(indices, size):
            undone_set = set(undone)
            remaining = [edit for index, edit in enumerate(edits)
                         if index not in undone_set]
            code, _ = apply_edits(job["reference"], remaining)
            run, actual = public_run(
                grader, code, job["visible_input"], mode, fn_name)
            fixes = run["status"] == "ok" and compare_output(
                actual, reference_actual, mode)
            partial_score = partial_agreement(
                actual, reference_actual, mode, run["status"])
            subset_rows.append({
                "undo_indices": list(undone),
                "undo_size": size,
                "is_proper_subset": size < current_k,
                "fixes_visible_input": fixes,
                "partial_score": partial_score,
                "delta_partial_score": (
                    partial_score["primary"]
                    - row["baseline_partial_score"]["primary"]),
                "run": run,
            })
    row["subset_results"] = subset_rows
    supported = [entry for entry in subset_rows if entry["fixes_visible_input"]]
    if not supported:
        row.update({
            "category": "no_reconstructed_subset_explains_X",
            "exclusion_reason": "even_full_undo_does_not_match_reference",
            "minimal_supported_undo_size": None,
            "target_overreach": None,
            "minimal_supported_undo_sets": [],
        })
        return row
    minimum = min(entry["undo_size"] for entry in supported)
    minimal = [entry["undo_indices"] for entry in supported
               if entry["undo_size"] == minimum]
    row["minimal_supported_undo_size"] = minimum
    row["minimal_supported_undo_sets"] = minimal
    row["target_overreach"] = current_k - minimum
    row["exclusion_reason"] = None
    if minimum == 1 and len(minimal) == 1:
        row["category"] = "unique_single_cause"
    elif minimum == 1:
        row["category"] = "multiple_single_causes"
    elif minimum < current_k:
        row["category"] = "requires_multiple_edits"
    else:
        # The full reference works, but no proper partial repair explains X.
        row["category"] = "no_reconstructed_subset_explains_X"
        row["exclusion_reason"] = "full_undo_required_no_proper_subset"
    return row


def stats(rows: list[dict]) -> dict:
    supported = [r for r in rows if r.get("target_overreach") is not None]
    overreach = [r["target_overreach"] for r in supported]
    current_multi = [r for r in supported if r.get("current_bug_count", 0) > 1]
    with_proper_improvement = [
        r for r in current_multi
        if any(s["is_proper_subset"] and s["delta_partial_score"] > 1e-12
               for s in r.get("subset_results", []))
    ]
    with_single_improvement = [
        r for r in current_multi
        if any(s["undo_size"] == 1 and s["delta_partial_score"] > 1e-12
               for s in r.get("subset_results", []))
    ]
    best_proper_deltas = [
        max((s["delta_partial_score"] for s in r.get("subset_results", [])
             if s["is_proper_subset"]), default=0.0)
        for r in current_multi
    ]
    return {
        "n": len(rows),
        "categories": dict(Counter(r["category"] for r in rows)),
        "exclusion_reasons": dict(Counter(
            r["exclusion_reason"] for r in rows if r.get("exclusion_reason"))),
        "reconstruction_unique": sum(r["reconstruction_count"] == 1 for r in rows),
        "reconstruction_failed": sum(r["reconstruction_count"] == 0 for r in rows),
        "reconstruction_ambiguous": sum(r["reconstruction_count"] > 1 for r in rows),
        "causally_evaluable": len(supported),
        "mean_target_overreach": statistics.mean(overreach) if overreach else None,
        "p_overreach_gt_0": (
            sum(value > 0 for value in overreach) / len(overreach)
            if overreach else None),
        "p_overreach_ge_2": (
            sum(value >= 2 for value in overreach) / len(overreach)
            if overreach else None),
        "target_overreach_distribution": dict(Counter(overreach)),
        "current_bug_count": dict(Counter(
            r.get("current_bug_count") for r in rows
            if r.get("current_bug_count") is not None)),
        "private_test_unit_count": dict(Counter(
            r.get("private_test_unit_count") for r in rows
            if r.get("private_test_unit_count") is not None)),
        "single_test_unit_rows": sum(
            r.get("private_test_unit_count") == 1 for r in rows),
        "current_multi_rows": len(current_multi),
        "proper_subset_improves": len(with_proper_improvement),
        "p_proper_subset_improves": (
            len(with_proper_improvement) / len(current_multi)
            if current_multi else None),
        "single_undo_improves": len(with_single_improvement),
        "p_single_undo_improves": (
            len(with_single_improvement) / len(current_multi)
            if current_multi else None),
        "mean_best_proper_delta": (
            statistics.mean(best_proper_deltas) if best_proper_deltas else None),
        "mean_baseline_partial_score": (
            statistics.mean(r["baseline_partial_score"]["primary"]
                            for r in supported) if supported else None),
    }


def main() -> int:
    args = arguments()
    jobs = read_inputs(args.dataset_dir, args.episodes, args.apps)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_one, job): job["sample_id"] for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "sample_id": futures[future],
                    "category": "audit_infrastructure_error",
                    "exclusion_reason": f"{type(exc).__name__}: {exc}",
                    "reconstruction_count": 0,
                    "target_overreach": None,
                })
            if index % 10 == 0 or index == len(futures):
                print(f"audited {index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: row["sample_id"])
    by_state = {
        state: stats([r for r in results if r.get("state_type") == state])
        for state in sorted(REPAIR_STATES)
    }
    summary = {
        "definition": {
            "correct_on_X": (
                "candidate subset run must have status=ok and output compare equal "
                "to the verified reference run on the identical visible input X"
            ),
            "target_overreach": "current_bug_count - minimal_supported_undo_size",
            "no_reconstructed_subset_explains_X": (
                "includes a valid reconstruction whose full undo is required because "
                "no proper subset repairs X; reconstruction/audit failures are separately "
                "identified by exclusion_reason"
            ),
            "multiround": (
                "audits the partial candidate immediately preceding the final target; "
                "current_bug_count is original_bug_count - 1"
            ),
        },
        "overall": stats(results),
        "by_state": by_state,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "details.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
        encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    categories = [
        "unique_single_cause", "multiple_single_causes",
        "requires_multiple_edits", "no_reconstructed_subset_explains_X",
        "ambiguous_reconstruction", "reconstruction_sanity_failure",
        "audit_infrastructure_error",
    ]
    lines = [
        "# Mutation-grounded causal-support audit",
        "",
        "All subset candidates were executed on the exact input visible immediately",
        "before the repair target. Correctness requires both successful execution and",
        "output equality with the verified reference on that same input.",
        "",
        "| state | N | unique single | multiple single | requires multiple | no proper subset | reconstruction ambiguous/fail | mean overreach | P(overreach>0) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for state in ["first_failure_direct_repair", "post_run_repair",
                  "multiround_final_submit"]:
        s = by_state[state]
        c = Counter(s["categories"])
        bad = sum(c[name] for name in categories[4:]) + s["reconstruction_failed"]
        mean = "n/a" if s["mean_target_overreach"] is None else f"{s['mean_target_overreach']:.3f}"
        prob = "n/a" if s["p_overreach_gt_0"] is None else f"{100*s['p_overreach_gt_0']:.1f}%"
        lines.append(
            f"| {state} | {s['n']} | {c['unique_single_cause']} | "
            f"{c['multiple_single_causes']} | {c['requires_multiple_edits']} | "
            f"{c['no_reconstructed_subset_explains_X']} | {bad} | {mean} | {prob} |")
    s = summary["overall"]
    c = Counter(s["categories"])
    bad = sum(c[name] for name in categories[4:]) + s["reconstruction_failed"]
    mean = "n/a" if s["mean_target_overreach"] is None else f"{s['mean_target_overreach']:.3f}"
    prob = "n/a" if s["p_overreach_gt_0"] is None else f"{100*s['p_overreach_gt_0']:.1f}%"
    lines.extend([
        f"| **overall** | **{s['n']}** | **{c['unique_single_cause']}** | "
        f"**{c['multiple_single_causes']}** | **{c['requires_multiple_edits']}** | "
        f"**{c['no_reconstructed_subset_explains_X']}** | **{bad}** | **{mean}** | **{prob}** |",
        "",
        "## Reconstruction and overreach",
        "",
        f"- Unique reconstruction: {s['reconstruction_unique']}/{s['n']}",
        f"- Failed reconstruction: {s['reconstruction_failed']}/{s['n']}",
        f"- Ambiguous reconstruction: {s['reconstruction_ambiguous']}/{s['n']}",
        f"- Target-overreach distribution: {s['target_overreach_distribution']}",
        f"- Rows with one APPS test unit: {s['single_test_unit_rows']}/{s['n']}",
        f"- Current multi-bug rows: {s['current_multi_rows']}",
        f"- Any proper subset improves normalized output: "
        f"{s['proper_subset_improves']}/{s['current_multi_rows']}",
        f"- Any single undo improves normalized output: "
        f"{s['single_undo_improves']}/{s['current_multi_rows']}",
        f"- Mean best proper-subset score gain: "
        f"{s['mean_best_proper_delta']}",
        f"- P(target_overreach >= 2): " + (
            "n/a" if s["p_overreach_ge_2"] is None
            else f"{100*s['p_overreach_ge_2']:.1f}%"),
        "",
        "See summary.json for definitions and details.jsonl for every reconstruction,",
        "subset execution, status, and hashed output.",
        "",
        "`no proper subset` is not an infrastructure failure: it means only the full",
        "undo set restores reference behavior on X. One APPS test unit may itself be a",
        "single stdin payload containing multiple problem test cases.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
