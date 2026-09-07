from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .io_utils import atomic_write_json, read_jsonl


def _counterfactual_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        item["metadata"]["counterfactual"]
        for item in episodes
        if item.get("metadata", {}).get("counterfactual") is not None
    ]
    utilities = [float(item.get("utility", 0.0)) for item in values]
    without = [float(item.get("without_run_rate", 0.0)) for item in values]
    with_run = [float(item.get("with_run_rate", 0.0)) for item in values]
    if not utilities:
        return {"count": 0, "utility_mean": None, "utility_min": None,
                "utility_max": None, "utility_values": []}
    return {
        "count": len(utilities),
        "utility_mean": mean(utilities),
        "utility_min": min(utilities),
        "utility_max": max(utilities),
        "utility_values": utilities,
        "without_run_rate_mean": mean(without),
        "with_run_rate_mean": mean(with_run),
    }


def build_manifest(
    output_dir: str | Path, target_count: int,
    quota_shortfalls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    episodes = (
        list(read_jsonl(root / "episodes.jsonl"))
        if (root / "episodes.jsonl").exists() else []
    )
    candidates = (
        list(read_jsonl(root / "candidates.jsonl"))
        if (root / "candidates.jsonl").exists() else []
    )
    rejected = (
        list(read_jsonl(root / "rejected.jsonl"))
        if (root / "rejected.jsonl").exists() else []
    )
    metadata = [row.get("metadata", {}) for row in episodes]
    by_origin = Counter(
        row.get("candidate", {}).get("origin") for row in metadata
    )
    by_behavior = Counter(
        (row.get("behavior_sequence") or [None])[0] for row in metadata
    )
    cross = Counter(
        str(row.get("candidate", {}).get("origin")) + "|" +
        str((row.get("behavior_sequence") or [None])[0])
        for row in metadata
    )
    by_difficulty = Counter(row.get("difficulty") for row in metadata)
    by_io_mode = Counter(row.get("io_mode") for row in metadata)
    by_bug = Counter(
        str((row.get("mutation") or {}).get("bug_count"))
        for row in metadata
        if row.get("candidate", {}).get("origin") == "synthetic_multi"
    )
    mutation_families = Counter()
    for row in metadata:
        for edit in (row.get("mutation") or {}).get("edits", []):
            family = edit.get("family")
            if family:
                mutation_families[family] += 1
    per_problem = Counter(str(row.get("id", row.get("problem_id"))) for row in metadata)
    reject_reasons = Counter(
        str(row.get("reject_reason", "unknown")) for row in rejected
    )
    by_reject_phase = Counter(
        str(row.get("phase", "unknown")) for row in rejected
    )
    candidate_rejected = sum(
        row.get("phase") == "candidate" for row in rejected
    )
    synthesis_rejected = sum(
        row.get("phase") == "synthesis" for row in rejected
    )
    sft_rows = (
        sum(1 for _ in read_jsonl(root / "sft_messages.jsonl"))
        if (root / "sft_messages.jsonl").exists() else 0
    )
    stage_metrics = {}
    metrics_path = root / "run_metrics.json"
    if metrics_path.exists():
        try:
            stage_metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            stage_metrics = {"invalid": True}
    resolved = {}
    resolved_path = root / "resolved_config.json"
    if resolved_path.exists():
        try:
            resolved = json.loads(
                resolved_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            resolved = {"invalid": True}
    # A synthesis reject is a candidate that was valid before final sampling
    # but was ambiguous, duplicated, or otherwise not accepted. This gives a
    # conservative, file-derived pre-sampling count.
    valid_before_sampling = len(episodes) + synthesis_rejected
    manifest = {
        "target_episodes": target_count,
        "accepted_episodes": len(episodes),
        "sft_jsonl_rows": sft_rows,
        "attempted_candidates": len(candidates) + candidate_rejected,
        "valid_before_sampling": valid_before_sampling,
        "accepted_after_sampling": len(episodes),
        "rejected_candidates": len(rejected),
        "rejected_candidate_rows": candidate_rejected,
        "rejected_synthesis_rows": synthesis_rejected,
        "by_reject_phase": dict(by_reject_phase),
        "reject_reasons": dict(reject_reasons),
        "by_candidate_origin": dict(by_origin),
        "by_behavior": dict(by_behavior),
        "origin_by_behavior": dict(cross),
        "by_difficulty": dict(by_difficulty),
        "by_io_mode": dict(by_io_mode),
        "by_bug_count": dict(by_bug),
        "mutation_families": dict(mutation_families),
        "per_problem_trajectory_count": dict(per_problem),
        "counterfactual": _counterfactual_summary(episodes),
        "quota_targets": resolved.get("quotas", {}),
        "quota_shortfalls": quota_shortfalls or [],
        "stage_metrics": stage_metrics,
        "config_hash": resolved.get("config_hash"),
        "code_revision": resolved.get("code_revision"),
    }
    atomic_write_json(root / "dataset_manifest.json", manifest)
    return manifest
