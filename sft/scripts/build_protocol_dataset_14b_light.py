#!/usr/bin/env python3
"""Build the deterministic protocol-light subset for the 14B warm start."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/coding_sft_protocol_v3"
OUTPUT = ROOT / "data/coding_sft_protocol_14b_light"

QUOTAS = {
    "problem_submit": 30,
    "first_failure_run": 38,
    "first_failure_direct_repair": 6,
    "post_run_repair": 36,
    "second_failure_run": 16,
    "multiround_final_submit": 18,
}


def rank(sample_id: str) -> str:
    return hashlib.sha256(f"14b-protocol-light-v1:{sample_id}".encode()).hexdigest()


def main() -> int:
    train = json.loads((SOURCE / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((SOURCE / "dev.json").read_text(encoding="utf-8"))
    manifests = [json.loads(line) for line in
                 (SOURCE / "protocol_warmup_manifest.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    by_id = {row["sample_id"]: row for row in train}

    eligible: dict[str, list[dict]] = {state: [] for state in QUOTAS}
    for meta in manifests:
        state = meta["state_type"]
        if state not in eligible:
            continue
        origin = meta.get("candidate_origin")
        source = meta.get("source")
        keep = False
        if state in {"problem_submit", "first_failure_run"}:
            keep = True
        elif state in {"first_failure_direct_repair", "post_run_repair"}:
            keep = origin == "synthetic_single"
        elif state == "second_failure_run":
            keep = source == "mutation_grounded_partial_trajectory"
        elif state == "multiround_final_submit":
            keep = meta.get("original_bug_count") == 2
        if keep:
            eligible[state].append(meta)

    selected_meta = []
    for state, quota in QUOTAS.items():
        candidates = sorted(eligible[state], key=lambda row: rank(row["sample_id"]))
        if len(candidates) < quota:
            raise RuntimeError(f"{state}: need {quota}, found {len(candidates)}")
        selected_meta.extend(candidates[:quota])

    selected_ids = {row["sample_id"] for row in selected_meta}
    selected = [row for row in train if row["sample_id"] in selected_ids]
    if len(selected) != sum(QUOTAS.values()) or len(selected_ids) != len(selected):
        raise RuntimeError("subset size or sample-id uniqueness failure")

    state_counts = Counter(row["state_type"] for row in selected_meta)
    tool_counts = Counter(row["target_tool"] for row in selected_meta)
    for row in selected:
        if not row.get("system") or not row.get("tools"):
            raise RuntimeError(f"{row['sample_id']}: empty system/tools")
        if row["conversations"][-1].get("from") != "function_call":
            raise RuntimeError(f"{row['sample_id']}: target is not a function call")
        action = json.loads(row["conversations"][-1]["value"])
        if action.get("name") not in {"submit", "run_candidate"}:
            raise RuntimeError(f"{row['sample_id']}: invalid target tool")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "train.json").write_text(
        json.dumps(selected, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "dev.json").write_text(
        json.dumps(dev, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "dataset_info.json").write_text(
        (SOURCE / "dataset_info.json").read_text(encoding="utf-8"), encoding="utf-8")
    with (OUTPUT / "protocol_warmup_manifest.jsonl").open("w", encoding="utf-8") as out:
        for row in selected_meta:
            out.write(json.dumps({**row, "light_subset": True}, ensure_ascii=False) + "\n")
    stats = {
        "total_samples": len(selected),
        "dev_samples": len(dev),
        "distinct_train_problems": len({str(row["problem_id"]) for row in selected_meta}),
        "by_state_type": dict(state_counts),
        "by_target_tool": dict(tool_counts),
        "selection_seed": "14b-protocol-light-v1",
        "policy": {
            "repair_states": "single-bug only",
            "second_failure_run": "mutation-grounded only",
            "multiround_final_submit": "original bug count 2 only",
            "multi_bug_direct_repair": "excluded",
        },
    }
    (OUTPUT / "protocol_warmup_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# 14B protocol-light SFT\n\n"
        "Deterministic 144-row subset of protocol v3. It emphasizes pure tool-call "
        "formatting and routing, retains only simple single-bug repairs, and excludes "
        "multi-bug direct repair targets. Dev remains the unchanged 24-row stratified set.\n",
        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
