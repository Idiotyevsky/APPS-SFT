#!/usr/bin/env python3
"""Build the final transition-clean Minimal Protocol SFT v3 dataset.

The v3 train set keeps simple repair transitions from protocol v2 and replaces
selected multi-bug one-shot repairs with mutation-grounded partial trajectories
whose intermediate code measurably improves the currently visible input.
Every new observation is produced by the existing local grader/runtime.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "sft/scripts"))

from audit_mutation_causal_support import (  # noqa: E402
    read_inputs, recovery_candidates,
)
from synthesis.apps_loader import parse_apps_record_strict  # noqa: E402
from synthesis.feedback_projection import project_submit_feedback  # noqa: E402
from synthesis.grader import PrivateGrader  # noqa: E402
from synthesis.mutation import apply_edits  # noqa: E402
from synthesis.sandbox import SandboxConfig, SandboxedExecutor  # noqa: E402

V2 = ROOT / "data/coding_sft_protocol_v2"
OUT = ROOT / "data/coding_sft_protocol_v3"
EPISODES = ROOT / "data/sft_v5_final/episodes.jsonl"
APPS = ROOT / "data/raw/apps/train.jsonl"
CAUSAL_DETAILS = ROOT / "sft/outputs/mutation_causal_support_audit_v2/details.jsonl"


def grader() -> PrivateGrader:
    return PrivateGrader(SandboxedExecutor(
        SandboxConfig(timeout_sec=3, memory_mb=512, backend="local")))


def grade(gr: PrivateGrader, problem: dict, code: str):
    spec = problem["input_output"]
    mode = "call" if spec.get("fn_name") else "stdin"
    return gr.grade(code, spec["inputs"], spec["outputs"], mode,
                    spec.get("fn_name"))


def run(gr: PrivateGrader, problem: dict, code: str, input_text: str) -> dict:
    spec = problem["input_output"]
    mode = "call" if spec.get("fn_name") else "stdin"
    result, _ = gr.run_candidate(code, input_text, mode, spec.get("fn_name"))
    return result.public_dict()


def call(name: str, arguments: dict) -> dict:
    return {
        "from": "function_call",
        "value": json.dumps({"name": name, "arguments": arguments},
                            ensure_ascii=False, separators=(",", ":")),
    }


def observation(value: dict) -> dict:
    return {
        "from": "observation",
        "value": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
    }


def main() -> int:
    train = json.loads((V2 / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((V2 / "dev.json").read_text(encoding="utf-8"))
    manifest = {
        row["sample_id"]: row for row in map(
            json.loads,
            (V2 / "protocol_warmup_manifest.jsonl").read_text(
                encoding="utf-8").splitlines())
    }
    ids = {str(row["problem_id"]) for row in manifest.values()}
    episodes = {}
    with EPISODES.open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            if str(episode["id"]) in ids:
                episodes[str(episode["id"])] = episode

    selected = []
    selected_manifest = []
    for row in train:
        meta = manifest[row["sample_id"]]
        episode_meta = episodes[str(meta["problem_id"])]["metadata"]
        origin = episode_meta["candidate"]["origin"]
        bug_count = (episode_meta.get("mutation") or {}).get("bug_count")
        state = meta["state_type"]
        keep = False
        reason = None
        if state == "problem_submit":
            keep, reason = True, "retain verified-reference protocol submit"
        elif state == "first_failure_run" and origin == "synthetic_single":
            keep, reason = True, "retain single-bug first-failure router"
        elif state in {"first_failure_direct_repair", "post_run_repair"} \
                and origin == "synthetic_single":
            keep, reason = True, "retain single-bug repair"
        elif state == "second_failure_run":
            keep, reason = True, "retain real multiround router state"
        elif state == "multiround_final_submit" and bug_count == 2:
            keep, reason = True, "retain final repair with one current bug"
        if keep:
            selected.append(row)
            selected_manifest.append({
                **meta,
                "source": "protocol_v2_retained",
                "candidate_origin": origin,
                "original_bug_count": bug_count,
                "selection_reason": reason,
            })

    details = {
        row["sample_id"]: row for row in map(
            json.loads, CAUSAL_DETAILS.read_text(encoding="utf-8").splitlines())
        if row.get("current_bug_count", 0) > 1
        and any(s["is_proper_subset"] and s["delta_partial_score"] > 1e-12
                for s in row.get("subset_results", []))
    }
    if len(details) != 23:
        raise RuntimeError(f"expected 23 improving multi repairs, got {len(details)}")

    jobs = {job["sample_id"]: job for job in read_inputs(V2, EPISODES, APPS)
            if job["sample_id"] in details}
    if jobs.keys() != details.keys():
        raise RuntimeError("causal details/jobs sample-id mismatch")
    gr = grader()
    grounded_stats = Counter()
    for sample_id in sorted(jobs):
        job = jobs[sample_id]
        detail = details[sample_id]
        matches, recovery = recovery_candidates(job)
        if len(matches) != 1:
            raise RuntimeError(f"{sample_id}: reconstruction count {len(matches)}")
        edits = matches[0]
        candidates = [
            item for item in detail["subset_results"]
            if item["is_proper_subset"] and item["delta_partial_score"] > 1e-12
        ]
        best = sorted(candidates, key=lambda item: (
            -item["delta_partial_score"], item["undo_size"], item["undo_indices"]))[0]
        undone = set(best["undo_indices"])
        remaining = [edit for index, edit in enumerate(edits) if index not in undone]
        partial_code, _ = apply_edits(job["reference"], remaining)
        partial_result = grade(gr, job["problem"], partial_code)
        if partial_result.accepted or not isinstance(partial_result.failing_input, str):
            raise RuntimeError(f"{sample_id}: partial must remain a gradeable failure")
        partial_feedback = project_submit_feedback(partial_result)
        next_input = partial_result.failing_input
        run_feedback = run(gr, job["problem"], partial_code, next_input)
        final_result = grade(gr, job["problem"], job["reference"])
        if not final_result.accepted:
            raise RuntimeError(f"{sample_id}: reference stopped passing")

        source_row = next(row for row in train if row["sample_id"] == sample_id)
        history = list(source_row["conversations"][:-1])
        stem = f"{sample_id}:grounded"
        rows_and_states = [
            (
                history + [call("submit", {"code": partial_code})],
                ("first_failure_direct_repair"
                 if job["state_type"] == "first_failure_direct_repair"
                 else "post_run_repair"),
                "partial submit measurably improves visible input",
            ),
            (
                history + [call("submit", {"code": partial_code}),
                           observation(partial_feedback),
                           call("run_candidate", {"input": next_input})],
                "second_failure_run",
                "run new real failing input after partial submit",
            ),
            (
                history + [call("submit", {"code": partial_code}),
                           observation(partial_feedback),
                           call("run_candidate", {"input": next_input}),
                           observation(run_feedback),
                           call("submit", {"code": job["reference"]})],
                "multiround_final_submit",
                "final single-step repair after real partial failure/run",
            ),
        ]
        for suffix, (conversations, state, reason) in enumerate(rows_and_states, 1):
            new_id = f"{stem}:{suffix}"
            new_row = {
                "sample_id": new_id,
                "system": source_row["system"],
                "tools": source_row["tools"],
                "conversations": conversations,
            }
            selected.append(new_row)
            selected_manifest.append({
                "sample_id": new_id,
                "problem_id": job["problem_id"],
                "state_type": state,
                "target_tool": json.loads(conversations[-1]["value"])["name"],
                "difficulty": job["difficulty"],
                "source": "mutation_grounded_partial_trajectory",
                "source_sample_id": sample_id,
                "candidate_origin": "synthetic_multi",
                "original_bug_count": job["original_bug_count"],
                "current_bug_count_before_partial": len(edits),
                "remaining_bug_count_after_partial": len(remaining),
                "visible_partial_score_gain": best["delta_partial_score"],
                "undo_indices": best["undo_indices"],
                "selection_reason": reason,
                "real_partial_submit_status": partial_result.status,
                "real_partial_submit_pass_rate": partial_result.pass_rate,
                "real_followup_run_status": run_feedback["status"],
            })
            grounded_stats[state] += 1

    if len(selected) != 238:
        raise RuntimeError(f"expected 238 v3 train rows, got {len(selected)}")
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise RuntimeError("duplicate sample ids in v3")
    manifest_by_id = {row["sample_id"]: row for row in selected_manifest}
    if manifest_by_id.keys() != {row["sample_id"] for row in selected}:
        raise RuntimeError("v3 manifest/train mismatch")

    # Structural hard gates.
    state_counts = Counter()
    tool_counts = Counter()
    for row in selected:
        if not row.get("system") or not row.get("tools"):
            raise RuntimeError(f"{row['sample_id']}: missing system/tools")
        if row["conversations"][-1]["from"] != "function_call":
            raise RuntimeError(f"{row['sample_id']}: target is not function_call")
        meta = manifest_by_id[row["sample_id"]]
        target = json.loads(row["conversations"][-1]["value"])
        if target["name"] != meta["target_tool"]:
            raise RuntimeError(f"{row['sample_id']}: manifest tool mismatch")
        state_counts[meta["state_type"]] += 1
        tool_counts[target["name"]] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.json").write_text(
        json.dumps(selected, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "dev.json").write_text(
        json.dumps(dev, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset_info = {
        f"coding_agent_{split}": {
            "file_name": f"{split}.json", "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system",
                        "tools": "tools"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt",
                     "observation_tag": "observation",
                     "function_tag": "function_call"},
        } for split in ("train", "dev")
    }
    (OUT / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with (OUT / "protocol_warmup_manifest.jsonl").open("w", encoding="utf-8") as out:
        for row in selected_manifest:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "total_samples": len(selected),
        "dev_samples": len(dev),
        "distinct_train_problems": len({str(m["problem_id"]) for m in selected_manifest}),
        "by_state_type": dict(state_counts),
        "by_target_tool": dict(tool_counts),
        "retained_v2_samples": len(selected) - 3 * len(details),
        "grounded_source_trajectories": len(details),
        "grounded_prefix_samples": 3 * len(details),
        "grounded_by_state": dict(grounded_stats),
        "removed_policy": {
            "multi_bug_direct_reference_target": "removed",
            "multi_bug_post_run_without_partial_improvement": "removed",
            "multi_bug_post_run_with_partial_improvement": "replaced by 3-step trajectory",
        },
    }
    (OUT / "protocol_warmup_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Minimal Protocol SFT v3 — transition-clean final 7B experiment\n\n"
        "- 238 train prefix samples; no quota padding.\n"
        "- Retains protocol and single-bug repairs from v2.\n"
        "- Removes one-shot multi-bug direct/reference transitions.\n"
        "- Adds 23 mutation-grounded real partial trajectories (3 prefixes each).\n"
        "- Weighted-repair loss parameters remain unchanged from run10b.\n",
        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
