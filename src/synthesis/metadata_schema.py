from __future__ import annotations

from typing import Any

from .schemas import Behavior, CandidateOrigin


REQUIRED_TOP_LEVEL = {
    "schema_version", "episode_id", "id", "dataset",
    "dataset_split", "pool", "difficulty", "io_mode",
    "question_hash", "starter_code_hash", "reference", "candidate",
    "mutation", "seed_submit", "counterfactual", "execution_query",
    "behavior_sequence", "decision_steps", "solution_origin",
    "final_code_hash", "final_pass_rate", "final_replay_passed",
    "tool_sequence", "tool_call_count", "model", "runtime",
    "quality", "created_at",
}
OPTIONAL_TOP_LEVEL = {"mutation_set_hash", "label_method"}


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
    )


def validate_metadata(value: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict):
        return ["metadata must be an object"]
    errors: list[str] = []
    keys = set(value)
    missing = REQUIRED_TOP_LEVEL - keys
    unknown = keys - REQUIRED_TOP_LEVEL - OPTIONAL_TOP_LEVEL
    if missing:
        errors.append(f"missing metadata keys: {sorted(missing)}")
    if unknown:
        errors.append(f"unknown metadata keys: {sorted(unknown)}")
    if value.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    identifier = value.get("id")
    if identifier is None or isinstance(identifier, bool) or not str(identifier).strip():
        errors.append("id must be a non-empty scalar")
    if value.get("dataset") != "codeparrot/apps":
        errors.append("dataset must be codeparrot/apps")
    if value.get("dataset_split") != "train":
        errors.append("dataset_split must be train")
    if value.get("pool") != "sft":
        errors.append("pool must be sft")
    if value.get("difficulty") not in {
        "introductory", "interview", "competition",
    }:
        errors.append("invalid difficulty")
    if value.get("io_mode") not in {"stdin", "call"}:
        errors.append("invalid io_mode")
    for key in ("question_hash", "starter_code_hash", "final_code_hash"):
        if not _hash(value.get(key)):
            errors.append(f"{key} is not a sha256 hash")
    reference = value.get("reference")
    if not isinstance(reference, dict) or set(reference) != {
        "solution_index", "code_hash", "verified",
        "verification_pass_rate", "replay_count",
    }:
        errors.append("invalid reference object")
    elif (
        not _hash(reference.get("code_hash"))
        or reference.get("verified") is not True
        or reference.get("verification_pass_rate") != 1.0
        or reference.get("replay_count", 0) < 2
    ):
        errors.append("reference is not twice verified")
    candidate = value.get("candidate")
    origins = {item.value for item in CandidateOrigin}
    if not isinstance(candidate, dict) or candidate.get("origin") not in origins:
        errors.append("invalid candidate origin")
        origin = None
    else:
        origin = candidate["origin"]
        if not _hash(candidate.get("code_hash")):
            errors.append("invalid candidate code hash")
    sequence = value.get("behavior_sequence")
    behaviors = {item.value for item in Behavior}
    if (
        not isinstance(sequence, list) or not sequence
        or any(item not in behaviors for item in sequence)
    ):
        errors.append("invalid behavior sequence")
        primary = None
    else:
        primary = sequence[0]
    if value.get("final_pass_rate") != 1.0:
        errors.append("final_pass_rate must be 1.0")
    if value.get("final_replay_passed") is not True:
        errors.append("final replay is not proven")
    tools = value.get("tool_sequence")
    if (
        not isinstance(tools, list)
        or any(item not in {"run_candidate", "submit"} for item in tools)
        or value.get("tool_call_count") != len(tools)
    ):
        errors.append("invalid tool sequence/count")
    if not tools or tools[-1] != "submit":
        errors.append("tool sequence must end in submit")
    if primary == Behavior.DIRECT_SUBMISSION.value and tools != ["submit"]:
        errors.append("direct submission contains extra tools")
    if primary == Behavior.POST_SUBMIT_DIRECT_REPAIR.value and (
        "run_candidate" in tools or not tools or tools[-1] != "submit"
    ):
        errors.append(
            "direct repair must not run and must end in a submit "
            "(masked context submits are allowed)"
        )
    counterfactual = value.get("counterfactual")
    rule_design = value.get("label_method") == "rule_design"
    if primary == Behavior.DIRECT_SUBMISSION.value:
        if counterfactual is not None:
            errors.append("direct submission must not have counterfactual data")
    elif rule_design and counterfactual is None:
        pass  # rule-designed trajectories carry a design label, not measurements
    elif not isinstance(counterfactual, dict):
        errors.append("repair behavior lacks counterfactual data")
    query = value.get("execution_query")
    if primary == Behavior.POST_SUBMIT_FAILURE_REPLAY.value:
        if (
            not isinstance(query, dict)
            or query.get("kind") != "submit_failing_case"
            or query.get("matches_submit_failing_input") is not True
        ):
            errors.append("failure replay query provenance is invalid")
    if primary == Behavior.PRE_SUBMIT_ACTIVE_VALIDATION.value:
        if rule_design:
            allowed_kinds = {"model_proposed", "rule_constructed"}
            if (
                not isinstance(query, dict)
                or query.get("kind") not in allowed_kinds
            ):
                errors.append("active validation query provenance is invalid")
        elif (
            not isinstance(query, dict)
            or query.get("kind") != "model_proposed"
            or not _hash(query.get("proposal_prompt_hash"))
            or not _hash(query.get("proposal_raw_hash"))
        ):
            errors.append("active validation query provenance is invalid")
    mutation = value.get("mutation")
    if origin in {
        CandidateOrigin.MODEL_NATURAL_FAILURE.value,
        CandidateOrigin.MODEL_NATURAL_CORRECT.value,
        CandidateOrigin.VERIFIED_REFERENCE.value,
    } and mutation is not None:
        errors.append("non-synthetic candidate has mutation metadata")
    if origin == CandidateOrigin.SYNTHETIC_SINGLE.value:
        edits = mutation.get("edits") if isinstance(mutation, dict) else None
        if (
            not isinstance(mutation, dict)
            or mutation.get("bug_count") != 1
            or mutation.get("semantic_edit_count") != 1
            or not isinstance(edits, list)
            or len(edits) != 1
        ):
            errors.append("invalid single-mutation evidence")
    if origin == CandidateOrigin.SYNTHETIC_MULTI.value:
        bug_count = mutation.get("bug_count") if isinstance(mutation, dict) else None
        edits = mutation.get("edits") if isinstance(mutation, dict) else None
        if (
            bug_count not in {2, 3, 4}
            or mutation.get("semantic_edit_count") != bug_count
            or not isinstance(edits, list)
            or len(edits) != bug_count
            or mutation.get("all_individually_harmful") is not True
            or mutation.get("survivor_check_passed") is not True
            or mutation.get("masked_mutation") is not False
        ):
            errors.append("invalid multi-mutation evidence")
    if not isinstance(value.get("decision_steps"), list):
        errors.append("decision_steps must be a list")
    quality = value.get("quality")
    if not isinstance(quality, dict) or set(quality) != {
        "schema_valid", "replay_valid", "leakage_scan_passed",
        "loss_mask_valid", "duplicate_of", "reject_reasons",
    }:
        errors.append("invalid quality object")
    return errors
