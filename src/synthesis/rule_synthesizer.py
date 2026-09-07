from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .feedback_projection import project_submit_feedback
from .grader import PrivateGrader, failure_signature
from .normalize import content_hash, normalized_code_hash
from .prompting.state_builder import (
    append_tool_observation,
    build_post_submit_state,
    build_pre_submit_state,
    build_problem_only_state,
)
from .schemas import CleanProblem
from .tools import tool_schemas

DIRECT_REPAIR = "post_submit_direct_repair"
FAILURE_REPLAY = "post_submit_failure_replay"
ACTIVE_VALIDATION = "pre_submit_active_validation"
DIRECT_SUBMISSION = "direct_submission"


def _grade_feedback(grader: PrivateGrader, problem: CleanProblem, code: str) -> dict[str, Any]:
    public = problem.public_problem
    private = problem.private_evaluation
    result = grader.grade(
        code, private.inputs, private.outputs,
        public.io_mode, public.fn_name,
    )
    return project_submit_feedback(result)


def _run_observation(
    grader: PrivateGrader, problem: CleanProblem, code: str, input_text: str,
) -> dict[str, Any]:
    public = problem.public_problem
    result, _ = grader.run_candidate(
        code, input_text, public.io_mode, public.fn_name,
    )
    return result.public_dict()


def _base_metadata(
    problem: CleanProblem, candidate: dict[str, Any],
    behavior: str, code_hash: str, config: Any,
) -> dict[str, Any]:
    public = problem.public_problem
    reference = problem.private_evaluation.selected_reference_code or ""
    return {
        "schema_version": "1.0",
        "dataset": "codeparrot/apps",
        "dataset_split": "train",
        "pool": "sft",
        "difficulty": public.difficulty,
        "io_mode": public.io_mode,
        "question_hash": content_hash(public.question),
        "starter_code_hash": content_hash(public.starter_code),
        "reference": {
            "solution_index": problem.private_evaluation.selected_reference_index,
            "code_hash": normalized_code_hash(reference),
            "verified": True,
            "verification_pass_rate": 1.0,
            "replay_count": 2,
        },
        "candidate": {
            "origin": candidate["origin"],
            "code_hash": code_hash,
            "is_buggy_context": candidate["origin"] not in {
                "verified_reference", "model_natural_correct",
            },
            "generator_model": None,
            "generator_revision": None,
            "generation_seed": None,
            "generation_prompt_hash": None,
        },
        "mutation": candidate.get("mutation"),
        "behavior_sequence": [behavior],
        "solution_origin": "verified_reference",
        "final_pass_rate": 1.0,
        "final_replay_passed": True,
        "model": {
            "name": "rule-based-fixture",
            "revision": None,
            "template_hash": None,
            "generation_config_hash": None,
        },
        "runtime": {
            "python_version": "3.x",
            "grader_version": "coding-agent-synthesis:0.1.0",
            "sandbox_config_hash": None,
        },
        "quality": {
            "schema_valid": True,
            "replay_valid": True,
            "leakage_scan_passed": True,
            "loss_mask_valid": True,
            "duplicate_of": None,
            "reject_reasons": [],
        },
        "label_method": "rule_design",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _seed_submit_block(candidate: dict[str, Any]) -> dict[str, Any]:
    seed = dict(candidate["seed_submit"])
    seed["failure_signature"] = candidate.get("failure_signature")
    if seed.get("failing_input") is not None:
        seed["failing_input_hash"] = content_hash(seed["failing_input"])
        seed.pop("failing_input", None)
    return seed


def _assemble_episode(
    problem: CleanProblem, behavior: str, messages: list[dict[str, Any]],
    metadata: dict[str, Any], candidate_hash: str,
) -> dict[str, Any]:
    episode_id = problem.problem_id
    metadata["episode_id"] = episode_id
    metadata["id"] = problem.problem_id
    calls = [
        call["name"] for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]
    metadata["tool_sequence"] = calls
    metadata["tool_call_count"] = len(calls)
    metadata["decision_steps"] = [
        {
            "step": index, "behavior": behavior,
            "assistant_trainable": True, "downstream_success": True,
        }
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and message.get("trainable")
    ]
    return {
        "id": episode_id,
        "tools": tool_schemas(),
        "messages": messages,
        "metadata": metadata,
    }


def build_direct_submission(
    problem: CleanProblem, candidate: dict[str, Any],
    grader: PrivateGrader, config: Any,
) -> dict[str, Any]:
    public = problem.public_problem
    private = problem.private_evaluation
    code = candidate["code"]
    first = grader.grade(
        code, private.inputs, private.outputs, public.io_mode, public.fn_name,
    )
    second = grader.grade(
        code, private.inputs, private.outputs, public.io_mode, public.fn_name,
    )
    if not (first.accepted and second.accepted):
        raise ValueError("direct submission code is not twice verified")
    state = build_problem_only_state(public)
    feedback = project_submit_feedback(second)
    messages = append_tool_observation(
        state, "submit", {"code": code}, feedback, trainable=True,
    )
    code_hash = normalized_code_hash(code)
    metadata = _base_metadata(
        problem, candidate, DIRECT_SUBMISSION, code_hash, config,
    )
    metadata["final_code_hash"] = code_hash
    metadata.update({
        "seed_submit": None,
        "counterfactual": None,
        "execution_query": None,
    })
    return _assemble_episode(
        problem, DIRECT_SUBMISSION, messages, metadata, code_hash,
    )


def build_post_submit_repair(
    problem: CleanProblem, candidate: dict[str, Any],
    reference: str, behavior: str, grader: PrivateGrader, config: Any,
    use_run: bool,
) -> dict[str, Any]:
    """Direct repair or failure replay over a real failing candidate."""
    public = problem.public_problem
    code = candidate["code"]
    seed_feedback = dict(candidate["seed_submit"])
    failing = seed_feedback.get("failing_input")
    if not isinstance(failing, str):
        raise ValueError("post-submit repair requires a failing input")
    state = build_post_submit_state(public, code, seed_feedback)
    if use_run:
        observation = _run_observation(grader, problem, code, failing)
        messages = append_tool_observation(
            state, "run_candidate", {"input": failing},
            observation, trainable=True,
        )
    else:
        messages = state
    feedback = _grade_feedback(grader, problem, reference)
    if feedback["status"] != "accepted":
        raise ValueError("reference repair is not accepted")
    messages = append_tool_observation(
        messages, "submit", {"code": reference}, feedback, trainable=True,
    )
    candidate_hash = normalized_code_hash(code)
    metadata = _base_metadata(
        problem, candidate, behavior, candidate_hash, config,
    )
    metadata["seed_submit"] = _seed_submit_block(candidate)
    metadata["counterfactual"] = None
    metadata["execution_query"] = {
        "kind": "submit_failing_case",
        "input_hash": content_hash(failing),
        "proposal_model": None,
        "proposal_prompt_hash": None,
        "proposal_raw_hash": None,
        "observable_fields": [
            "question", "starter_code", "candidate",
            "submit_feedback",
        ],
        "matches_submit_failing_input": True,
        "rule_note": (
            "behavior assigned by deterministic rule; no empirical "
            "counterfactual measurement was performed"
        ),
    }
    metadata["final_code_hash"] = normalized_code_hash(reference)
    return _assemble_episode(
        problem, behavior, messages, metadata, candidate_hash,
    )


def build_active_validation(
    problem: CleanProblem, candidate: dict[str, Any],
    reference: str, probe_input: str, grader: PrivateGrader, config: Any,
) -> dict[str, Any]:
    """Pre-submit active validation on a rule-constructed diagnostic input."""
    public = problem.public_problem
    code = candidate["code"]
    state = build_pre_submit_state(public, code)
    observation = _run_observation(grader, problem, code, probe_input)
    if observation["status"] in {"invalid_input", "timeout", "output_limit"}:
        raise ValueError("probe input is not executable")
    messages = append_tool_observation(
        state, "run_candidate", {"input": probe_input},
        observation, trainable=True,
    )
    feedback = _grade_feedback(grader, problem, reference)
    if feedback["status"] != "accepted":
        raise ValueError("reference repair is not accepted")
    messages = append_tool_observation(
        messages, "submit", {"code": reference}, feedback, trainable=True,
    )
    candidate_hash = normalized_code_hash(code)
    metadata = _base_metadata(
        problem, candidate, ACTIVE_VALIDATION, candidate_hash, config,
    )
    metadata["seed_submit"] = _seed_submit_block(candidate)
    metadata["counterfactual"] = None
    metadata["execution_query"] = {
        "kind": "rule_constructed",
        "input_hash": content_hash(probe_input),
        "proposal_model": None,
        "proposal_prompt_hash": None,
        "proposal_raw_hash": None,
        "observable_fields": [
            "question", "starter_code", "io_mode",
            "input_format_note", "fn_name", "candidate",
        ],
        "matches_submit_failing_input": False,
        "rule_note": (
            "diagnostic input constructed by a deterministic rule probe, "
            "not proposed by a model and never sourced from hidden tests"
        ),
    }
    metadata["final_code_hash"] = normalized_code_hash(reference)
    return _assemble_episode(
        problem, ACTIVE_VALIDATION, messages, metadata, candidate_hash,
    )


def build_multiround(
    problem: CleanProblem, candidate: dict[str, Any],
    reference: str, partial: str, grader: PrivateGrader,
    config: Any,
) -> dict[str, Any]:
    """Multi-round: run, fail a partial fix, run the new failure, submit."""
    public = problem.public_problem
    code = candidate["code"]
    seed_feedback = dict(candidate["seed_submit"])
    failing_one = seed_feedback.get("failing_input")
    mutation = candidate.get("mutation") or {}
    if not isinstance(failing_one, str) or mutation.get("bug_count", 0) < 2:
        raise ValueError("multiround requires a failing input and >=2 edits")
    state = build_post_submit_state(public, code, seed_feedback)

    partial_feedback = _grade_feedback(grader, problem, partial)
    if partial_feedback["status"] == "accepted":
        raise ValueError("partial repair unexpectedly passes")
    failing_two = partial_feedback.get("failing_input")
    if not isinstance(failing_two, str):
        raise ValueError("partial repair exposes no failing input")

    observation_one = _run_observation(grader, problem, code, failing_one)
    messages = append_tool_observation(
        state, "run_candidate", {"input": failing_one},
        observation_one, trainable=True,
    )
    messages = append_tool_observation(
        messages, "submit", {"code": partial},
        partial_feedback, trainable=False,
    )
    observation_two = _run_observation(
        grader, problem, partial, failing_two,
    )
    messages = append_tool_observation(
        messages, "run_candidate", {"input": failing_two},
        observation_two, trainable=True,
    )
    final_feedback = _grade_feedback(grader, problem, reference)
    if final_feedback["status"] != "accepted":
        raise ValueError("reference repair is not accepted")
    messages = append_tool_observation(
        messages, "submit", {"code": reference},
        final_feedback, trainable=True,
    )
    candidate_hash = normalized_code_hash(code)
    metadata = _base_metadata(
        problem, candidate, FAILURE_REPLAY, candidate_hash, config,
    )
    metadata["seed_submit"] = _seed_submit_block(candidate)
    metadata["counterfactual"] = None
    metadata["execution_query"] = {
        "kind": "submit_failing_case",
        "input_hash": content_hash(failing_one),
        "proposal_model": None,
        "proposal_prompt_hash": None,
        "proposal_raw_hash": None,
        "observable_fields": [
            "question", "starter_code", "candidate",
            "submit_feedback",
        ],
        "matches_submit_failing_input": True,
        "rule_note": (
            "scripted multi-round fixture: the failed partial submit is "
            "context-only (trainable=false); behavior assigned by rule"
        ),
    }
    metadata["final_code_hash"] = normalized_code_hash(reference)
    return _assemble_episode(
        problem, FAILURE_REPLAY, messages, metadata, candidate_hash,
    )
