from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .config import SynthesisConfig
from .feedback_projection import project_submit_feedback
from .grader import PrivateGrader, failure_signature, same_grade
from .io_utils import read_jsonl
from .normalize import content_hash, normalized_input_equal
from .sandbox import SandboxConfig, SandboxedExecutor
from .schemas import CleanProblem


_CANDIDATE_PATTERN = re.compile(
    r"Current candidate:\n```python\n(.*?)\n```",
    flags=re.DOTALL,
)
_TEMP_PATTERN = re.compile(r"/[^\s:]*/synthesis-episode-[^/\s:]+")


def _executor(config: SynthesisConfig) -> PrivateGrader:
    return PrivateGrader(SandboxedExecutor(SandboxConfig(
        timeout_sec=config.candidate_timeout_sec,
        memory_mb=config.candidate_memory_mb,
        max_output_bytes=config.max_output_bytes,
        max_input_bytes=config.max_input_bytes,
        backend=config.sandbox_backend,
    )))


def _initial_candidate(messages: list[dict[str, Any]]) -> str:
    """Return the starting program under replay.

    Native V4 format: the first assistant action is a masked submit(code);
    legacy format embeds the candidate in the user state text.
    """
    for message in messages:
        if message.get("role") != "user":
            continue
        match = _CANDIDATE_PATTERN.search(message.get("content") or "")
        if match:
            return match.group(1) + "\n"
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        call = calls[0]
        if call.get("name") == "submit" and isinstance(
            call.get("arguments"), dict
        ) and isinstance(call["arguments"].get("code"), str):
            return call["arguments"]["code"]
    return ""


def _is_native_history(messages: list[dict[str, Any]]) -> bool:
    """True when the failure state is a real assistant submit->tool history.

    Native V4 shape begins with a masked ``assistant submit(code)`` whose very
    next message is the matching ``tool`` failure response. A lone final
    trainable submit (legacy V2) must NOT qualify.
    """
    for index, message in enumerate(messages[:-1]):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not calls or calls[0].get("name") != "submit":
            continue
        if not message.get("trainable"):
            following = messages[index + 1]
            if (following.get("role") == "tool"
                    and following.get("name") == "submit"):
                return True
            return False
        # a masked submit must precede any trainable submit in native history
        return False
    return False


def _stable_run(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    normalized["stderr"] = _TEMP_PATTERN.sub("/work", value.get("stderr", ""))
    error = value.get("error")
    if isinstance(error, dict):
        normalized["error"] = {
            "type": error.get("type"),
            "message": _TEMP_PATTERN.sub(
                "/work", str(error.get("message", "")),
            ),
        }
    return normalized


def replay_artifacts(
    output_dir: str | Path, config: SynthesisConfig,
) -> list[dict[str, str]]:
    root = Path(output_dir)
    problems = {
        str(row["id"]): CleanProblem.from_dict(row)
        for row in read_jsonl(root / "cleaned" / "problems.jsonl")
    }
    grader = _executor(config)
    issues: list[dict[str, str]] = []
    for episode in read_jsonl(root / "episodes.jsonl"):
        episode_id = episode["id"]
        metadata = episode["metadata"]
        problem = problems.get(str(metadata["id"]))
        if problem is None:
            issues.append({
                "episode_id": episode_id,
                "code": "replay_missing_problem",
                "message": "cleaned problem is unavailable",
            })
            continue
        public = problem.public_problem
        private = problem.private_evaluation
        current = _initial_candidate(episode["messages"])
        latest_feedback: dict[str, Any] | None = None
        primary = metadata["behavior_sequence"][0]
        messages = episode["messages"]
        if primary.startswith("post_submit_") and not _is_native_history(messages):
            first = grader.grade(
                current, private.inputs, private.outputs,
                public.io_mode, public.fn_name,
            )
            second = grader.grade(
                current, private.inputs, private.outputs,
                public.io_mode, public.fn_name,
            )
            if first.accepted or not same_grade(first, second):
                issues.append({
                    "episode_id": episode_id,
                    "code": "seed_replay",
                    "message": "seed candidate is not a stable failure",
                })
                continue
            latest_feedback = project_submit_feedback(first)
            seed = metadata.get("seed_submit") or {}
            if (
                seed.get("status") != first.status
                or seed.get("passed") != first.passed
                or seed.get("total") != first.total
                or seed.get("pass_rate") != first.pass_rate
                or seed.get("failure_signature")
                != failure_signature(first)
                or seed.get("failing_input_hash")
                != content_hash(first.failing_input or "")
            ):
                issues.append({
                    "episode_id": episode_id,
                    "code": "seed_replay",
                    "message": "seed failure evidence does not replay",
                })
        else:
            # Native V4 history: the leading masked submit and each later turn
            # are validated message-by-message in the loop below.
            pass
        final_code: str | None = None
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            if index + 1 >= len(messages):
                issues.append({
                    "episode_id": episode_id,
                    "code": "replay_protocol",
                    "message": "assistant action has no tool response",
                })
                break
            response_message = messages[index + 1]
            call = message["tool_calls"][0]
            recorded = json.loads(response_message["content"])
            if call["name"] == "run_candidate":
                if latest_feedback is not None:
                    failing = latest_feedback.get("failing_input")
                    if (
                        not isinstance(failing, str)
                        or not normalized_input_equal(
                            call["arguments"]["input"], failing,
                            public.io_mode,
                        )
                    ):
                        issues.append({
                            "episode_id": episode_id,
                            "code": "replay_input_mismatch",
                            "message": "run input differs from latest failure",
                        })
                result, _ = grader.run_candidate(
                    current, call["arguments"]["input"],
                    public.io_mode, public.fn_name,
                )
                if _stable_run(result.public_dict()) != _stable_run(recorded):
                    issues.append({
                        "episode_id": episode_id,
                        "code": "observation_replay",
                        "message": "run_candidate observation changed",
                    })
            else:
                current = call["arguments"]["code"]
                final_code = current
                grade = grader.grade(
                    current, private.inputs, private.outputs,
                    public.io_mode, public.fn_name,
                )
                projected = project_submit_feedback(grade)
                if projected != recorded:
                    issues.append({
                        "episode_id": episode_id,
                        "code": "submit_replay",
                        "message": "submit projection changed",
                    })
                latest_feedback = projected if not grade.accepted else None
        if final_code is None:
            issues.append({
                "episode_id": episode_id,
                "code": "final_replay",
                "message": "episode has no final code",
            })
            continue
        checks = [
            grader.grade(
                final_code, private.inputs, private.outputs,
                public.io_mode, public.fn_name,
            )
            for _ in range(2)
        ]
        if not all(item.accepted for item in checks):
            issues.append({
                "episode_id": episode_id,
                "code": "final_replay",
                "message": "final code did not pass two fresh replays",
            })
    return issues
