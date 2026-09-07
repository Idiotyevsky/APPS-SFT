from __future__ import annotations

import json
from typing import Any

from ..schemas import PublicProblem


SYSTEM_PROMPT = """You are solving a programming problem. You may only call tools currently provided to you. Submit a complete Python 3 solution with submit(code). After a candidate exists, run_candidate(input) may be provided to test it on one input. Never invent tool results. The grader executes Python 3; do not submit C++, Java, or a patch."""


def _problem_text(problem: PublicProblem) -> str:
    starter = problem.starter_code or "(none)"
    fn = f"\nFunction name: {problem.fn_name}" if problem.fn_name else ""
    return (
        f"Problem:\n{problem.question}\n\nStarter code:\n{starter}\n\n"
        f"I/O mode: {problem.io_mode}{fn}\nInput convention: {problem.input_format_note}"
    )


def build_problem_only_state(problem: PublicProblem) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT, "trainable": False},
        {"role": "user", "content": _problem_text(problem), "trainable": False},
    ]


def build_post_submit_state(problem: PublicProblem, candidate: str, feedback: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = {"status", "passed", "total", "pass_rate", "failing_input"}
    if set(feedback) not in (allowed, allowed | {"error"}):
        raise ValueError("post-submit state received non-public grader fields")
    if "error" in feedback and not isinstance(feedback["error"], dict):
        raise ValueError("post-submit state received invalid error feedback")
    content = (
        _problem_text(problem)
        + f"\n\nCurrent candidate:\n```python\n{candidate}\n```"
        + "\n\nPrevious submit result:\n"
        + json.dumps(feedback, ensure_ascii=False, sort_keys=True)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT, "trainable": False},
        {"role": "user", "content": content, "trainable": False},
    ]


def build_pre_submit_state(problem: PublicProblem, candidate: str) -> list[dict[str, Any]]:
    content = (
        _problem_text(problem)
        + f"\n\nCurrent candidate:\n```python\n{candidate}\n```"
        + "\n\nNo grader feedback is available."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT, "trainable": False},
        {"role": "user", "content": content, "trainable": False},
    ]


def append_tool_observation(messages: list[dict[str, Any]], name: str, arguments: dict[str, str], observation: dict[str, Any], trainable: bool = True) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    result.append({"role": "assistant", "tool_calls": [{"name": name, "arguments": arguments}], "trainable": trainable})
    result.append({"role": "tool", "name": name, "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")), "trainable": False})
    return result

