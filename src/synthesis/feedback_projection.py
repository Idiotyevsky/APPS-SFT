from __future__ import annotations

from math import isfinite
from typing import Any

from .schemas import GradeResult


SUBMIT_PUBLIC_KEYS = {"status", "passed", "total", "pass_rate", "failing_input"}


def project_submit_feedback(result: GradeResult) -> dict[str, Any]:
    public = {
        "status": result.status,
        "passed": result.passed,
        "total": result.total,
        "pass_rate": result.pass_rate,
        "failing_input": result.failing_input,
    }
    assert set(public) == SUBMIT_PUBLIC_KEYS
    return public


def validate_public_feedback(value: dict[str, Any]) -> None:
    if set(value) != SUBMIT_PUBLIC_KEYS:
        raise ValueError("submit feedback contains missing or secret fields")
    total, passed = value["total"], value["passed"]
    status = value["status"]
    if status not in {
        "accepted", "wrong_answer", "runtime_error",
        "timeout", "compile_error",
    }:
        raise ValueError("invalid submit status")
    if (
        not isinstance(total, int) or isinstance(total, bool) or total <= 0
        or not isinstance(passed, int) or isinstance(passed, bool)
        or not 0 <= passed <= total
    ):
        raise ValueError("invalid passed/total")
    pass_rate = value["pass_rate"]
    if (
        not isinstance(pass_rate, (int, float))
        or isinstance(pass_rate, bool)
        or not isfinite(float(pass_rate))
        or not 0 <= float(pass_rate) <= 1
        or abs(float(pass_rate) - passed / total) > 1e-12
    ):
        raise ValueError("inconsistent pass_rate")
    failing_input = value["failing_input"]
    if failing_input is not None and not isinstance(failing_input, str):
        raise ValueError("failing_input must be a string or null")
    if passed == total:
        if status != "accepted":
            raise ValueError("fully passing result must be accepted")
        if failing_input is not None:
            raise ValueError("accepted result must not expose a failing input")
    elif status == "accepted":
        raise ValueError("accepted result must pass every test")

