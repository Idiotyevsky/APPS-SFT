import json

import pytest

from synthesis.apps_loader import clean_problem
from synthesis.feedback_projection import SUBMIT_PUBLIC_KEYS
from synthesis.grader import PrivateGrader
from synthesis.sandbox import SandboxConfig, SandboxedExecutor
from synthesis.schemas import CleanProblem
from synthesis.tools import TOOL_SCHEMAS, ToolEnvironment


def problem():
    raw = {
        "id": "p",
        "question": "Echo one integer.",
        "solutions": json.dumps(["print(input())\n"]),
        "input_output": json.dumps({
            "inputs": ["7\n"], "outputs": ["7\n"],
        }),
        "difficulty": "introductory",
        "starter_code": "",
    }
    grader = PrivateGrader(
        SandboxedExecutor(SandboxConfig(timeout_sec=2))
    )
    return clean_problem(raw, grader), grader


def test_only_two_exact_tool_schemas():
    assert [
        item["function"]["name"] for item in TOOL_SCHEMAS
    ] == ["run_candidate", "submit"]
    for item in TOOL_SCHEMAS:
        assert item["function"]["parameters"]["additionalProperties"] is False


def test_submit_projection_has_no_oracle_fields():
    value, grader = problem()
    assert isinstance(value, CleanProblem)
    env = ToolEnvironment(value, "print('bad')\n", grader)
    result = env.call("submit", {"code": "print(input())\n"})
    assert set(result) == SUBMIT_PUBLIC_KEYS
    assert "expected" not in result
    assert "actual" not in result


def test_tool_arguments_are_strict():
    value, grader = problem()
    assert isinstance(value, CleanProblem)
    env = ToolEnvironment(value, "print(input())\n", grader)
    with pytest.raises(ValueError):
        env.call("run_candidate", {"input": "1\n", "code": "x"})
