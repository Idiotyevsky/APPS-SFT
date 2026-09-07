import json

import pytest

from synthesis.apps_loader import (
    RejectedProblem, clean_problem, parse_apps_record_strict,
)
from synthesis.grader import PrivateGrader
from synthesis.sandbox import SandboxConfig, SandboxedExecutor


def grader():
    return PrivateGrader(SandboxedExecutor(SandboxConfig(timeout_sec=2)))


def raw_record(fn_name=None):
    io = {"inputs": ["2\n", "3\n"], "outputs": ["4\n", "6\n"]}
    if fn_name:
        io = {
            "inputs": [[2], [3]],
            "outputs": [4, 6],
            "fn_name": fn_name,
        }
    code = (
        "def twice(x):\n    return x * 2\n"
        if fn_name else
        "n = int(input())\nprint(n * 2)\n"
    )
    return {
        "id": 1,
        "question": "Double an integer.",
        "solutions": json.dumps([code]),
        "input_output": json.dumps(io),
        "difficulty": "introductory",
        "starter_code": "",
    }


def test_strict_double_json_parse():
    parsed = parse_apps_record_strict(raw_record())
    assert isinstance(parsed["solutions"], list)
    assert isinstance(parsed["input_output"], dict)
    assert parsed["id"] == "1"
    assert "problem_id" not in parsed


def test_legacy_problem_id_input_is_canonicalized():
    value = raw_record()
    value["problem_id"] = value.pop("id")
    parsed = parse_apps_record_strict(value)
    assert parsed["id"] == "1"
    assert "problem_id" not in parsed


def test_rejects_already_parsed_fields():
    value = raw_record()
    value["solutions"] = ["print(1)"]
    with pytest.raises(ValueError, match="invalid_solutions_json"):
        parse_apps_record_strict(value)


def test_clean_stdin_reference_is_double_verified():
    result = clean_problem(raw_record(), grader())
    assert not isinstance(result, RejectedProblem)
    assert result.public_problem.io_mode == "stdin"
    assert result.private_evaluation.verified_reference_indices == [0]


def test_clean_call_reference():
    result = clean_problem(raw_record("twice"), grader())
    assert not isinstance(result, RejectedProblem)
    assert result.public_problem.io_mode == "call"
    assert result.public_problem.fn_name == "twice"
