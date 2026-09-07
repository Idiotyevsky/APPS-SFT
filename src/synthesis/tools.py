from __future__ import annotations

from copy import deepcopy
from typing import Any

from .feedback_projection import project_submit_feedback
from .grader import PrivateGrader
from .schemas import CleanProblem


RUN_CANDIDATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_candidate",
        "description": "Run the current candidate program on one model-provided input and return its observable execution result.",
        "parameters": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"], "additionalProperties": False},
    },
}
SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit a complete solution to the private grader.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False},
    },
}
TOOL_SCHEMAS = [RUN_CANDIDATE_SCHEMA, SUBMIT_SCHEMA]


def validate_arguments(name: str, arguments: dict[str, Any]) -> None:
    keys = {"run_candidate": {"input"}, "submit": {"code"}}
    if name not in keys or not isinstance(arguments, dict) or set(arguments) != keys[name]:
        raise ValueError("invalid tool call")
    key = next(iter(keys[name]))
    if not isinstance(arguments[key], str) or (name == "submit" and not arguments[key].strip()):
        raise ValueError("tool argument must be a string")


class ToolEnvironment:
    def __init__(self, problem: CleanProblem, current_candidate: str, grader: PrivateGrader | None = None):
        self.problem = problem
        self.current_candidate = current_candidate
        self.grader = grader or PrivateGrader()

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        validate_arguments(name, arguments)
        public = self.problem.public_problem
        private = self.problem.private_evaluation
        if name == "run_candidate":
            result, _ = self.grader.run_candidate(self.current_candidate, arguments["input"], public.io_mode, public.fn_name)
            return result.public_dict()
        self.current_candidate = arguments["code"]
        result = self.grader.grade(self.current_candidate, private.inputs, private.outputs, public.io_mode, public.fn_name)
        return project_submit_feedback(result)


def tool_schemas() -> list[dict[str, Any]]:
    return deepcopy(TOOL_SCHEMAS)

