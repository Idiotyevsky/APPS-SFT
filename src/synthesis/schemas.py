from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class IOMode(str, Enum):
    STDIN = "stdin"
    CALL = "call"


class CandidateOrigin(str, Enum):
    SYNTHETIC_SINGLE = "synthetic_single"
    SYNTHETIC_MULTI = "synthetic_multi"
    MODEL_NATURAL_FAILURE = "model_natural_failure"
    VERIFIED_REFERENCE = "verified_reference"
    MODEL_NATURAL_CORRECT = "model_natural_correct"


class Behavior(str, Enum):
    POST_SUBMIT_DIRECT_REPAIR = "post_submit_direct_repair"
    POST_SUBMIT_FAILURE_REPLAY = "post_submit_failure_replay"
    PRE_SUBMIT_ACTIVE_VALIDATION = "pre_submit_active_validation"
    DIRECT_SUBMISSION = "direct_submission"


@dataclass(slots=True)
class PublicProblem:
    question: str
    starter_code: str
    difficulty: str
    io_mode: str
    input_format_note: str
    fn_name: str | None = None


@dataclass(slots=True)
class PrivateEvaluation:
    inputs: list[Any]
    outputs: list[Any]
    verified_reference_indices: list[int] = field(default_factory=list)
    selected_reference_index: int | None = None
    selected_reference_code: str | None = None


@dataclass(slots=True)
class CleanProblem:
    problem_id: str
    public_problem: PublicProblem
    private_evaluation: PrivateEvaluation
    source: dict[str, Any]
    solutions: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Canonical public APPS identifier.

        ``problem_id`` remains an internal attribute for backwards
        compatibility with existing Python callers; serialized records use
        ``id`` exclusively.
        """
        return self.problem_id

    def to_dict(self, include_reference_code: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = payload.pop("problem_id")
        if not include_reference_code:
            payload["private_evaluation"].pop("selected_reference_code", None)
            payload.pop("solutions", None)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CleanProblem":
        identifier = value.get("id", value.get("problem_id"))
        if identifier is None or isinstance(identifier, bool):
            raise ValueError("cleaned problem is missing id")
        return cls(
            problem_id=str(identifier),
            public_problem=PublicProblem(**value["public_problem"]),
            private_evaluation=PrivateEvaluation(**value["private_evaluation"]),
            source=dict(value.get("source", {})),
            solutions=list(value.get("solutions", [])),
        )


@dataclass(slots=True)
class ExecutionError:
    type: str
    message: str


@dataclass(slots=True)
class RunResult:
    status: str
    stdout: str = ""
    stderr: str = ""
    error: dict[str, str] | None = None
    exit_code: int | None = 0
    truncated: bool = False

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaseResult:
    input: str
    expected: Any
    actual: Any
    passed: bool
    run: RunResult


@dataclass(slots=True)
class GradeResult:
    status: str
    passed: int
    total: int
    pass_rate: float
    failing_input: str | None
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.total > 0 and self.passed == self.total


@dataclass(slots=True)
class Message:
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None
    trainable: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = {"role": self.role, "trainable": self.trainable}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.name is not None:
            result["name"] = self.name
        return result


def validate_tool_call(call: dict[str, Any]) -> None:
    if set(call) != {"name", "arguments"}:
        raise ValueError("tool call must contain exactly name and arguments")
    name = call["name"]
    args = call["arguments"]
    expected = {"run_candidate": {"input"}, "submit": {"code"}}
    if name not in expected or not isinstance(args, dict) or set(args) != expected[name]:
        raise ValueError("invalid tool name or argument keys")
    value = args[next(iter(expected[name]))]
    if not isinstance(value, str) or (name == "submit" and not value.strip()):
        raise ValueError("tool argument must be a valid non-empty string")

