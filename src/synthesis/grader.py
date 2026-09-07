from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any, Iterable

from .harness import build_call_harness, parse_call_stdout
from .normalize import compare_output, normalize_input
from .sandbox import SandboxedExecutor
from .schemas import CaseResult, GradeResult, IOMode, RunResult


class PrivateGrader:
    def __init__(self, executor: SandboxedExecutor | None = None):
        self.executor = executor or SandboxedExecutor()
        self.grade_calls = 0
        self.run_candidate_calls = 0

    def run_candidate(self, code: str, input_text: str, io_mode: str, fn_name: str | None = None) -> tuple[RunResult, Any]:
        self.run_candidate_calls += 1
        if io_mode not in {IOMode.STDIN.value, IOMode.CALL.value}:
            raise ValueError(f"unsupported I/O mode: {io_mode}")
        if not isinstance(code, str) or not isinstance(input_text, str):
            result = RunResult(
                status="invalid_input",
                error={"type": "TypeError", "message": "code and input must be strings"},
                exit_code=None,
            )
            return result, None
        if io_mode == IOMode.CALL.value:
            try:
                arguments = json.loads(input_text)
                if not isinstance(arguments, list):
                    raise ValueError("call input must be a JSON argument list")
            except (json.JSONDecodeError, ValueError) as exc:
                result = RunResult(status="invalid_input", error={"type": type(exc).__name__, "message": str(exc)}, exit_code=None)
                return result, None
            try:
                source = build_call_harness(code, fn_name or "")
            except ValueError as exc:
                result = RunResult(status="invalid_input", error={"type": type(exc).__name__, "message": str(exc)}, exit_code=None)
                return result, None
            result = self.executor.run(source, input_text)
            if result.status != "ok":
                return result, None
            try:
                actual, visible_stdout = parse_call_stdout(result.stdout)
            except (ValueError, json.JSONDecodeError) as exc:
                result.status = "runtime_error"
                result.error = {"type": type(exc).__name__, "message": str(exc)}
                return result, None
            result.stdout = visible_stdout + json.dumps(actual, ensure_ascii=False)
            return result, actual
        result = self.executor.run(code, input_text)
        return result, result.stdout

    def grade(self, code: str, inputs: Iterable[Any], outputs: Iterable[Any], io_mode: str, fn_name: str | None = None) -> GradeResult:
        if io_mode not in {IOMode.STDIN.value, IOMode.CALL.value}:
            raise ValueError(f"unsupported I/O mode: {io_mode}")
        input_list = list(inputs)
        self.grade_calls += 1
        output_list = list(outputs)
        if not input_list or len(input_list) != len(output_list):
            raise ValueError("grader requires equally sized non-empty inputs and outputs")
        try:
            compile(code, "<candidate>", "exec")
        except (SyntaxError, ValueError, TypeError):
            return GradeResult("compile_error", 0, len(input_list), 0.0, None, [])
        cases: list[CaseResult] = []
        first_failure: str | None = None
        overall_status = "accepted"
        for raw_input, expected in zip(input_list, output_list):
            normalized = normalize_input(raw_input, io_mode)
            run, actual = self.run_candidate(code, normalized, io_mode, fn_name)
            passed = run.status == "ok" and compare_output(actual, expected, io_mode)
            cases.append(CaseResult(normalized, expected, actual, passed, run))
            if not passed and first_failure is None:
                first_failure = normalized
                if run.status == "timeout":
                    overall_status = "timeout"
                elif run.status in {"runtime_error", "output_limit", "invalid_input"}:
                    overall_status = "runtime_error"
                else:
                    overall_status = "wrong_answer"
        passed_count = sum(case.passed for case in cases)
        if passed_count == len(cases):
            overall_status = "accepted"
            first_failure = None
        return GradeResult(overall_status, passed_count, len(cases), passed_count / len(cases), first_failure, cases)


def failure_signature(result: GradeResult) -> str:
    exception_type = next((case.run.error["type"] for case in result.cases if case.run.error), "")
    payload = json.dumps(
        {"status": result.status, "passed": result.passed, "total": result.total,
         "failing_input": result.failing_input, "exception_type": exception_type},
        sort_keys=True, separators=(",", ":"),
    )
    return "sha256:" + sha256(payload.encode()).hexdigest()


def same_grade(left: GradeResult, right: GradeResult) -> bool:
    return failure_signature(left) == failure_signature(right)

