from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

from .grader import PrivateGrader, same_grade
from .normalize import ast_node_count, normalize_input, parses_and_compiles
from .schemas import CleanProblem, IOMode, PrivateEvaluation, PublicProblem


@dataclass(slots=True)
class RejectedProblem:
    problem_id: str | None
    reject_reason: str
    detail: str = ""


def stream_apps_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {
                    "_parse_error": f"invalid_json:{exc.msg}",
                }
                continue
            if not isinstance(value, dict):
                yield line_number, {
                    "_parse_error": "record_not_object",
                }
                continue
            yield line_number, value


def parse_apps_record_strict(raw: dict[str, Any]) -> dict[str, Any]:
    # The raw Hugging Face JSONL and all serialized project records use id.
    # Accept problem_id only as a legacy input alias.
    identifier = raw.get("id", raw.get("problem_id"))
    if identifier is None or isinstance(identifier, bool):
        raise ValueError("missing_id")
    if not isinstance(raw.get("question"), str) or not raw["question"].strip():
        raise ValueError("empty_question")
    difficulty = raw.get("difficulty")
    if difficulty not in {"introductory", "interview", "competition"}:
        raise ValueError("invalid_difficulty")
    starter = raw.get("starter_code")
    if starter is not None and not isinstance(starter, str):
        raise ValueError("invalid_starter_code")
    url = raw.get("url")
    if url is not None and not isinstance(url, str):
        raise ValueError("invalid_url")
    try:
        solutions = json.loads(raw.get("solutions", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_solutions_json") from exc
    try:
        input_output = json.loads(raw.get("input_output", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_input_output_json") from exc
    if not isinstance(solutions, list) or not solutions or not all(isinstance(code, str) and code.strip() for code in solutions):
        raise ValueError("invalid_solutions")
    if not isinstance(input_output, dict):
        raise ValueError("invalid_input_output")
    inputs, outputs = input_output.get("inputs"), input_output.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
        raise ValueError("invalid_tests")
    fn_name = input_output.get("fn_name")
    if fn_name is not None and (not isinstance(fn_name, str) or not fn_name.strip()):
        raise ValueError("invalid_fn_name")
    record = dict(raw)
    record.pop("problem_id", None)
    record.update({
        "id": str(identifier),
        "solutions": solutions,
        "input_output": input_output,
    })
    return record


def clean_problem(raw: dict[str, Any], grader: PrivateGrader) -> CleanProblem | RejectedProblem:
    raw_identifier = raw.get("id", raw.get("problem_id"))
    problem_id = str(raw_identifier) if raw_identifier is not None else None
    try:
        record = parse_apps_record_strict(raw)
    except ValueError as exc:
        return RejectedProblem(problem_id, str(exc))
    spec = record["input_output"]
    mode = IOMode.CALL.value if spec.get("fn_name") else IOMode.STDIN.value
    try:
        for value in spec["inputs"]:
            normalize_input(value, mode)
        json.dumps(spec["outputs"], ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        return RejectedProblem(record["id"], "unusable_io", str(exc))
    verified: list[tuple[int, str]] = []
    for index, code in enumerate(record["solutions"]):
        if not parses_and_compiles(code):
            continue
        first = grader.grade(code, spec["inputs"], spec["outputs"], mode, spec.get("fn_name"))
        second = grader.grade(code, spec["inputs"], spec["outputs"], mode, spec.get("fn_name"))
        if first.accepted and second.accepted and same_grade(first, second):
            verified.append((index, code))
    if not verified:
        return RejectedProblem(record["id"], "no_verified_reference")
    selected_index, selected_code = min(verified, key=lambda item: (ast_node_count(item[1]), item[0]))
    public = PublicProblem(
        question=record["question"], starter_code=record.get("starter_code") or "",
        difficulty=record["difficulty"], io_mode=mode,
        input_format_note=("Pass one complete stdin string to run_candidate(input)." if mode == "stdin" else "Pass a canonical JSON argument list to run_candidate(input)."),
        fn_name=spec.get("fn_name"),
    )
    private = PrivateEvaluation(
        inputs=spec["inputs"], outputs=spec["outputs"],
        verified_reference_indices=[item[0] for item in verified],
        selected_reference_index=selected_index, selected_reference_code=selected_code,
    )
    return CleanProblem(
        problem_id=record["id"], public_problem=public, private_evaluation=private,
        source={"dataset": "codeparrot/apps", "split": "train", "url": record.get("url")},
        solutions=record["solutions"],
    )

