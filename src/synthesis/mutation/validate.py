from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

from ..grader import PrivateGrader, failure_signature, same_grade
from ..schemas import CleanProblem, GradeResult
from .base import MutationEdit, apply_edits, compatible


def _dump(source: str) -> str:
    return ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)


def validate_single_ast_edit(reference: str, mutant: str, edit: MutationEdit) -> bool:
    try:
        expected, records = apply_edits(reference, [edit])
        compile(mutant, "<mutant>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    return len(records) == 1 and records[0].semantic_edit_count == 1 and _dump(expected) == _dump(mutant) and _dump(reference) != _dump(mutant)


def validate_multi_ast_edits(reference: str, mutant: str, edits: list[MutationEdit]) -> bool:
    if not 2 <= len(edits) <= 4:
        return False
    if any(edit.record.semantic_edit_count != 1 for edit in edits):
        return False
    for index, left in enumerate(edits):
        if any(not compatible(left, right) for right in edits[index + 1:]):
            return False
    try:
        expected, records = apply_edits(reference, edits)
        compile(mutant, "<mutant>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    return len(records) == len(edits) and _dump(expected) == _dump(mutant)


@dataclass(slots=True)
class MultiMutationEvidence:
    valid: bool
    all_individually_harmful: bool
    survivor_check_passed: bool
    stable_failure: bool
    masked_mutation: bool
    combined_result: GradeResult | None
    individual_signatures: list[str]


def validate_multi_execution(problem: CleanProblem, edits: list[MutationEdit], grader: PrivateGrader) -> MultiMutationEvidence:
    reference = problem.private_evaluation.selected_reference_code or ""
    mutant, _ = apply_edits(reference, edits)
    public, private = problem.public_problem, problem.private_evaluation
    individual_results = []
    for edit in edits:
        code, _ = apply_edits(reference, [edit])
        individual_results.append(grader.grade(code, private.inputs, private.outputs, public.io_mode, public.fn_name))
    harmful = all(not result.accepted for result in individual_results)
    survivor = True
    for index in range(len(edits)):
        remaining = edits[:index] + edits[index + 1:]
        code, _ = apply_edits(reference, remaining)
        if grader.grade(code, private.inputs, private.outputs, public.io_mode, public.fn_name).accepted:
            survivor = False
            break
    first = grader.grade(mutant, private.inputs, private.outputs, public.io_mode, public.fn_name)
    second = grader.grade(mutant, private.inputs, private.outputs, public.io_mode, public.fn_name)
    stable = same_grade(first, second)
    masked = not harmful or not survivor
    valid = validate_multi_ast_edits(reference, mutant, edits) and harmful and survivor and not first.accepted and stable and not masked
    return MultiMutationEvidence(valid, harmful, survivor, stable, masked, first, [failure_signature(value) for value in individual_results])

