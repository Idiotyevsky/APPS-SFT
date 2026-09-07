from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import random

from .grader import PrivateGrader, failure_signature, same_grade
from .mutation import apply_edits, enumerate_supported_single_edits, sample_compatible_sets, validate_single_ast_edit
from .mutation.validate import validate_multi_execution
from .normalize import normalized_code_hash
from .schemas import CleanProblem


@dataclass(slots=True)
class CandidateArtifact:
    problem_id: str
    code: str
    origin: str
    code_hash: str
    seed_submit: dict[str, Any]
    failure_signature: str | None
    mutation: dict[str, Any] | None
    generator: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.problem_id, "code": self.code, "origin": self.origin,
            "code_hash": self.code_hash, "seed_submit": self.seed_submit,
            "failure_signature": self.failure_signature, "mutation": self.mutation,
            "generator": self.generator,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateArtifact":
        identifier = value.get("id", value.get("problem_id"))
        if identifier is None or isinstance(identifier, bool):
            raise ValueError("candidate is missing id")
        return cls(
            problem_id=str(identifier), code=str(value["code"]),
            origin=str(value["origin"]), code_hash=str(value["code_hash"]),
            seed_submit=dict(value["seed_submit"]),
            failure_signature=value.get("failure_signature"),
            mutation=value.get("mutation"), generator=value.get("generator"),
        )


def _internal_grade(result) -> dict[str, Any]:
    return {
        "status": result.status, "passed": result.passed, "total": result.total,
        "pass_rate": result.pass_rate, "failing_input": result.failing_input,
    }


def build_single_mutants(problem: CleanProblem, grader: PrivateGrader, maximum: int = 3, families: Iterable[str] | None = None) -> list[CandidateArtifact]:
    reference = problem.private_evaluation.selected_reference_code or ""
    public, private = problem.public_problem, problem.private_evaluation
    artifacts: list[CandidateArtifact] = []
    seen: set[str] = set()
    for edit in enumerate_supported_single_edits(reference, families):
        mutant, records = apply_edits(reference, [edit])
        if not validate_single_ast_edit(reference, mutant, edit):
            continue
        first = grader.grade(mutant, private.inputs, private.outputs, public.io_mode, public.fn_name)
        second = grader.grade(mutant, private.inputs, private.outputs, public.io_mode, public.fn_name)
        if not (0 < first.pass_rate < 1) or not same_grade(first, second):
            continue
        code_hash = normalized_code_hash(mutant)
        if code_hash in seen:
            continue
        seen.add(code_hash)
        artifacts.append(CandidateArtifact(
            problem.problem_id, mutant, "synthetic_single", code_hash, _internal_grade(first), failure_signature(first),
            {"is_mutated": True, "bug_count": 1, "semantic_edit_count": 1,
             "all_individually_harmful": True, "survivor_check_passed": True,
             "masked_mutation": False, "edits": [record.to_dict() for record in records]},
        ))
        if len(artifacts) >= maximum:
            break
    return artifacts


def build_multi_mutants(
    problem: CleanProblem,
    grader: PrivateGrader,
    bug_counts: Iterable[int] = (2, 3, 4),
    bug_weights: Iterable[float] = (0.6, 0.3, 0.1),
    maximum: int = 3,
    seed: int = 0,
    families: Iterable[str] | None = None,
) -> list[CandidateArtifact]:
    reference = problem.private_evaluation.selected_reference_code or ""
    edits = enumerate_supported_single_edits(reference, families)
    counts = list(bug_counts)
    weights = list(bug_weights)
    if len(counts) != len(weights) or not counts:
        raise ValueError("bug counts and weights must have equal length")
    rng = random.Random(seed)
    iterators = {
        count: iter(sample_compatible_sets(
            edits, count, seed=seed + count, limit=100,
        ))
        for count in counts
    }
    schedule = rng.choices(
        counts, weights=weights, k=max(100, maximum * 20),
    )
    # Ensure every requested count gets attempts even under a short run.
    schedule[:0] = counts
    artifacts: list[CandidateArtifact] = []
    seen: set[str] = set()
    for bug_count in schedule:
        try:
            edit_set = next(iterators[bug_count])
        except StopIteration:
            continue
        mutant, records = apply_edits(reference, edit_set)
        evidence = validate_multi_execution(problem, edit_set, grader)
        if not evidence.valid or evidence.combined_result is None:
            continue
        code_hash = normalized_code_hash(mutant)
        if code_hash in seen:
            continue
        seen.add(code_hash)
        result = evidence.combined_result
        artifacts.append(CandidateArtifact(
            problem.problem_id, mutant, "synthetic_multi",
            code_hash, _internal_grade(result), failure_signature(result),
            {
                "is_mutated": True,
                "bug_count": bug_count,
                "semantic_edit_count": bug_count,
                "all_individually_harmful": evidence.all_individually_harmful,
                "survivor_check_passed": evidence.survivor_check_passed,
                "masked_mutation": evidence.masked_mutation,
                "individual_failure_signatures": evidence.individual_signatures,
                "edits": [record.to_dict() for record in records],
                "all_tests_failed": result.pass_rate == 0.0,
            },
        ))
        if len(artifacts) >= maximum:
            return artifacts
    return artifacts

