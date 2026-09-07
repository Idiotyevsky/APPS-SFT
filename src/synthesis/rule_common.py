"""Shared helpers for rule-based (deterministic) SFT builders.

Everything here is offline: candidates come from AST mutations of twice-verified
APPS references and observations come from real local executions. No model/API
is used and no counterfactual claim is made (labels are rule_design).
"""
from __future__ import annotations

import json

from .apps_loader import RejectedProblem, clean_problem
from .grader import PrivateGrader, failure_signature
from .mutation import (
    apply_edits, enumerate_supported_single_edits,
    sample_compatible_sets,
)
from .mutation.validate import validate_multi_execution
from .normalize import normalized_code_hash
from .sandbox import SandboxConfig, SandboxedExecutor
from .schemas import CleanProblem


def make_grader() -> PrivateGrader:
    return PrivateGrader(SandboxedExecutor(
        SandboxConfig(timeout_sec=3, memory_mb=512, backend="local"),
    ))


def clean(gr: PrivateGrader, raw: dict):
    """Return a CleanProblem or None (rejected)."""
    problem = clean_problem(raw, gr)
    return None if isinstance(problem, RejectedProblem) else problem


def grade(gr: PrivateGrader, problem: CleanProblem, code: str):
    public = problem.public_problem
    private = problem.private_evaluation
    return gr.grade(
        code, private.inputs, private.outputs,
        public.io_mode, public.fn_name,
    )


def curate_problem(gr: PrivateGrader, problem: CleanProblem):
    """Return entry dict {ref, singles, multis} or None.

    Verified reference must pass; single/multi mutants must be stable
    partial failures (0 < pass_rate < 1). Multi entries also store a
    rule-constructed partial fix (all-but-first edit applied).
    """
    public = problem.public_problem
    private = problem.private_evaluation
    ref = private.selected_reference_code or ""
    ref_grade = grade(gr, problem, ref)
    if not ref_grade.accepted:
        return None
    ref_artifact = {
        "id": problem.problem_id,
        "code": ref,
        "origin": "verified_reference",
        "code_hash": normalized_code_hash(ref),
        "seed_submit": {
            "status": ref_grade.status, "passed": ref_grade.passed,
            "total": ref_grade.total, "pass_rate": ref_grade.pass_rate,
            "failing_input": None,
        },
        "failure_signature": None,
        "mutation": None,
    }

    singles: list[dict] = []
    seen = set()
    for edit in enumerate_supported_single_edits(ref):
        if len(singles) >= 4:
            break
        try:
            mutant, records = apply_edits(ref, [edit])
            compile(mutant, "<m>", "exec")
        except Exception:
            continue
        first = grade(gr, problem, mutant)
        if not (0 < first.pass_rate < 1):
            continue
        second = grade(gr, problem, mutant)
        if failure_signature(first) != failure_signature(second):
            continue
        code_hash = normalized_code_hash(mutant)
        if code_hash in seen:
            continue
        seen.add(code_hash)
        singles.append({
            "id": problem.problem_id,
            "code": mutant,
            "origin": "synthetic_single",
            "code_hash": code_hash,
            "seed_submit": {
                "status": first.status, "passed": first.passed,
                "total": first.total, "pass_rate": first.pass_rate,
                "failing_input": first.failing_input,
            },
            "failure_signature": failure_signature(first),
            "mutation": {
                "is_mutated": True, "bug_count": 1,
                "semantic_edit_count": 1,
                "all_individually_harmful": True,
                "survivor_check_passed": True,
                "masked_mutation": False,
                "edits": [record.to_dict() for record in records],
            },
        })

    multis: list[dict] = []
    seen_multi = set()
    edits = enumerate_supported_single_edits(ref)
    for bug_count in (2, 3):
        for edit_set in sample_compatible_sets(
            edits, bug_count, seed=0x5EED + bug_count, limit=10,
        ):
            if len(multis) >= 3:
                break
            evidence = validate_multi_execution(problem, edit_set, gr)
            if not evidence.valid or evidence.combined_result is None:
                continue
            mutant, records = apply_edits(ref, edit_set)
            code_hash = normalized_code_hash(mutant)
            if code_hash in seen_multi:
                continue
            seen_multi.add(code_hash)
            partial, _ = apply_edits(ref, edit_set[1:])
            result = evidence.combined_result
            multis.append({
                "id": problem.problem_id,
                "code": mutant,
                "origin": "synthetic_multi",
                "code_hash": code_hash,
                "partial_code": partial,
                "seed_submit": {
                    "status": result.status, "passed": result.passed,
                    "total": result.total,
                    "pass_rate": result.pass_rate,
                    "failing_input": result.failing_input,
                },
                "failure_signature": failure_signature(result),
                "mutation": {
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
            })
    return {"ref": ref_artifact, "singles": singles, "multis": multis}


def probe_input(gr: PrivateGrader, problem: CleanProblem, code: str) -> str | None:
    """Deterministic stdin probe on which the buggy candidate observably
    differs from the reference. Never equals a hidden test; provenance is
    rule_constructed."""
    public = problem.public_problem
    private = problem.private_evaluation
    ref = private.selected_reference_code or ""
    hidden = {str(value) for value in private.inputs}

    if public.io_mode == "call":
        patterns: list[str] = []
        for args in (
            [], [0], [1], [2], [0, 0], [1, 1], [1, 2], [2, 1],
            [2, 2], [1, 2, 3], [3, 2, 1], [0, 1, 2], [1, 1, 1],
            [2, 1, 1], [5], [7],
            [[1, 2, 3], 1], [[1, 2, 3], 2], [[1, 2], 1], [[2, 1], 1],
            [[1, 1], 1], [[1, 2, 3, 4, 5], 1], [1, [2]], [1, 2, [3]],
            [[0, 1], 2], [[3, 2, 1], 2], [1, [1, 2]],
        ):
            probe = json.dumps(args)
            if probe not in hidden:
                patterns.append(probe)
        for pattern in patterns:
            bug_run, _ = gr.run_candidate(code, pattern, "call", public.fn_name)
            if bug_run.status in {"invalid_input", "timeout", "output_limit"}:
                continue
            ref_run, _ = gr.run_candidate(ref, pattern, "call", public.fn_name)
            if ref_run.status != "ok":
                continue
            if (bug_run.stdout, bug_run.status) != (ref_run.stdout, ref_run.status):
                return pattern
        return None

    patterns: list[str] = []

    def add(value: str) -> None:
        if value not in hidden:
            patterns.append(value)

    for value in range(0, 9):
        add(f"{value}\n")
    for values in (
        [1, 2, 3], [5, 4, 3, 2, 1], [1, 1, 1], [1, 2, 2],
        [2, 2, 3, 3], [3, 3, 3], [7, 7, 1, 2],
        [1, 1, 2, 2, 3, 3], [4, 4, 4, 4, 1], [2, 2, 2, 1],
    ):
        add(f"{len(values)}\n{' '.join(map(str, values))}\n")
    for text in (
        "aa", "ab", "aba", "abab", "baa", "aaab", "ba", "b",
        "a", "zz", "abba", "aabb", "cc",
    ):
        for n in (2, 3, 4, 5, 7):
            add(f"{n} {len(text)}\n{text}\n")
    for words in (
        ["aa", "bb"], ["ab", "ab"], ["a", "a"], ["x", "y"],
        ["hariton", "hkariton"], ["buoi", "boooi", "bui"],
        ["k", "h"], ["ab", "aa", "ab"], ["aaaa", "aaaa"],
        ["o", "u", "k"],
    ):
        add(f"{len(words)}\n" + "\n".join(words) + "\n")
    for grid in (
        ["..", ".."], [".R", "R."], ["..R", "..."], ["RR", "RR"],
        ["..", "RR"], ["R.", ".R"], ["...", ".R.", "..."],
        [".RR", "RR.", ".R."],
    ):
        add(f"{len(grid)} {len(grid[0])}\n" + "\n".join(grid) + "\n")
    for pattern in patterns:
        bug_run, _ = gr.run_candidate(
            code, pattern, public.io_mode, public.fn_name,
        )
        if bug_run.status in {"invalid_input", "timeout", "output_limit"}:
            continue
        ref_run, _ = gr.run_candidate(
            ref, pattern, public.io_mode, public.fn_name,
        )
        if ref_run.status != "ok":
            continue
        if (bug_run.stdout, bug_run.status) != (ref_run.stdout, ref_run.status):
            return pattern
    return None
