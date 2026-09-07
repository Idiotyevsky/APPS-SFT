from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from .generation import TextGenerator, generate_artifact, parse_complete_code
from .grader import PrivateGrader, same_grade
from .normalize import normalized_code_hash
from .schemas import CleanProblem, GradeResult


@dataclass(slots=True)
class NaturalCandidate:
    code: str | None
    origin: str
    result: GradeResult | None
    seed: int
    model: str
    revision: str
    prompt_hash: str
    raw_hash: str
    reject_reason: str | None = None
    mutation: None = None


def natural_solution_prompt(problem: CleanProblem) -> str:
    public = problem.public_problem
    return (
        "Write a complete Python solution. Return only one Python code block.\n\n"
        f"Problem:\n{public.question}\n\nStarter code:\n{public.starter_code}\n\n"
        f"I/O mode: {public.io_mode}\nFunction name: {public.fn_name or '(none)'}\n{public.input_format_note}"
    )


def harvest_natural_candidates(problem: CleanProblem, generator: TextGenerator, seeds: Iterable[int], grader: PrivateGrader) -> list[NaturalCandidate]:
    prompt = natural_solution_prompt(problem)
    public, private = problem.public_problem, problem.private_evaluation
    seen: set[str] = set()
    collected: list[NaturalCandidate] = []
    for seed in seeds:
        artifact = generate_artifact(generator, prompt, seed)
        code = parse_complete_code(artifact.raw_output)
        if code is None:
            collected.append(NaturalCandidate(None, "discarded", None, seed, generator.name, generator.revision, artifact.prompt_hash, artifact.raw_hash, "unparseable_code"))
            continue
        code_hash = normalized_code_hash(code)
        if code_hash in seen:
            collected.append(NaturalCandidate(code, "discarded", None, seed, generator.name, generator.revision, artifact.prompt_hash, artifact.raw_hash, "duplicate_code"))
            continue
        seen.add(code_hash)
        first = grader.grade(code, private.inputs, private.outputs, public.io_mode, public.fn_name)
        second = grader.grade(code, private.inputs, private.outputs, public.io_mode, public.fn_name)
        if not same_grade(first, second):
            collected.append(NaturalCandidate(code, "discarded", first, seed, generator.name, generator.revision, artifact.prompt_hash, artifact.raw_hash, "nondeterministic"))
        else:
            origin = "model_natural_correct" if first.accepted else "model_natural_failure"
            collected.append(NaturalCandidate(code, origin, first, seed, generator.name, generator.revision, artifact.prompt_hash, artifact.raw_hash))
    return collected

