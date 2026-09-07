from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import floor
from typing import Any, Iterable, Mapping

from .config import DIFFICULTY_ORDER, largest_remainder


CROSS_RATIOS: dict[tuple[str, str], float] = {
    ("synthetic_single", "post_submit_direct_repair"): 0.10,
    ("synthetic_single", "post_submit_failure_replay"): 0.10,
    ("synthetic_single", "pre_submit_active_validation"): 0.10,
    ("synthetic_multi", "post_submit_direct_repair"): 0.05,
    ("synthetic_multi", "post_submit_failure_replay"): 0.125,
    ("synthetic_multi", "pre_submit_active_validation"): 0.125,
    ("model_natural_failure", "post_submit_direct_repair"): 0.10,
    ("model_natural_failure", "post_submit_failure_replay"): 0.075,
    ("model_natural_failure", "pre_submit_active_validation"): 0.075,
    ("verified_reference", "direct_submission"): 0.10,
    ("model_natural_correct", "direct_submission"): 0.05,
}


def _configured_cross_ratios(
    source_ratios: dict[str, float] | None = None,
    behavior_ratios: dict[str, float] | None = None,
) -> dict[tuple[str, str], float]:
    if source_ratios is None and behavior_ratios is None:
        return dict(CROSS_RATIOS)
    if source_ratios is None or behavior_ratios is None:
        raise ValueError("source and behavior ratios must be supplied together")
    source_order = (
        "synthetic_single", "synthetic_multi",
        "model_natural_failure", "verified_reference",
        "model_natural_correct",
    )
    behavior_order = (
        "post_submit_direct_repair", "post_submit_failure_replay",
        "pre_submit_active_validation", "direct_submission",
    )
    if set(source_ratios) != set(source_order):
        raise ValueError("source ratios do not match candidate origins")
    if set(behavior_ratios) != set(behavior_order):
        raise ValueError("behavior ratios do not match behavior names")
    if abs(sum(source_ratios.values()) - 1.0) > 1e-6:
        raise ValueError("source ratios must sum to 1")
    if abs(sum(behavior_ratios.values()) - 1.0) > 1e-6:
        raise ValueError("behavior ratios must sum to 1")
    if any(value < 0 for value in source_ratios.values()):
        raise ValueError("source ratios must be non-negative")
    if any(value < 0 for value in behavior_ratios.values()):
        raise ValueError("behavior ratios must be non-negative")

    # Iterative proportional fitting preserves the documented cross-cell
    # structure while honoring explicitly configured source/behavior margins.
    cells = dict(CROSS_RATIOS)
    rows = {origin: [key for key in cells if key[0] == origin]
            for origin in source_order}
    columns = {behavior: [key for key in cells if key[1] == behavior]
               for behavior in behavior_order}
    for _ in range(200):
        for origin in source_order:
            keys = rows[origin]
            current = sum(cells[key] for key in keys)
            target = float(source_ratios[origin])
            if target and current <= 0:
                raise ValueError(
                    f"source origin {origin} has no compatible behavior cells"
                )
            for key in keys:
                cells[key] = (
                    cells[key] * target / current if current > 0 else 0.0
                )
        for behavior in behavior_order:
            keys = columns[behavior]
            current = sum(cells[key] for key in keys)
            target = float(behavior_ratios[behavior])
            if target and current <= 0:
                raise ValueError(
                    f"behavior {behavior} has no compatible source cells"
                )
            for key in keys:
                cells[key] = (
                    cells[key] * target / current if current > 0 else 0.0
                )
    normalizer = sum(cells.values())
    if normalizer <= 0:
        raise ValueError("configured cross ratios have zero mass")
    return {key: value / normalizer for key, value in cells.items()}


def cross_quotas(
    total: int,
    source_ratios: dict[str, float] | None = None,
    behavior_ratios: dict[str, float] | None = None,
) -> dict[tuple[str, str], int]:
    ratios = _configured_cross_ratios(source_ratios, behavior_ratios)
    string_ratios = {
        f"{origin}|{behavior}": ratio
        for (origin, behavior), ratio in ratios.items()
    }
    values = largest_remainder(total, string_ratios, tuple(string_ratios))
    return {
        tuple(key.split("|", 1)): count
        for key, count in values.items()
    }


def transportation_quotas(
    total: int, difficulty_ratios: dict[str, float],
    source_ratios: dict[str, float] | None = None,
    behavior_ratios: dict[str, float] | None = None,
) -> dict[tuple[str, str, str], int]:
    ratios = _configured_cross_ratios(source_ratios, behavior_ratios)
    rows = cross_quotas(total, source_ratios, behavior_ratios)
    columns = largest_remainder(total, difficulty_ratios, DIFFICULTY_ORDER)
    remaining_rows = dict(rows)
    remaining_columns = dict(columns)
    result: dict[tuple[str, str, str], int] = defaultdict(int)
    raw = {
        (origin, behavior, difficulty): total * ratios[(origin, behavior)] * difficulty_ratios[difficulty]
        for origin, behavior in rows for difficulty in difficulty_ratios
    }
    # Allocate one item at a time by greatest unmet ideal. This preserves exact row/column margins.
    for _ in range(total):
        choices = [
            (raw[(origin, behavior, difficulty)] - result[(origin, behavior, difficulty)], origin, behavior, difficulty)
            for origin, behavior in rows for difficulty in difficulty_ratios
            if remaining_rows[(origin, behavior)] > 0 and remaining_columns[difficulty] > 0
        ]
        if not choices:
            raise RuntimeError("could not construct quota transportation table")
        _, origin, behavior, difficulty = max(choices, key=lambda item: (item[0], item[1], item[2], item[3]))
        result[(origin, behavior, difficulty)] += 1
        remaining_rows[(origin, behavior)] -= 1
        remaining_columns[difficulty] -= 1
    return dict(result)
MULTI_BUG_RATIOS = {"2": 0.6, "3": 0.3, "4": 0.1}


def sampling_quotas(
    total: int,
    difficulty_ratios: dict[str, float],
    source_ratios: dict[str, float] | None = None,
    behavior_ratios: dict[str, float] | None = None,
    bug_ratios: Mapping[int | str, float] | None = None,
) -> dict[tuple[str, str, str, int | None], int]:
    base = transportation_quotas(
        total, difficulty_ratios, source_ratios, behavior_ratios,
    )
    result = {
        (origin, behavior, difficulty, None): count
        for (origin, behavior, difficulty), count in base.items()
        if origin != "synthetic_multi"
    }
    multi_rows = {
        key: count for key, count in base.items()
        if key[0] == "synthetic_multi"
    }
    multi_total = sum(multi_rows.values())
    configured_bug_ratios = {
        str(key): float(value)
        for key, value in (bug_ratios or MULTI_BUG_RATIOS).items()
    }
    bug_order = tuple(configured_bug_ratios)
    bug_targets = largest_remainder(
        multi_total, configured_bug_ratios, bug_order,
    )
    remaining_rows = dict(multi_rows)
    remaining_bugs = dict(bug_targets)
    allocated: dict[tuple[str, str, str, int], int] = defaultdict(int)
    raw = {
        (*key, int(bug)): count * configured_bug_ratios[bug]
        for key, count in multi_rows.items()
        for bug in configured_bug_ratios
    }
    for _ in range(multi_total):
        choices = [
            (
                raw[(*key, int(bug))]
                - allocated[(*key, int(bug))],
                key,
                bug,
            )
            for key in multi_rows
            for bug in configured_bug_ratios
            if remaining_rows[key] > 0 and remaining_bugs[bug] > 0
        ]
        if not choices:
            raise RuntimeError("multi-bug quota allocation became infeasible")
        _, key, bug = max(
            choices, key=lambda item: (item[0], item[1], item[2]),
        )
        allocated[(*key, int(bug))] += 1
        remaining_rows[key] -= 1
        remaining_bugs[bug] -= 1
    result.update(allocated)
    return result


@dataclass(slots=True)
class SamplingResult:
    selected: list[dict[str, Any]]
    shortfalls: list[dict[str, Any]]


def constrained_sample(
    episodes: Iterable[dict[str, Any]], total: int,
    difficulty_ratios: dict[str, float],
    max_per_problem: int = 6, max_problem_behavior: int = 2,
    max_problem_origin: int = 3,
    source_ratios: dict[str, float] | None = None,
    behavior_ratios: dict[str, float] | None = None,
    bug_ratios: Mapping[int | str, float] | None = None,
) -> SamplingResult:
    quotas = sampling_quotas(
        total, difficulty_ratios, source_ratios, behavior_ratios, bug_ratios,
    )
    buckets: dict[tuple[str, str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        metadata = episode["metadata"]
        origin = metadata["candidate"]["origin"]
        behavior = metadata["behavior_sequence"][0]
        difficulty = metadata["difficulty"]
        bug_count = (
            (metadata.get("mutation") or {}).get("bug_count")
            if origin == "synthetic_multi" else None
        )
        key = (origin, behavior, difficulty, bug_count)
        if key in quotas:
            buckets[key].append(episode)
    for values in buckets.values():
        values.sort(key=lambda item: (-float((item["metadata"].get("counterfactual") or {}).get("utility", 0)), item["id"]))

    selected: list[dict[str, Any]] = []
    problem_count: Counter[str] = Counter()
    problem_behavior: Counter[tuple[str, str]] = Counter()
    problem_origin: Counter[tuple[str, str]] = Counter()
    code_hashes: set[str] = set()
    shortfalls = []
    for key in sorted(quotas):
        origin, behavior, difficulty, bug_count = key
        need = quotas[key]
        for episode in buckets.get(key, []):
            problem = str(episode["metadata"]["id"])
            code_hash = episode["metadata"]["candidate"]["code_hash"]
            if code_hash in code_hashes or problem_count[problem] >= max_per_problem:
                continue
            if problem_behavior[(problem, behavior)] >= max_problem_behavior or problem_origin[(problem, origin)] >= max_problem_origin:
                continue
            selected.append(episode)
            code_hashes.add(code_hash)
            problem_count[problem] += 1
            problem_behavior[(problem, behavior)] += 1
            problem_origin[(problem, origin)] += 1
            need -= 1
            if need == 0:
                break
        if need:
            shortfalls.append({"origin": origin, "behavior": behavior, "difficulty": difficulty, "bug_count": bug_count, "missing": need, "available": len(buckets.get(key, []))})
    return SamplingResult(selected, shortfalls)

