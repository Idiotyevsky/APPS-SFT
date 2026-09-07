from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable, Mapping

from .config import largest_remainder


POOL_RATIOS = {"sft": 0.2, "rl": 0.7, "behavior_dev": 0.1}


def _transport_counts(
    group_sizes: dict[str, int],
) -> dict[tuple[str, str], int]:
    total = sum(group_sizes.values())
    pool_targets = largest_remainder(
        total, POOL_RATIOS, ("sft", "rl", "behavior_dev"),
    )
    remaining_groups = dict(group_sizes)
    remaining_pools = dict(pool_targets)
    result: dict[tuple[str, str], int] = defaultdict(int)
    raw = {
        (difficulty, pool): size * POOL_RATIOS[pool]
        for difficulty, size in group_sizes.items()
        for pool in POOL_RATIOS
    }
    for _ in range(total):
        choices = [
            (
                raw[(difficulty, pool)] - result[(difficulty, pool)],
                difficulty,
                pool,
            )
            for difficulty in group_sizes
            for pool in POOL_RATIOS
            if remaining_groups[difficulty] > 0
            and remaining_pools[pool] > 0
        ]
        if not choices:
            raise RuntimeError("split allocation became infeasible")
        _, difficulty, pool = max(
            choices, key=lambda item: (item[0], item[1], item[2]),
        )
        result[(difficulty, pool)] += 1
        remaining_groups[difficulty] -= 1
        remaining_pools[pool] -= 1
    return dict(result)


def difficulty_stratified_split(
    records: Iterable[Mapping[str, object]], seed: int = 42,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        identifier = record.get("id", record.get("problem_id"))
        if identifier is None or isinstance(identifier, bool):
            raise ValueError("record is missing id")
        problem_id = str(identifier)
        if problem_id in seen:
            raise ValueError(f"duplicate id: {problem_id}")
        seen.add(problem_id)
        difficulty = str(record.get("difficulty") or "unknown")
        groups[difficulty].append(problem_id)
    counts = _transport_counts({
        difficulty: len(ids) for difficulty, ids in groups.items()
    })
    result = {name: [] for name in POOL_RATIOS}
    for difficulty in sorted(groups):
        ids = groups[difficulty]
        rng = random.Random(f"{seed}:{difficulty}")
        rng.shuffle(ids)
        cursor = 0
        for pool in POOL_RATIOS:
            count = counts.get((difficulty, pool), 0)
            result[pool].extend(ids[cursor:cursor + count])
            cursor += count
        if cursor != len(ids):
            raise RuntimeError("split allocation did not consume a stratum")
    assert_split_disjoint(result)
    if sum(map(len, result.values())) != len(seen):
        raise RuntimeError("split allocation lost problem ids")
    return result


def assert_split_disjoint(splits: Mapping[str, Iterable[str]]) -> None:
    sets = {name: set(values) for name, values in splits.items()}
    names = list(sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"split overlap between {left} and {right}: {sorted(overlap)[:3]}")

