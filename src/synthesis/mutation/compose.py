from __future__ import annotations

import random
from typing import Iterable, Iterator

from .base import MutationEdit, compatible


def build_compatibility_graph(edits: Iterable[MutationEdit]) -> dict[int, set[int]]:
    values = list(edits)
    graph = {index: set() for index in range(len(values))}
    for index, left in enumerate(values):
        for other in range(index + 1, len(values)):
            if compatible(left, values[other]):
                graph[index].add(other)
                graph[other].add(index)
    return graph


def sample_compatible_sets(edits: list[MutationEdit], size: int, seed: int = 0, limit: int | None = None) -> Iterator[list[MutationEdit]]:
    if size < 2:
        raise ValueError("multi mutation size must be >= 2")
    graph = build_compatibility_graph(edits)
    rng = random.Random(seed)
    starts = list(range(len(edits)))
    rng.shuffle(starts)
    yielded = 0

    def extend(chosen: list[int], candidates: list[int]):
        nonlocal yielded
        if limit is not None and yielded >= limit:
            return
        if len(chosen) == size:
            yielded += 1
            yield [edits[index] for index in chosen]
            return
        rng.shuffle(candidates)
        for position, candidate in enumerate(candidates):
            if all(candidate in graph[index] for index in chosen):
                future = [item for item in candidates[position + 1:] if item in graph[candidate]]
                yield from extend(chosen + [candidate], future)

    yield from extend([], starts)

