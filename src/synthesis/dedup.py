from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class DedupResult:
    unique: list[dict[str, Any]]
    rejected: list[tuple[str, str]]


def deduplicate_episodes(episodes: Iterable[dict[str, Any]]) -> DedupResult:
    unique: list[dict[str, Any]] = []
    rejected: list[tuple[str, str]] = []
    ids: set[str] = set()
    code_hashes: set[tuple[str, str]] = set()
    mutation_sets: set[tuple[str, str]] = set()
    failure_counts: Counter[tuple[str, str]] = Counter()
    failure_variants: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for episode in episodes:
        episode_id = episode["id"]
        metadata = episode["metadata"]
        problem_id = str(metadata["id"])
        code_hash = metadata["candidate"]["code_hash"]
        mutation_hash = metadata.get("mutation_set_hash")
        failure = (metadata.get("seed_submit") or {}).get("failure_signature")
        origin = str(metadata.get("candidate", {}).get("origin"))
        behavior = str((metadata.get("behavior_sequence") or [""])[0])
        reason = None
        if episode_id in ids:
            reason = "duplicate_episode_id"
        elif (problem_id, code_hash) in code_hashes:
            reason = "duplicate_normalized_candidate_code"
        elif mutation_hash and (problem_id, mutation_hash) in mutation_sets:
            reason = "duplicate_ordered_mutation_set"
        elif failure:
            failure_key = (problem_id, failure)
            variants = failure_variants.setdefault(failure_key, set())
            if failure_counts[failure_key] >= 2:
                reason = "failure_signature_per_problem_limit"
            elif (origin, behavior) in variants:
                reason = "failure_signature_same_source_behavior"
        if reason:
            rejected.append((episode_id, reason))
            continue
        ids.add(episode_id)
        code_hashes.add((problem_id, code_hash))
        if mutation_hash:
            mutation_sets.add((problem_id, mutation_hash))
        if failure:
            failure_key = (problem_id, failure)
            failure_counts[failure_key] += 1
            failure_variants.setdefault(failure_key, set()).add(
                (origin, behavior)
            )
        unique.append(episode)
    return DedupResult(unique, rejected)
