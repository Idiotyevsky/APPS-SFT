from __future__ import annotations

from dataclasses import dataclass

from .schemas import Behavior


@dataclass(slots=True)
class CounterfactualStats:
    k: int
    without_run_successes: int
    with_run_successes: int
    paired_seeds: list[int]
    high: float = 2 / 3
    low: float = 1 / 3
    min_utility: float = 1 / 3

    @property
    def without_run_rate(self) -> float:
        return self.without_run_successes / self.k

    @property
    def with_run_rate(self) -> float:
        return self.with_run_successes / self.k

    @property
    def utility(self) -> float:
        return self.with_run_rate - self.without_run_rate

    def validate(self) -> None:
        if self.k <= 0 or len(self.paired_seeds) != self.k or len(set(self.paired_seeds)) != self.k:
            raise ValueError("counterfactual samples require k distinct paired seeds")
        if not (0 <= self.without_run_successes <= self.k and 0 <= self.with_run_successes <= self.k):
            raise ValueError("success count outside [0,k]")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "k": self.k, "without_run_successes": self.without_run_successes,
            "without_run_rate": self.without_run_rate, "with_run_successes": self.with_run_successes,
            "with_run_rate": self.with_run_rate, "utility": self.utility,
            "threshold_high": self.high, "threshold_low": self.low,
            "min_utility": self.min_utility, "paired_seeds": self.paired_seeds,
        }


def classify_post_submit(stats: CounterfactualStats, has_failing_input: bool, replay_input_matches: bool) -> str | None:
    stats.validate()
    if stats.without_run_rate >= stats.high:
        return Behavior.POST_SUBMIT_DIRECT_REPAIR.value
    if (
        has_failing_input and replay_input_matches and stats.without_run_rate <= stats.low
        and stats.with_run_rate >= stats.high and stats.utility >= stats.min_utility
    ):
        return Behavior.POST_SUBMIT_FAILURE_REPLAY.value
    return None


def classify_pre_submit(stats: CounterfactualStats, provenance_valid: bool, input_executable: bool) -> str | None:
    stats.validate()
    if (
        provenance_valid and input_executable and stats.without_run_rate <= stats.low
        and stats.with_run_rate >= stats.high and stats.utility >= stats.min_utility
    ):
        return Behavior.PRE_SUBMIT_ACTIVE_VALIDATION.value
    return None


def validate_direct_submission(first_accepted: bool, second_accepted: bool, has_run: bool) -> bool:
    return first_accepted and second_accepted and not has_run

