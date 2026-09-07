import pytest

from synthesis.classify import (
    CounterfactualStats, classify_post_submit,
    classify_pre_submit, validate_direct_submission,
)


def stats(without, with_run):
    return CounterfactualStats(
        3, without, with_run, [11, 22, 33],
        high=2 / 3, low=1 / 3, min_utility=1 / 3,
    )


def test_direct_repair_boundary():
    assert classify_post_submit(stats(2, 0), True, True) == (
        "post_submit_direct_repair"
    )


def test_failure_replay_thresholds_and_identity():
    assert classify_post_submit(stats(1, 2), True, True) == (
        "post_submit_failure_replay"
    )
    assert classify_post_submit(stats(1, 2), True, False) is None
    assert classify_post_submit(stats(2, 3), True, True) == (
        "post_submit_direct_repair"
    )


def test_active_validation_requires_provenance_and_utility():
    assert classify_pre_submit(stats(1, 2), True, True) == (
        "pre_submit_active_validation"
    )
    assert classify_pre_submit(stats(1, 2), False, True) is None
    assert classify_pre_submit(stats(1, 1), True, True) is None


def test_direct_submission_is_twice_verified_without_run():
    assert validate_direct_submission(True, True, False)
    assert not validate_direct_submission(True, False, False)
    assert not validate_direct_submission(True, True, True)


def test_paired_seed_evidence_is_strict():
    with pytest.raises(ValueError):
        CounterfactualStats(3, 0, 3, [1, 1, 2]).validate()
