from collections import Counter

from synthesis.splitter import difficulty_stratified_split


def test_apps_scale_split_has_exact_global_margins_and_no_overlap():
    difficulties = (
        ["introductory"] * 1000
        + ["interview"] * 2500
        + ["competition"] * 1500
    )
    records = [
        {"id": index, "difficulty": difficulty}
        for index, difficulty in enumerate(difficulties)
    ]
    split = difficulty_stratified_split(records, seed=42)
    assert {key: len(value) for key, value in split.items()} == {
        "sft": 1000, "rl": 3500, "behavior_dev": 500,
    }
    assert len(set().union(*map(set, split.values()))) == 5000
    assert difficulty_stratified_split(records, seed=42) == split
