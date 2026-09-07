from collections import Counter

from synthesis.config import SynthesisConfig
from synthesis.quota import cross_quotas, transportation_quotas


def test_gate_and_final_cross_quotas_are_exact():
    gate = cross_quotas(200)
    assert sum(gate.values()) == 200
    assert gate[("synthetic_multi", "post_submit_failure_replay")] == 25
    final = cross_quotas(1200)
    assert final[("synthetic_multi", "post_submit_failure_replay")] == 150
    assert final[("model_natural_correct", "direct_submission")] == 60


def test_transportation_has_exact_cross_and_difficulty_margins():
    config = SynthesisConfig(target_count=200)
    table = transportation_quotas(200, config.difficulty_ratios)
    rows = Counter()
    cols = Counter()
    for (origin, behavior, difficulty), count in table.items():
        rows[(origin, behavior)] += count
        cols[difficulty] += count
    assert dict(rows) == cross_quotas(200)
    assert cols == Counter({
        "introductory": 50,
        "interview": 90,
        "competition": 60,
    })
def test_sampling_quotas_enforce_multi_bug_count_margins():
    from synthesis.quota import sampling_quotas

    table = sampling_quotas(200, SynthesisConfig().difficulty_ratios)
    counts = Counter()
    for (origin, _behavior, _difficulty, bug_count), value in table.items():
        if origin == "synthetic_multi":
            counts[bug_count] += value
    assert counts == Counter({2: 36, 3: 18, 4: 6})
