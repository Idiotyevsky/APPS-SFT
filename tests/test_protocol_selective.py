import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "sft/scripts/train_protocol_selective.py"
SPEC = importlib.util.spec_from_file_location("train_protocol_selective", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _action(code):
    return {
        "from": "function_call",
        "value": json.dumps(
            {"name": "submit", "arguments": {"code": code}},
            ensure_ascii=False,
        ),
    }


def test_delta_ranges_use_json_escaped_offsets_and_include_change():
    previous = 'print("old")\n'
    target = 'print("新\\n")\n'
    ranges = MODULE._delta_body_ranges(previous, target, context_chars=1)
    escaped = json.dumps(target, ensure_ascii=False)[1:-1]

    assert ranges == sorted(ranges)
    assert all(0 <= left < right <= len(escaped) for left, right in ranges)
    assert any("新" in escaped[left:right] for left, right in ranges)


def test_delta_ranges_keep_context_for_pure_deletion():
    ranges = MODULE._delta_body_ranges("abcXdef", "abcdef", context_chars=0)

    assert ranges
    assert all(left < right for left, right in ranges)


def test_previous_submit_code_uses_latest_candidate():
    row = {
        "sample_id": "sample",
        "conversations": [
            _action("first"),
            {"from": "observation", "value": "failed"},
            _action("second"),
            _action("target"),
        ],
    }

    assert MODULE._previous_submit_code(row) == "second"


def test_full_repair_selection_is_stratified_and_deterministic():
    rows = []
    states = []
    for state in sorted(MODULE.REPAIR_SUBMIT_STATES):
        for index in range(5):
            rows.append({
                "sample_id": f"{state}-{index}",
                "conversations": [_action("candidate"), _action("target")],
            })
            states.append(state)

    selected_a, counts, selected_counts = MODULE._choose_full_repairs(
        rows, states, fraction=0.2, seed=42
    )
    selected_b, _, _ = MODULE._choose_full_repairs(
        rows, states, fraction=0.2, seed=42
    )

    assert selected_a == selected_b
    assert set(counts.values()) == {5}
    assert set(selected_counts.values()) == {1}
    assert len(selected_a) == len(MODULE.REPAIR_SUBMIT_STATES)
