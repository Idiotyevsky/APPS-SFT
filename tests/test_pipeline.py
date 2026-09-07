import json

from synthesis.config import SynthesisConfig
from synthesis.pipeline import (
    generate_candidates, prepare_apps, synthesize,
)
from synthesis.qa import run_qa
from synthesis.io_utils import read_jsonl


def test_prepare_to_direct_submission_pipeline(tmp_path):
    apps = tmp_path / "train.jsonl"
    records = []
    for problem_id in range(99, 104):
        records.append({
            "id": problem_id,
            "question": "Echo the input.",
            "solutions": json.dumps(["print(input())\n"]),
            "input_output": json.dumps({
                "inputs": ["hello\n"], "outputs": ["hello\n"],
            }),
            "difficulty": "introductory",
            "starter_code": "",
        })
    apps.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    config = SynthesisConfig(
        target_count=1,
        sandbox_backend="local",
        candidate_sources=("synthetic_single", "synthetic_multi"),
        source_ratios={
            "synthetic_single": 0.0,
            "synthetic_multi": 0.0,
            "model_natural_failure": 0.0,
            "verified_reference": 1.0,
            "model_natural_correct": 0.0,
        },
        behavior_ratios={
            "post_submit_direct_repair": 0.0,
            "post_submit_failure_replay": 0.0,
            "pre_submit_active_validation": 0.0,
            "direct_submission": 1.0,
        },
        difficulty_ratios={
            "introductory": 1.0, "interview": 0.0,
            "competition": 0.0,
        },
    )
    output = tmp_path / "out"
    assert prepare_apps(apps, output, config)["cleaned"] == 5
    counts = generate_candidates(output, config)
    assert counts["verified_reference"] == 1
    manifest = synthesize(output, config)
    assert manifest["accepted_episodes"] == 1
    assert len(list(read_jsonl(output / "sft_messages.jsonl"))) == 1
    report = run_qa(output, target_count=1, strict_quota=True, replay_config=config)
    assert report.passed, [issue.to_dict() for issue in report.issues]
