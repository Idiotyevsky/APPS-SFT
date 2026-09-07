from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from .classify import CounterfactualStats, classify_post_submit, classify_pre_submit
from .config import SynthesisConfig
from .feedback_projection import validate_public_feedback
from .export_sft import validate_decoded_supervision
from .io_utils import atomic_write_json, read_jsonl
from .quota import cross_quotas, sampling_quotas
from .metadata_schema import validate_metadata
from .normalize import normalized_code_hash
from .schemas import Behavior, validate_tool_call
from .replay import replay_artifacts
from .splitter import assert_split_disjoint
from .tools import TOOL_SCHEMAS


P0_CODES = {
    "oracle_leakage", "buggy_code_trainable", "final_not_accepted",
    "split_overlap", "tool_contract", "active_provenance",
    "manual_audit_severe",
    "replay_input_mismatch",
}
INTERNAL_FIELD_MARKERS = (
    '"expected":', '"actual":', '"test_index":', '"reference_solution":',
    '"source_span":', '"mutation_operator":',
)


@dataclass(slots=True)
class QAIssue:
    code: str
    message: str
    episode_id: str | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "message": self.message,
            "episode_id": self.episode_id, "severity": self.severity,
        }


@dataclass(slots=True)
class QAReport:
    passed: bool
    counts: dict[str, Any]
    issues: list[QAIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed, "counts": self.counts,
            "issues": [item.to_dict() for item in self.issues],
        }


def _issue(
    issues: list[QAIssue], code: str, message: str,
    episode_id: str | None = None,
) -> None:
    severity = "p0" if code in P0_CODES else "error"
    issues.append(QAIssue(code, message, episode_id, severity))


def _tool_calls(messages: list[dict[str, Any]]):
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):
                yield index, message, call


def _check_counterfactual(
    metadata: dict[str, Any], behavior: str, issues: list[QAIssue],
    episode_id: str,
) -> None:
    if metadata.get("label_method") == "rule_design":
        return
    value = metadata.get("counterfactual")
    if behavior == Behavior.DIRECT_SUBMISSION.value:
        if value is not None:
            _issue(issues, "counterfactual", "direct submission has counterfactual data", episode_id)
        return
    if not isinstance(value, dict):
        _issue(issues, "counterfactual", "missing counterfactual evidence", episode_id)
        return
    try:
        stats = CounterfactualStats(
            k=value["k"],
            without_run_successes=value["without_run_successes"],
            with_run_successes=value["with_run_successes"],
            paired_seeds=value["paired_seeds"],
            high=value["threshold_high"],
            low=value["threshold_low"],
            min_utility=value["min_utility"],
        )
        if behavior == Behavior.PRE_SUBMIT_ACTIVE_VALIDATION.value:
            classified = classify_pre_submit(stats, True, True)
        else:
            classified = classify_post_submit(stats, True, True)
        if classified != behavior:
            _issue(
                issues, "counterfactual",
                f"statistics classify as {classified!r}, not {behavior}", episode_id,
            )
    except (KeyError, TypeError, ValueError) as exc:
        _issue(issues, "counterfactual", f"invalid evidence: {exc}", episode_id)


def validate_episode_static(
    episode: dict[str, Any], issues: list[QAIssue],
) -> None:
    if not isinstance(episode, dict):
        _issue(issues, "schema", "episode must be an object")
        return
    episode_id = str(episode.get("id") or "")
    metadata = episode.get("metadata") or {}
    messages = episode.get("messages") or []
    if not isinstance(metadata, dict):
        _issue(issues, "metadata_schema", "metadata must be an object", episode_id)
        return
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) for message in messages
    ):
        _issue(issues, "schema", "messages must be a list of objects", episode_id)
        return
    for error in validate_metadata(metadata):
        _issue(issues, "metadata_schema", error, episode_id)
    if episode.get("tools") != TOOL_SCHEMAS:
        _issue(
            issues, "tool_contract",
            "tool schemas differ from the fixed contract", episode_id,
        )
    calls: list[dict[str, Any]] = []
    for index, message, call in _tool_calls(messages):
        try:
            validate_tool_call(call)
        except (ValueError, TypeError) as exc:
            _issue(issues, "tool_contract", str(exc), episode_id)
            continue
        calls.append(call)
        if message.get("trainable") and call["name"] == "submit":
            if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
                _issue(
                    issues, "tool_contract",
                    "submit lacks adjacent tool response", episode_id,
                )
            else:
                try:
                    response = json.loads(messages[index + 1]["content"])
                    if response.get("status") != "accepted":
                        _issue(
                            issues, "buggy_code_trainable",
                            "a failed submit is trainable", episode_id,
                        )
                except (TypeError, json.JSONDecodeError):
                    _issue(
                        issues, "tool_contract",
                        "submit response is invalid JSON", episode_id,
                    )
    for message in messages:
        if message.get("trainable") and message.get("role") != "assistant":
            _issue(
                issues, "loss_mask",
                "non-assistant message marked trainable", episode_id,
            )
        if message.get("role") == "tool" and message.get("name") == "submit":
            try:
                validate_public_feedback(json.loads(message["content"]))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                _issue(
                    issues, "oracle_leakage",
                    f"invalid public submit response: {exc}", episode_id,
                )
        elif message.get("role") == "tool" and message.get("name") == "run_candidate":
            try:
                observation = json.loads(message["content"])
                if set(observation) != {
                    "status", "stdout", "stderr", "error",
                    "exit_code", "truncated",
                }:
                    raise ValueError(
                        "run response fields differ from contract"
                    )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                _issue(
                    issues, "tool_contract",
                    f"invalid run response: {exc}", episode_id,
                )
        elif message.get("role") != "tool":
            visible = message.get("content")
            if isinstance(visible, str) and any(
                marker in visible for marker in INTERNAL_FIELD_MARKERS
            ):
                _issue(
                    issues, "oracle_leakage",
                    "internal field marker appears in visible content", episode_id,
                )
    if (
        not messages or messages[-1].get("role") != "tool"
        or messages[-1].get("name") != "submit"
    ):
        _issue(
            issues, "final_not_accepted",
            "episode does not end in submit response", episode_id,
        )
    else:
        try:
            final = json.loads(messages[-1]["content"])
            if final.get("status") != "accepted" or final.get("pass_rate") != 1.0:
                _issue(
                    issues, "final_not_accepted",
                    "final submit is not accepted", episode_id,
                )
            final_call = next(
                (
                    call for message in reversed(messages[:-1])
                    if message.get("role") == "assistant"
                    for call in reversed(message.get("tool_calls", []))
                    if call.get("name") == "submit"
                ),
                None,
            )
            if final_call is None:
                _issue(
                    issues, "final_not_accepted",
                    "accepted response has no preceding submit call",
                    episode_id,
                )
            elif (
                isinstance(final_call.get("arguments"), dict)
                and isinstance(final_call["arguments"].get("code"), str)
                and metadata.get("final_code_hash")
                != normalized_code_hash(final_call["arguments"]["code"])
            ):
                _issue(
                    issues, "final_not_accepted",
                    "metadata final_code_hash differs from final submit code",
                    episode_id,
                )
        except (TypeError, json.JSONDecodeError):
            _issue(
                issues, "final_not_accepted",
                "final submit response is invalid JSON", episode_id,
            )

    sequence = metadata.get("behavior_sequence") or []
    behavior = sequence[0] if sequence else None
    names = [call.get("name") for call in calls]
    trainable_names = [
        call.get("name")
        for message in messages
        if message.get("role") == "assistant" and message.get("trainable")
        for call in (message.get("tool_calls") or [])
    ]
    if behavior == Behavior.POST_SUBMIT_DIRECT_REPAIR.value:
        if (
            not names or names[-1] != "submit"
            or "run_candidate" in names
            or trainable_names != ["submit"]
        ):
            _issue(
                issues, "behavior_protocol",
                "direct repair may carry masked context submits but must "
                "never run and must end with a single trainable submit",
                episode_id,
            )
    elif behavior == Behavior.POST_SUBMIT_FAILURE_REPLAY.value:
        if (
            not names or names[-1] != "submit"
            or "run_candidate" not in names
            or not trainable_names
            or trainable_names[0] != "run_candidate"
        ):
            _issue(
                issues, "behavior_protocol",
                "failure replay may carry a masked submit prefix but its first "
                "trainable action must be run_candidate and it must end with "
                "submit", episode_id,
            )
        query = metadata.get("execution_query") or {}
        if not query.get("matches_submit_failing_input"):
            _issue(
                issues, "replay_input_mismatch",
                "failing-input byte identity is not proven", episode_id,
            )
    elif behavior == Behavior.PRE_SUBMIT_ACTIVE_VALIDATION.value:
        if (
            not names or names[-1] != "submit"
            or "run_candidate" not in names
            or not trainable_names
            or trainable_names[0] != "run_candidate"
        ):
            _issue(
                issues, "behavior_protocol",
                "active validation must run first and submit last", episode_id,
            )
        query = metadata.get("execution_query") or {}
        expected_fields = [
            "question", "starter_code", "io_mode",
            "input_format_note", "fn_name", "candidate",
        ]
        rule_design = metadata.get("label_method") == "rule_design"
        if rule_design:
            if (
                query.get("kind") not in {"model_proposed", "rule_constructed"}
                or not query.get("observable_fields")
                or any(
                    field not in expected_fields
                    for field in query.get("observable_fields")
                )
            ):
                _issue(
                    issues, "active_provenance",
                    "active input lacks observable-only provenance", episode_id,
                )
        elif (
            query.get("kind") != "model_proposed"
            or query.get("observable_fields") != expected_fields
        ):
            _issue(
                issues, "active_provenance",
                "active input lacks observable-only provenance", episode_id,
            )
    elif behavior == Behavior.DIRECT_SUBMISSION.value:
        if names != ["submit"] or trainable_names != ["submit"]:
            _issue(
                issues, "behavior_protocol",
                "direct submission must contain only a trainable submit",
                episode_id,
            )
    else:
        _issue(issues, "schema", "unknown or missing behavior", episode_id)

    _check_counterfactual(metadata, behavior, issues, episode_id)
    candidate = metadata.get("candidate") or {}
    origin = candidate.get("origin")
    mutation = metadata.get("mutation")
    if origin in {"model_natural_failure", "model_natural_correct"}:
        if mutation is not None:
            _issue(
                issues, "mutation",
                "natural candidate has fabricated mutation metadata", episode_id,
            )
    if origin == "synthetic_single":
        edits = mutation.get("edits") if isinstance(mutation, dict) else None
        if (
            not isinstance(mutation, dict)
            or mutation.get("bug_count") != 1
            or not isinstance(edits, list)
            or len(edits) != 1
        ):
            _issue(
                issues, "mutation",
                "single mutant lacks exact one-edit evidence", episode_id,
            )
    if origin == "synthetic_multi":
        edits = mutation.get("edits") if isinstance(mutation, dict) else None
        invalid = (
            not isinstance(mutation, dict)
            or mutation.get("bug_count") not in {2, 3, 4}
            or mutation.get("semantic_edit_count") != mutation.get("bug_count")
            or not isinstance(edits, list)
            or len(edits) != mutation.get("bug_count")
            or not mutation.get("all_individually_harmful")
            or not mutation.get("survivor_check_passed")
            or mutation.get("masked_mutation")
        )
        if invalid:
            _issue(
                issues, "mutation",
                "multi mutant lacks edit-count/harmfulness/survivor evidence",
                episode_id,
            )


def validate_tokenized(
    path: Path, issues: list[QAIssue],
    sft_by_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            _issue(issues, "loss_mask", "tokenized row must be an object")
            continue
        row_id = row.get("id")
        input_ids = row.get("input_ids")
        attention = row.get("attention_mask")
        labels = row.get("labels")
        valid_lists = all(
            isinstance(item, list) for item in (input_ids, attention, labels)
        )
        if not valid_lists or not (
            len(input_ids) == len(attention) == len(labels)
        ):
            _issue(
                issues, "loss_mask",
                "token arrays have inconsistent shape", row_id,
            )
            continue
        if not all(
            isinstance(token, int) and not isinstance(token, bool)
            for token in input_ids
        ) or not all(
            isinstance(mask, int) and not isinstance(mask, bool)
            for mask in attention
        ) or not all(
            isinstance(label, int) and not isinstance(label, bool)
            for label in labels
        ):
            _issue(
                issues, "loss_mask",
                "token arrays contain non-integer values", row_id,
            )
        if not any(label != -100 for label in labels):
            _issue(
                issues, "loss_mask",
                "sample has no supervised tokens", row_id,
            )
        if any(mask != 1 for mask in attention):
            _issue(
                issues, "loss_mask",
                "non-padding export hides context", row_id,
            )
        spans = row.get("spans", [])
        if not isinstance(spans, list):
            _issue(issues, "loss_mask", "spans must be a list", row_id)
            spans = []
        for span in spans:
            if not isinstance(span, dict):
                _issue(issues, "loss_mask", "span must be an object", row_id)
                continue
            start, end = span.get("start"), span.get("end")
            if (
                not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or start < 0 or end < start or end > len(labels)
                or not isinstance(span.get("role"), str)
                or not isinstance(span.get("trainable"), bool)
            ):
                _issue(issues, "loss_mask", "invalid token span", row_id)
                continue
            supervised = labels[start:end]
            expected = span["role"] == "assistant" and span["trainable"]
            actual = bool(supervised and all(label != -100 for label in supervised))
            if expected != actual:
                _issue(
                    issues, "loss_mask",
                    "span supervision differs from trainable flag", row_id,
                )
        supervised_text = row.get("supervised_text")
        if not isinstance(supervised_text, str):
            _issue(
                issues, "loss_mask",
                "missing readable supervised decode", row_id,
            )
        elif sft_by_id and row_id in sft_by_id:
            try:
                validate_decoded_supervision(
                    sft_by_id[row_id], supervised_text,
                )
            except (KeyError, TypeError, ValueError) as exc:
                _issue(
                    issues, "loss_mask", str(exc), row_id,
                )


def validate_manual_gate_audit(
    root: Path, episodes: list[dict[str, Any]],
    issues: list[QAIssue], minimum_reviews: int = 40,
) -> None:
    path = root / "manual_audit.jsonl"
    if not path.exists():
        _issue(
            issues, "manual_audit",
            f"gate-200 requires manual_audit.jsonl with at least {minimum_reviews} reviews",
        )
        return
    rows = list(read_jsonl(path))
    episode_by_id = {
        item.get("id"): item
        for item in episodes
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    unique: dict[str, dict[str, Any]] = {}
    required = {
        "episode_id", "reviewer",
        "observable_evidence_supported",
        "trajectory_natural", "severe_issue",
    }
    for row in rows:
        if not isinstance(row, dict):
            _issue(issues, "manual_audit", "audit row must be an object")
            continue
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str):
            _issue(
                issues, "manual_audit",
                "audit episode_id must be a string",
            )
            continue
        if episode_id not in episode_by_id:
            _issue(
                issues, "manual_audit",
                f"audit references unknown episode {episode_id!r}",
            )
            continue
        if episode_id in unique:
            _issue(
                issues, "manual_audit",
                f"duplicate manual audit for {episode_id}",
            )
            continue
        if (
            set(row) != required
            or not isinstance(row.get("reviewer"), str)
            or not row.get("reviewer").strip()
            or any(
                not isinstance(row.get(key), bool)
                for key in (
                    "observable_evidence_supported",
                    "trajectory_natural", "severe_issue",
                )
            )
        ):
            _issue(
                issues, "manual_audit",
                f"invalid audit schema for {episode_id}",
            )
            continue
        unique[episode_id] = row
        if row["severe_issue"]:
            _issue(
                issues, "manual_audit_severe",
                "manual reviewer found a severe issue", episode_id,
            )
    if len(unique) < minimum_reviews:
        _issue(
            issues, "manual_audit",
            f"expected at least {minimum_reviews} unique reviews, "
            f"found {len(unique)}",
        )

    def behavior_of(episode: dict[str, Any]) -> str | None:
        metadata = episode.get("metadata")
        sequence = metadata.get("behavior_sequence") if isinstance(metadata, dict) else None
        return sequence[0] if isinstance(sequence, list) and sequence and isinstance(sequence[0], str) else None

    behavior_counts = Counter(
        behavior_of(episode_by_id[key])
        for key in unique
    )
    for behavior in (item.value for item in Behavior):
        available = sum(
            behavior_of(episode) == behavior for episode in episode_by_id.values()
        )
        required_count = min(10, available)
        if behavior_counts[behavior] < required_count:
            _issue(
                issues, "manual_audit",
                f"{behavior} requires {required_count} reviews, "
                f"found {behavior_counts[behavior]}",
            )
    positive = sum(
        row["observable_evidence_supported"]
        and row["trajectory_natural"]
        and not row["severe_issue"]
        for row in unique.values()
    )
    if unique and positive / len(unique) < 0.95:
        _issue(
            issues, "manual_audit",
            f"positive audit rate {positive / len(unique):.3f} is below 0.95",
        )
def validate_resource_budget(
    root: Path, config: SynthesisConfig,
    accepted_count: int, issues: list[QAIssue],
) -> None:
    path = root / "run_metrics.json"
    if not path.exists():
        _issue(
            issues, "resource_budget",
            "gate-200 requires run_metrics.json",
        )
        return
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        _issue(issues, "resource_budget", f"invalid run_metrics.json: {exc}")
        return
    if not isinstance(metrics, dict):
        _issue(issues, "resource_budget", "run_metrics.json must contain an object")
        return

    def metric_float(value: Any, label: str, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            _issue(issues, "resource_budget", f"{label} is not numeric")
            return default
        if not math.isfinite(number) or number < 0:
            _issue(issues, "resource_budget", f"{label} is not finite/non-negative")
            return default
        return number

    stages = metrics.get("stages", {})
    if not isinstance(stages, dict):
        _issue(issues, "resource_budget", "run_metrics.stages must be an object")
        stages = {}
    wall_seconds = 0.0
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            _issue(issues, "resource_budget", f"stage {stage_name!r} is not an object")
            continue
        wall_seconds += metric_float(
            stage.get("wall_seconds", 0.0),
            f"stage {stage_name!r}.wall_seconds",
        )
    wall_hours = wall_seconds / 3600
    if wall_hours > config.max_wall_hours:
        _issue(
            issues, "resource_budget",
            f"wall hours {wall_hours:.3f} exceed "
            f"{config.max_wall_hours:.3f}",
        )
    generator_seconds = metric_float(
        metrics.get("generator_wall_seconds_total", 0.0),
        "generator_wall_seconds_total",
    )
    gpu_hours = (
        generator_seconds * config.tensor_parallel_size / 3600
    )
    if gpu_hours > config.max_gpu_hours:
        _issue(
            issues, "resource_budget",
            f"GPU hours {gpu_hours:.3f} exceed "
            f"{config.max_gpu_hours:.3f}",
        )
    candidates = (
        sum(1 for row in read_jsonl(root / "candidates.jsonl") if isinstance(row, dict))
        if (root / "candidates.jsonl").exists() else 0
    )
    rejected = (
        sum(
            isinstance(row, dict) and row.get("phase") == "candidate"
            for row in read_jsonl(root / "rejected.jsonl")
        )
        if (root / "rejected.jsonl").exists() else 0
    )
    attempted = candidates + rejected
    failure_rate = rejected / attempted if attempted else 1.0
    if failure_rate > config.max_failed_candidate_rate:
        _issue(
            issues, "resource_budget",
            f"candidate failure rate {failure_rate:.3f} exceeds "
            f"{config.max_failed_candidate_rate:.3f}",
        )
    minutes = wall_seconds / 60
    throughput = accepted_count / minutes if minutes > 0 else 0.0
    if throughput < config.minimum_candidates_per_minute:
        _issue(
            issues, "resource_budget",
            f"accepted throughput {throughput:.4f}/min is below "
            f"{config.minimum_candidates_per_minute:.4f}/min",
        )


def run_qa(
    output_dir: str | Path, target_count: int | None = None,
    strict_quota: bool = True,
    replay_config: SynthesisConfig | None = None,
) -> QAReport:
    root = Path(output_dir)
    episode_path = root / "episodes.jsonl"
    metadata_path = root / "metadata.jsonl"
    sft_path = root / "sft_messages.jsonl"
    issues: list[QAIssue] = []
    episodes = list(read_jsonl(episode_path)) if episode_path.exists() else []
    metadata = list(read_jsonl(metadata_path)) if metadata_path.exists() else []
    sft = list(read_jsonl(sft_path)) if sft_path.exists() else []
    if not episodes:
        _issue(
            issues, "missing_artifact",
            "episodes.jsonl is missing or empty",
        )
    for episode in episodes:
        validate_episode_static(episode, issues)
    if target_count == 200:
        validate_manual_gate_audit(
            root, episodes, issues,
            minimum_reviews=replay_config.manual_audit_minimum
            if replay_config is not None else 40,
        )
        if replay_config is None:
            _issue(
                issues, "resource_budget",
                "gate-200 QA requires resolved config",
            )
        else:
            validate_resource_budget(root, replay_config, len(episodes), issues)
    ids = [
        episode.get("id") if isinstance(episode, dict) else None
        for episode in episodes
    ]
    if len(ids) != len(set(ids)):
        _issue(issues, "duplicate", "duplicate episode ids")
    hashes = [
        (
            (episode.get("metadata") or {}).get("candidate", {}).get("code_hash")
            if isinstance(episode, dict) else None
        )
        for episode in episodes
    ]
    if len(hashes) != len(set(hashes)):
        _issue(
            issues, "duplicate",
            "duplicate normalized candidate code hashes",
        )
    if metadata and {item.get("episode_id") for item in metadata} != set(ids):
        _issue(
            issues, "schema",
            "metadata ids are not one-to-one with episodes",
        )
    if sft and {item.get("id") for item in sft} != set(ids):
        _issue(
            issues, "schema",
            "SFT rows are not one-to-one with episodes",
        )
    source_manifest_path = root / "source_manifest.json"
    if source_manifest_path.exists():
        try:
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            if source_manifest.get("dataset") != "codeparrot/apps":
                _issue(issues, "schema", "source manifest dataset is not APPS")
            if source_manifest.get("split") != "train":
                _issue(
                    issues, "split_overlap",
                    "source manifest is not restricted to APPS train",
                )
            source_path = Path(str(source_manifest.get("path", "")))
            source_name = source_path.name.lower()
            if source_name in {"test.jsonl", "test.parquet"}:
                _issue(
                    issues, "split_overlap",
                    "source manifest points at APPS test data",
                )
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            _issue(issues, "schema", f"invalid source manifest: {exc}")
    split_path = root / "splits.json"
    if split_path.exists():
        try:
            assert_split_disjoint(
                json.loads(split_path.read_text(encoding="utf-8"))
            )
        except (ValueError, json.JSONDecodeError) as exc:
            _issue(issues, "split_overlap", str(exc))
    tokenized = root / "sft_tokenized.jsonl"
    if tokenized.exists():
        validate_tokenized(tokenized, issues, {row["id"]: row for row in sft})
    if replay_config is not None and episodes:
        try:
            replay_issues = replay_artifacts(root, replay_config)
        except Exception as exc:
            replay_issues = [{
                "code": "replay_protocol",
                "message": f"offline replay failed: {type(exc).__name__}: {exc}",
                "episode_id": None,
            }]
        for replay_issue in replay_issues:
            _issue(
                issues, replay_issue["code"], replay_issue["message"],
                replay_issue.get("episode_id"),
            )


    by_origin = Counter(
        (item.get("metadata") or {}).get("candidate", {}).get("origin")
        if isinstance(item, dict) else None
        for item in episodes
    )
    by_behavior = Counter(
        (
            (item.get("metadata") or {}).get("behavior_sequence") or [None]
        )[0] if isinstance(item, dict) else None
        for item in episodes
    )
    by_difficulty = Counter(
        (item.get("metadata") or {}).get("difficulty")
        if isinstance(item, dict) else None
        for item in episodes
    )
    by_bug_count = Counter(
        str(((item.get("metadata") or {}).get("mutation") or {}).get("bug_count"))
        for item in episodes
        if isinstance(item, dict)
        and (item.get("metadata") or {}).get("candidate", {}).get("origin")
        == "synthetic_multi"
    )
    if target_count is not None and len(episodes) != target_count:
        _issue(
            issues, "quota",
            f"expected {target_count} episodes, found {len(episodes)}",
        )
    if strict_quota and target_count is not None:
        expected_cross = cross_quotas(
            target_count,
            replay_config.source_ratios if replay_config else None,
            replay_config.behavior_ratios if replay_config else None,
        )
        actual_cross = Counter()
        for item in episodes:
            if not isinstance(item, dict):
                continue
            item_meta = item.get("metadata") or {}
            item_origin = (item_meta.get("candidate") or {}).get("origin")
            item_sequence = item_meta.get("behavior_sequence") or []
            if item_origin is not None and item_sequence:
                actual_cross[(item_origin, item_sequence[0])] += 1
        for key, expected in expected_cross.items():
            if actual_cross[key] != expected:
                _issue(
                    issues, "quota",
                    f"cross quota {key}: expected {expected}, "
                    f"found {actual_cross[key]}",
                )
        for key, actual in actual_cross.items():
            if key not in expected_cross and actual:
                _issue(
                    issues, "quota",
                    f"unexpected cross-quota cell {key}: found {actual}",
                )
        if replay_config is not None:
            expected_sampling = sampling_quotas(
                target_count,
                replay_config.difficulty_ratios,
                replay_config.source_ratios,
                replay_config.behavior_ratios,
                dict(zip(
                    replay_config.multi_bug_counts,
                    replay_config.multi_bug_weights,
                )),
            )
            actual_sampling = Counter()
            for item in episodes:
                if not isinstance(item, dict):
                    continue
                item_meta = item.get("metadata") or {}
                item_origin = (item_meta.get("candidate") or {}).get("origin")
                item_sequence = item_meta.get("behavior_sequence") or []
                item_difficulty = item_meta.get("difficulty")
                if item_origin is None or not item_sequence:
                    continue
                item_behavior = item_sequence[0]
                item_bug = (
                    (item_meta.get("mutation") or {}).get("bug_count")
                    if item_origin == "synthetic_multi" else None
                )
                actual_sampling[(
                    item_origin, item_behavior, item_difficulty, item_bug,
                )] += 1
            for key, expected in expected_sampling.items():
                if actual_sampling[key] != expected:
                    _issue(
                        issues, "quota",
                        f"sampling quota {key}: expected {expected}, "
                        f"found {actual_sampling[key]}",
                    )
            for key, actual in actual_sampling.items():
                if key not in expected_sampling and actual:
                    _issue(
                        issues, "quota",
                        f"unexpected sampling-quota cell {key}: found {actual}",
                    )
    counts = {
        "episodes": len(episodes),
        "sft_rows": len(sft),
        "metadata_rows": len(metadata),
        "by_candidate_origin": dict(by_origin),
        "by_behavior": dict(by_behavior),
        "by_difficulty": dict(by_difficulty),
        "by_bug_count": dict(by_bug_count),
        "p0_issues": sum(issue.severity == "p0" for issue in issues),
        "total_issues": len(issues),
    }
    return QAReport(not issues, counts, issues)


def write_qa_report(report: QAReport, output_dir: str | Path) -> None:
    root = Path(output_dir)
    atomic_write_json(root / "qa_report.json", report.to_dict())
    fence = chr(96) * 3
    lines = [
        "# QA Report", "",
        f"Status: **{'PASS' if report.passed else 'FAIL'}**", "",
        "## Counts", "", fence + "json",
        json.dumps(report.counts, ensure_ascii=False, indent=2),
        fence, "", "## Issues", "",
    ]
    if report.issues:
        lines.extend(
            f"- {item.severity} {item.code} {item.episode_id or '-'}: "
            f"{item.message}"
            for item in report.issues
        )
    else:
        lines.append("No issues found.")
    (root / "qa_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
