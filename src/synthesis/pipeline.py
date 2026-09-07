from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import time
from typing import Any, Callable

from .apps_loader import (
    RejectedProblem, clean_problem, parse_apps_record_strict,
    stream_apps_jsonl,
)
from .candidates import (
    CandidateArtifact, build_multi_mutants, build_single_mutants,
)
from .checkpoint import JSONLCheckpoint
from .classify import validate_direct_submission
from .config import SynthesisConfig, code_revision, save_resolved_config
from .counterfactual import (
    CounterfactualOutcome, evaluate_post_submit, evaluate_pre_submit,
)
from .dedup import deduplicate_episodes
from .export_sft import export_messages
from .feedback_projection import project_submit_feedback
from .generation import TextGenerator
from .grader import PrivateGrader, failure_signature
from .io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl
from .natural_candidates import harvest_natural_candidates
from .normalize import content_hash, normalized_code_hash
from .prompting.state_builder import (
    append_tool_observation, build_problem_only_state,
)
from .qa import run_qa, write_qa_report
from .quota import constrained_sample
from .report import build_manifest
from .sandbox import SandboxConfig, SandboxedExecutor
from .schemas import CleanProblem
from .splitter import difficulty_stratified_split
from .tools import ToolEnvironment, tool_schemas


def make_grader(config: SynthesisConfig) -> PrivateGrader:
    sandbox = SandboxConfig(
        timeout_sec=config.candidate_timeout_sec,
        memory_mb=config.candidate_memory_mb,
        max_output_bytes=config.max_output_bytes,
        max_input_bytes=config.max_input_bytes,
        backend=config.sandbox_backend,
    )
    return PrivateGrader(SandboxedExecutor(sandbox))

def _record_stage_metrics(
    root: Path, stage: str, started: float,
    grader: PrivateGrader,
    generator: TextGenerator | None = None,
    generator_start_calls: int = 0,
    generator_start_wall_seconds: float = 0.0,
) -> None:
    path = root / "run_metrics.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists() else {"stages": {}}
    )
    previous_stage = payload["stages"].get(stage, {})
    payload["stages"][stage] = {
        "wall_seconds": float(previous_stage.get("wall_seconds", 0.0))
        + time.monotonic() - started,
        "grader_calls": int(previous_stage.get("grader_calls", 0))
        + grader.grade_calls,
        "candidate_case_runs": int(
            previous_stage.get("candidate_case_runs", 0)
        ) + grader.run_candidate_calls,
    }
    if generator is not None:
        current_calls = int(getattr(generator, "calls", 0))
        current_wall = float(getattr(generator, "wall_seconds", 0.0))
        delta_calls = max(0, current_calls - generator_start_calls)
        delta_wall = max(
            0.0, current_wall - generator_start_wall_seconds,
        )
        payload["stages"][stage]["generator_calls"] = int(
            previous_stage.get("generator_calls", 0)
        ) + delta_calls
        payload["stages"][stage]["generator_wall_seconds"] = float(
            previous_stage.get("generator_wall_seconds", 0.0)
        ) + delta_wall
        payload["generator_calls_total"] = sum(
            int(value.get("generator_calls", 0))
            for value in payload["stages"].values()
        )
        payload["generator_wall_seconds_total"] = sum(
            float(value.get("generator_wall_seconds", 0.0))
            for value in payload["stages"].values()
        )
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, payload)

def _artifact_allowed(
    path: Path, resume: bool, overwrite: bool,
) -> bool:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if not path.exists():
        return True
    if overwrite:
        return True
    if resume:
        return False
    raise FileExistsError(
        f"artifact already exists: {path}; use --resume or --overwrite"
    )


def _existing_rejects(root: Path) -> list[dict[str, Any]]:
    path = root / "rejected.jsonl"
    return list(read_jsonl(path)) if path.exists() else []


def _bind_config(root: Path, config: SynthesisConfig, overwrite: bool) -> None:
    path = root / "resolved_config.json"
    if path.exists() and not overwrite:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid resolved config: {exc}") from exc
        previous_hash = previous.get("config_hash")
        if previous_hash and previous_hash != config.digest():
            raise ValueError(
                "resolved config hash differs; use the original config for "
                "resume or pass --overwrite"
            )
    if overwrite:
        checkpoint_dir = root / "checkpoints"
        for checkpoint in (
            checkpoint_dir / "prepare.jsonl",
            checkpoint_dir / "candidates.jsonl",
            checkpoint_dir / "synthesis.jsonl",
        ):
            if checkpoint.exists():
                checkpoint.unlink()
        metrics = root / "run_metrics.json"
        if metrics.exists():
            metrics.unlink()
    if overwrite or not path.exists():
        save_resolved_config(config, root)


def prepare_apps(
    apps_path: str | Path, output_dir: str | Path,
    config: SynthesisConfig, resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _bind_config(root, config, overwrite)
    cleaned_path = root / "cleaned" / "problems.jsonl"
    if not _artifact_allowed(cleaned_path, resume, overwrite):
        return {
            "cleaned": sum(1 for _ in read_jsonl(cleaned_path)),
            "rejected": len(_existing_rejects(root)),
        }
    started = time.monotonic()
    source_bytes = Path(apps_path).read_bytes()
    source_digest = content_hash(source_bytes)
    if config.apps_sha256:
        expected = config.apps_sha256
        if not expected.startswith("sha256:"):
            expected = "sha256:" + expected
        if source_digest != expected:
            raise ValueError(
                f"APPS SHA256 mismatch: expected {expected}, "
                f"found {source_digest}"
            )

    raw_records: list[dict[str, Any]] = []
    rejects = [
        item for item in _existing_rejects(root)
        if item.get("phase") != "prepare"
    ]
    seen: set[str] = set()
    split_inputs: list[dict[str, Any]] = []
    for line_number, raw in stream_apps_jsonl(apps_path):
        if raw.get("_parse_error"):
            rejects.append({
                "phase": "prepare", "line": line_number,
                "id": None,
                "reject_reason": raw["_parse_error"],
            })
            continue
        record_split = raw.get("split")
        if (
            record_split is not None
            and str(record_split).strip().lower() not in {"", "train"}
        ):
            rejects.append({
                "phase": "prepare", "line": line_number,
                "id": raw.get("id", raw.get("problem_id")),
                "reject_reason": "apps_test_split_contamination",
            })
            continue
        problem_id = raw.get("id", raw.get("problem_id"))
        if problem_id is None or isinstance(problem_id, bool):
            rejects.append({
                "phase": "prepare", "line": line_number,
                "id": None, "reject_reason": "missing_id",
            })
            continue
        stable_id = str(problem_id)
        # Canonicalize the official APPS id spelling before splitting/cleaning.
        raw = {**raw, "id": stable_id}
        raw.pop("problem_id", None)
        if stable_id in seen:
            rejects.append({
                "phase": "prepare", "line": line_number,
                "id": stable_id,
                "reject_reason": "duplicate_id",
            })
            continue
        seen.add(stable_id)
        raw_records.append(raw)
        split_inputs.append({
            "id": stable_id,
            "difficulty": raw.get("difficulty") or "unknown",
        })

    splits = difficulty_stratified_split(
        split_inputs, seed=config.split_seed,
    )
    atomic_write_json(root / "splits.json", splits)
    pool_by_id = {
        problem_id: pool for pool, ids in splits.items()
        for problem_id in ids
    }
    grader = make_grader(config)
    checkpoint = JSONLCheckpoint(
        root / "checkpoints" / "prepare.jsonl", config.digest(),
    )
    cached_prepare = {
        record.key: record.payload for record in checkpoint.records()
    }
    cleaned: list[dict[str, Any]] = []
    for raw in raw_records:
        stable_id = str(raw["id"])
        cached = cached_prepare.get(stable_id)
        if cached is not None:
            if cached.get("status") == "accepted":
                cleaned.append(cached["row"])
            elif cached.get("status") == "rejected":
                rejects.append(cached["reject"])
            else:
                raise ValueError(
                    f"unknown prepare checkpoint status for {stable_id}"
                )
            continue
        result = clean_problem(raw, grader)
        if isinstance(result, RejectedProblem):
            reject = {
                "phase": "prepare",
                "id": result.problem_id,
                "reject_reason": result.reject_reason,
                "detail": result.detail,
            }
            rejects.append(reject)
            checkpoint.append(stable_id, {
                "status": "rejected", "reject": reject,
            })
            continue
        value = result.to_dict(include_reference_code=True)
        value["pool"] = pool_by_id[result.problem_id]
        cleaned.append(value)
        checkpoint.append(stable_id, {
            "status": "accepted", "row": value,
        })
    atomic_write_jsonl(cleaned_path, cleaned)
    atomic_write_jsonl(root / "rejected.jsonl", rejects)
    atomic_write_json(
        root / "source_manifest.json",
        {
            "dataset": "codeparrot/apps",
            "revision": config.apps_revision,
            "split": "train",
            "path": str(Path(apps_path).resolve()),
            "sha256": source_digest,
            "records": len(raw_records),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _record_stage_metrics(root, "prepare", started, grader)
    return {"cleaned": len(cleaned), "rejected": len(rejects)}


def _candidate_from_natural(
    problem: CleanProblem, natural,
) -> CandidateArtifact | None:
    if natural.code is None or natural.result is None:
        return None
    return CandidateArtifact(
        problem_id=problem.problem_id,
        code=natural.code,
        origin=natural.origin,
        code_hash=normalized_code_hash(natural.code),
        seed_submit={
            "status": natural.result.status,
            "passed": natural.result.passed,
            "total": natural.result.total,
            "pass_rate": natural.result.pass_rate,
            "failing_input": natural.result.failing_input,
        },
        failure_signature=failure_signature(natural.result),
        mutation=None,
        generator={
            "model": natural.model,
            "revision": natural.revision,
            "seed": natural.seed,
            "prompt_hash": natural.prompt_hash,
            "raw_hash": natural.raw_hash,
        },
    )


def generate_candidates(
    output_dir: str | Path, config: SynthesisConfig,
    generator: TextGenerator | None = None, resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _bind_config(root, config, overwrite)
    path = root / "candidates.jsonl"
    if not _artifact_allowed(path, resume, overwrite):
        rows = list(read_jsonl(path))
        return dict(Counter(row["origin"] for row in rows))
    started = time.monotonic()
    generator_start_calls = int(getattr(generator, "calls", 0))
    generator_start_wall = float(getattr(generator, "wall_seconds", 0.0))
    problems_path = root / "cleaned" / "problems.jsonl"
    if not problems_path.exists():
        raise FileNotFoundError("run prepare-apps first")
    grader = make_grader(config)
    rows = list(read_jsonl(problems_path))
    # Rotate by difficulty and problem id rather than exhausting one problem.
    rows.sort(key=lambda row: (
        row["public_problem"]["difficulty"], str(row["id"]),
    ))
    problems = [
        CleanProblem.from_dict(row) for row in rows
        if row.get("pool") == "sft"
    ]
    artifacts: list[CandidateArtifact] = []
    rejects = [
        item for item in _existing_rejects(root)
        if item.get("phase") != "candidate"
    ]
    checkpoint = JSONLCheckpoint(
        root / "checkpoints" / "candidates.jsonl", config.digest(),
    )
    cached_candidates = {
        record.key: record.payload for record in checkpoint.records()
    }
    for index, problem in enumerate(problems):
        cached = cached_candidates.get(problem.problem_id)
        if cached is not None:
            artifacts.extend(
                CandidateArtifact.from_dict(item)
                for item in cached.get("artifacts", [])
            )
            rejects.extend(cached.get("rejects", []))
            continue
        artifact_start = len(artifacts)
        reject_start = len(rejects)
        reference = problem.private_evaluation.selected_reference_code or ""
        private = problem.private_evaluation
        public = problem.public_problem
        reference_grade = grader.grade(
            reference, private.inputs, private.outputs,
            public.io_mode, public.fn_name,
        )
        artifacts.append(CandidateArtifact(
            problem.problem_id, reference, "verified_reference",
            normalized_code_hash(reference),
            {
                "status": reference_grade.status,
                "passed": reference_grade.passed,
                "total": reference_grade.total,
                "pass_rate": reference_grade.pass_rate,
                "failing_input": reference_grade.failing_input,
            },
            None, None,
        ))
        if "synthetic_single" in config.candidate_sources:
            artifacts.extend(build_single_mutants(
                problem, grader,
                maximum=config.max_single_mutants_per_problem,
                families=config.mutation_families or None,
            ))
        if "synthetic_multi" in config.candidate_sources:
            artifacts.extend(build_multi_mutants(
                problem, grader,
                bug_counts=config.multi_bug_counts,
                maximum=config.max_multi_mutants_per_problem,
                bug_weights=config.multi_bug_weights,
                seed=config.generation_seed + index,
                families=config.mutation_families or None,
            ))
        if "model_natural" in config.candidate_sources:
            if generator is None:
                raise ValueError(
                    "model_natural source requires a pinned model generator"
                )
            seeds = [
                config.generation_seed + index * 100 + offset
                for offset in range(config.natural_generations_per_problem)
            ]
            natural_values = harvest_natural_candidates(
                problem, generator, seeds, grader,
            )
            failure_count = 0
            correct_count = 0
            for natural in natural_values:
                if natural.reject_reason:
                    rejects.append({
                        "phase": "candidate",
                        "id": problem.problem_id,
                        "reject_reason": natural.reject_reason,
                        "origin": "model_natural",
                        "seed": natural.seed,
                        "raw_hash": natural.raw_hash,
                    })
                    continue
                artifact = _candidate_from_natural(problem, natural)
                if artifact is None:
                    continue
                if artifact.origin == "model_natural_failure":
                    failure_count += 1
                    if failure_count > config.max_natural_failures_per_problem:
                        continue
                elif artifact.origin == "model_natural_correct":
                    correct_count += 1
                    if correct_count > 2:
                        continue
                artifacts.append(artifact)
        checkpoint.append(problem.problem_id, {
            "artifacts": [
                item.to_dict() for item in artifacts[artifact_start:]
            ],
            "rejects": rejects[reject_start:],
        })
    # Candidate-stage dedup is per problem and normalized code.
    unique: list[CandidateArtifact] = []
    keys: set[tuple[str, str]] = set()
    for artifact in artifacts:
        key = (artifact.problem_id, artifact.code_hash)
        if key in keys:
            rejects.append({
                "phase": "candidate",
                "id": artifact.problem_id,
                "reject_reason": "duplicate_normalized_code",
                "origin": artifact.origin,
            })
            continue
        keys.add(key)
        unique.append(artifact)
    atomic_write_jsonl(path, (item.to_dict() for item in unique))
    atomic_write_jsonl(root / "rejected.jsonl", rejects)
    _record_stage_metrics(
        root, "generate_candidates", started, grader, generator,
        generator_start_calls, generator_start_wall,
    )
    return dict(Counter(item.origin for item in unique))


def _runtime_metadata(config: SynthesisConfig) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "code_revision": code_revision(),
        "grader_version": "coding-agent-synthesis:0.1.0",
        "sandbox_config_hash": content_hash(json.dumps({
            "timeout": config.candidate_timeout_sec,
            "memory": config.candidate_memory_mb,
            "output": config.max_output_bytes,
            "input": config.max_input_bytes,
            "sandbox_backend": config.sandbox_backend,
        }, sort_keys=True)),
    }


def _common_metadata(
    problem: CleanProblem, candidate: dict[str, Any],
    behavior: str, final_code: str, config: SynthesisConfig,
) -> dict[str, Any]:
    reference = problem.private_evaluation.selected_reference_code or ""
    return {
        "dataset": "codeparrot/apps",
        "dataset_split": "train",
        "pool": "sft",
        "difficulty": problem.public_problem.difficulty,
        "io_mode": problem.public_problem.io_mode,
        "question_hash": content_hash(problem.public_problem.question),
        "starter_code_hash": content_hash(problem.public_problem.starter_code),
        "reference": {
            "solution_index": problem.private_evaluation.selected_reference_index,
            "code_hash": normalized_code_hash(reference),
            "verified": True,
            "verification_pass_rate": 1.0,
            "replay_count": 2,
        },
        "candidate": {
            "origin": candidate["origin"],
            "code_hash": candidate["code_hash"],
            "is_buggy_context": candidate["origin"] not in {
                "verified_reference", "model_natural_correct",
            },
            "generator_model": (candidate.get("generator") or {}).get("model"),
            "generator_revision": (
                candidate.get("generator") or {}
            ).get("revision"),
            "generation_seed": (
                candidate.get("generator") or {}
            ).get("seed"),
            "generation_prompt_hash": (
                candidate.get("generator") or {}
            ).get("prompt_hash"),
        },
        "mutation": candidate.get("mutation"),
        "behavior_sequence": [behavior],
        "solution_origin": (
            "verified_reference"
            if candidate["origin"] == "verified_reference"
            else "model_natural_correct"
            if candidate["origin"] == "model_natural_correct"
            else "model_repair"
        ),
        "final_code_hash": normalized_code_hash(final_code),
        "final_pass_rate": 1.0,
        "final_replay_passed": True,
        "model": {
            "name": config.model,
            "revision": config.model_revision,
            "template_hash": None,
            "generation_config_hash": content_hash(json.dumps({
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_new_tokens": config.max_new_tokens,
            }, sort_keys=True)),
        },
        "runtime": _runtime_metadata(config),
        "quality": {
            "schema_valid": True,
            "replay_valid": True,
            "leakage_scan_passed": True,
            "loss_mask_valid": True,
            "duplicate_of": None,
            "reject_reasons": [],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _episode_dict(
    problem: CleanProblem, candidate: dict[str, Any],
    outcome: CounterfactualOutcome, config: SynthesisConfig,
) -> dict[str, Any] | None:
    behavior = outcome.behavior
    if behavior is None:
        return None
    pool = (
        outcome.direct if behavior == "post_submit_direct_repair"
        else outcome.with_run
    )
    attempt = next((item for item in pool if item.accepted), None)
    if attempt is None or attempt.action is None:
        return None
    final_code = attempt.action["arguments"]["code"]
    if candidate["origin"] in {"synthetic_single", "synthetic_multi"}:
        reference = (
            problem.private_evaluation.selected_reference_code or ""
        )
        if normalized_code_hash(final_code) != normalized_code_hash(reference):
            # A correct compensating rewrite is not proof that every recorded
            # edit was removed. Keep only statically provable complete repairs.
            return None
    metadata = _common_metadata(
        problem, candidate, behavior, final_code, config,
    )
    if candidate.get("mutation"):
        metadata["mutation_set_hash"] = content_hash(json.dumps(
            candidate["mutation"].get("edits", []),
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ))
    seed_submit = dict(candidate["seed_submit"])
    seed_submit["failure_signature"] = candidate.get("failure_signature")
    if seed_submit.get("failing_input") is not None:
        seed_submit["failing_input_hash"] = content_hash(
            seed_submit["failing_input"]
        )
        seed_submit.pop("failing_input", None)
    metadata["seed_submit"] = seed_submit
    metadata["counterfactual"] = outcome.stats.to_dict()
    if behavior == "post_submit_failure_replay":
        metadata["execution_query"] = {
            "kind": "submit_failing_case",
            "input_hash": content_hash(outcome.execution_input or ""),
            "proposal_model": None,
            "proposal_prompt_hash": None,
            "proposal_raw_hash": None,
            "observable_fields": [
                "question", "starter_code", "candidate",
                "submit_feedback",
            ],
            "matches_submit_failing_input": (
                outcome.execution_input
                == candidate["seed_submit"].get("failing_input")
            ),
        }
    elif behavior == "pre_submit_active_validation":
        proposal = outcome.proposal or {}
        metadata["execution_query"] = {
            "kind": "model_proposed",
            "input_hash": content_hash(outcome.execution_input or ""),
            "proposal_model": proposal.get("model"),
            "proposal_prompt_hash": proposal.get("prompt_hash"),
            "proposal_raw_hash": proposal.get("raw_hash"),
            "observable_fields": [
                "question", "starter_code", "io_mode",
                "input_format_note", "fn_name", "candidate",
            ],
            "matches_submit_failing_input": False,
        }
    else:
        metadata["execution_query"] = {
            "kind": "none",
            "input_hash": None,
            "proposal_model": None,
            "observable_fields": [],
            "matches_submit_failing_input": False,
        }
    calls = [
        call["name"] for message in attempt.messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]
    metadata["tool_sequence"] = calls
    metadata["tool_call_count"] = len(calls)
    metadata["decision_steps"] = [
        {
            "step": index,
            "behavior": behavior,
            "assistant_trainable": message.get("trainable", False),
            "downstream_success": True,
        }
        for index, message in enumerate(attempt.messages)
        if message.get("role") == "assistant"
        and message.get("trainable")
    ]
    stable = content_hash(
        problem.problem_id + ":" + behavior + ":" + candidate["code_hash"]
    ).split(":", 1)[1][:12]
    episode_id = f"apps-{problem.problem_id}-{behavior}-{stable}"
    metadata["schema_version"] = "1.0"
    metadata["episode_id"] = episode_id
    metadata["id"] = problem.problem_id
    return {
        "id": episode_id,
        "tools": tool_schemas(),
        "messages": attempt.messages,
        "metadata": metadata,
    }


def _direct_episode(
    problem: CleanProblem, candidate: dict[str, Any],
    config: SynthesisConfig, grader: PrivateGrader,
) -> dict[str, Any] | None:
    code = candidate["code"]
    public, private = problem.public_problem, problem.private_evaluation
    first = grader.grade(
        code, private.inputs, private.outputs, public.io_mode, public.fn_name,
    )
    second = grader.grade(
        code, private.inputs, private.outputs, public.io_mode, public.fn_name,
    )
    if not validate_direct_submission(
        first.accepted, second.accepted, has_run=False,
    ):
        return None
    state = build_problem_only_state(public)
    response = project_submit_feedback(first)
    messages = append_tool_observation(
        state, "submit", {"code": code}, response, trainable=True,
    )
    behavior = "direct_submission"
    metadata = _common_metadata(
        problem, candidate, behavior, code, config,
    )
    metadata.update({
        "counterfactual": None,
        "execution_query": None,
        "seed_submit": None,
        "behavior_sequence": [behavior],
        "tool_sequence": ["submit"],
        "tool_call_count": 1,
        "decision_steps": [{
            "step": 0, "behavior": behavior,
            "assistant_trainable": True, "downstream_success": True,
        }],
    })
    stable = content_hash(
        problem.problem_id + ":" + behavior + ":" + candidate["code_hash"]
    ).split(":", 1)[1][:12]
    episode_id = f"apps-{problem.problem_id}-{behavior}-{stable}"
    metadata.update({
        "schema_version": "1.0",
        "episode_id": episode_id,
        "id": problem.problem_id,
    })
    return {
        "id": episode_id, "tools": tool_schemas(),
        "messages": messages, "metadata": metadata,
    }


def synthesize(
    output_dir: str | Path, config: SynthesisConfig,
    generator: TextGenerator | None = None, resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _bind_config(root, config, overwrite)
    episode_path = root / "episodes.jsonl"
    if not _artifact_allowed(episode_path, resume, overwrite):
        return build_manifest(root, config.target_count)
    started = time.monotonic()
    generator_start_calls = int(getattr(generator, "calls", 0))
    generator_start_wall = float(getattr(generator, "wall_seconds", 0.0))
    problem_rows = list(read_jsonl(root / "cleaned" / "problems.jsonl"))
    problems = {
        str(row["id"]): CleanProblem.from_dict(row)
        for row in problem_rows if row.get("pool") == "sft"
    }
    candidates = list(read_jsonl(root / "candidates.jsonl"))
    grader = make_grader(config)

    def environment_factory(problem, code):
        return ToolEnvironment(problem, code, grader)

    seeds = [
        config.generation_seed + offset
        for offset in range(config.counterfactual_samples)
    ]
    valid: list[dict[str, Any]] = []
    rejects = [
        item for item in _existing_rejects(root)
        if item.get("phase") != "synthesis"
    ]
    checkpoint = JSONLCheckpoint(
        root / "checkpoints" / "synthesis.jsonl", config.digest(),
    )
    cached_synthesis = {
        record.key: record.payload for record in checkpoint.records()
    }
    for index, candidate in enumerate(candidates):
        key = f"{candidate['id']}:{candidate['code_hash']}"
        cached = cached_synthesis.get(key)
        if cached is not None:
            valid.extend(cached.get("episodes", []))
            rejects.extend(cached.get("rejects", []))
            continue
        valid_start = len(valid)
        reject_start = len(rejects)
        problem = problems.get(str(candidate["id"]))
        if problem is None:
            continue
        if candidate["origin"] in {
            "verified_reference", "model_natural_correct",
        }:
            episode = _direct_episode(
                problem, candidate, config, grader,
            )
            if episode:
                valid.append(episode)
            checkpoint.append(key, {
                "episodes": valid[valid_start:],
                "rejects": rejects[reject_start:],
            })
            continue
        if generator is None:
            raise ValueError(
                "repair synthesis requires a pinned model generator"
            )
        feedback = candidate["seed_submit"]
        thresholds = (
            config.high_threshold, config.low_threshold,
            config.min_utility,
        )
        post = evaluate_post_submit(
            problem, candidate["code"], feedback, generator, seeds,
            environment_factory, thresholds,
            (
                config.max_actions_per_episode,
                config.max_submit_calls,
                config.max_run_calls,
            ),
        )
        episode = _episode_dict(problem, candidate, post, config)
        if episode:
            valid.append(episode)
        else:
            rejects.append({
                "phase": "synthesis",
                "id": problem.problem_id,
                "candidate_hash": candidate["code_hash"],
                "reject_reason": "post_submit_ambiguous_or_failed",
            })
        pre = evaluate_pre_submit(
            problem, candidate["code"], generator,
            config.generation_seed + 10_000_000 + index,
            seeds, environment_factory, thresholds,
            (
                config.max_actions_per_episode,
                config.max_submit_calls,
                config.max_run_calls,
            ),
        )
        episode = _episode_dict(problem, candidate, pre, config)
        if episode:
            valid.append(episode)
        else:
            rejects.append({
                "phase": "synthesis",
                "id": problem.problem_id,
                "candidate_hash": candidate["code_hash"],
                "reject_reason": "pre_submit_ambiguous_or_failed",
            })
        checkpoint.append(key, {
            "episodes": valid[valid_start:],
            "rejects": rejects[reject_start:],
        })

    deduped = deduplicate_episodes(valid)
    rejects.extend({
        "phase": "synthesis", "episode_id": episode_id,
        "reject_reason": reason,
    } for episode_id, reason in deduped.rejected)
    sampled = constrained_sample(
        deduped.unique, config.target_count,
        config.difficulty_ratios,
        max_per_problem=4 if config.target_count == 200 else 6,
        source_ratios=config.source_ratios,
        behavior_ratios=config.behavior_ratios,
        bug_ratios=dict(zip(config.multi_bug_counts, config.multi_bug_weights)),
    )
    atomic_write_jsonl(episode_path, sampled.selected)
    atomic_write_jsonl(
        root / "metadata.jsonl",
        (episode["metadata"] for episode in sampled.selected),
    )
    export_messages(sampled.selected, root / "sft_messages.jsonl")
    atomic_write_jsonl(root / "rejected.jsonl", rejects)
    if sampled.shortfalls:
        shortage = {
            "target_count": config.target_count,
            "accepted_count": len(sampled.selected),
            "shortfalls": sampled.shortfalls,
            "message": (
                "Strict quota was not met; no labels were reclassified and "
                "no substitute samples were inserted."
            ),
        }
        atomic_write_json(root / "shortage_report.json", shortage)
        (root / "shortage_report.md").write_text(
            "# Quota Shortage\n\n"
            + json.dumps(shortage, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    manifest = build_manifest(
        root, config.target_count, sampled.shortfalls,
    )
    _record_stage_metrics(
        root, "synthesize", started, grader, generator,
        generator_start_calls, generator_start_wall,
    )
    return manifest
