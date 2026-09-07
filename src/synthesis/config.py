from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


SOURCE_ORDER = (
    "synthetic_single", "synthetic_multi", "model_natural_failure",
    "verified_reference", "model_natural_correct",
)
BEHAVIOR_ORDER = (
    "post_submit_direct_repair", "post_submit_failure_replay",
    "pre_submit_active_validation", "direct_submission",
)
DIFFICULTY_ORDER = ("introductory", "interview", "competition")


def largest_remainder(total: int, ratios: Mapping[str, float], order: tuple[str, ...] | None = None) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    keys = list(order or ratios.keys())
    if set(keys) != set(ratios):
        raise ValueError("quota order and ratio keys differ")
    values = [float(ratios[k]) for k in keys]
    if any(v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-6:
        raise ValueError("ratios must be non-negative and sum to 1")
    raw = [total * value for value in values]
    base = [int(value // 1) for value in raw]
    remainder = total - sum(base)
    ranked = sorted(range(len(keys)), key=lambda i: (-(raw[i] - base[i]), i))
    for index in ranked[:remainder]:
        base[index] += 1
    return dict(zip(keys, base))


def code_revision() -> str:
    explicit = os.environ.get("SYNTHESIS_CODE_REVISION")
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True, timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


@dataclass(slots=True)
class SynthesisConfig:
    split_seed: int = 42
    generation_seed: int = 2026
    model: str = ""
    model_revision: str | None = None
    apps_revision: str | None = None
    apps_sha256: str | None = None
    dtype: str = "bf16"
    tensor_parallel_size: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 4096
    counterfactual_samples: int = 3
    high_threshold: float = 2 / 3
    low_threshold: float = 1 / 3
    min_utility: float = 1 / 3
    candidate_sources: tuple[str, ...] = ("synthetic_single", "synthetic_multi", "model_natural")
    max_single_mutants_per_problem: int = 3
    max_multi_mutants_per_problem: int = 3
    max_natural_failures_per_problem: int = 3
    natural_generations_per_problem: int = 5
    multi_bug_counts: tuple[int, ...] = (2, 3, 4)
    multi_bug_weights: tuple[float, ...] = (0.6, 0.3, 0.1)
    mutation_families: tuple[str, ...] = ()
    max_actions_per_episode: int = 6
    max_submit_calls: int = 3
    max_run_calls: int = 3
    candidate_timeout_sec: float = 3.0
    candidate_memory_mb: int = 512
    max_output_bytes: int = 65536
    max_input_bytes: int = 65536
    max_code_tokens: int = 16384
    sandbox_backend: str = "bwrap"
    rationale_mode: str = "none"
    label_method: str = "rule_design"
    model_backend: str = "none"
    target_count: int = 1200
    strict_quota: bool = True
    source_ratios: dict[str, float] = field(default_factory=lambda: dict(zip(SOURCE_ORDER, (0.30, 0.30, 0.25, 0.10, 0.05))))
    behavior_ratios: dict[str, float] = field(default_factory=lambda: dict(zip(BEHAVIOR_ORDER, (0.25, 0.30, 0.30, 0.15))))
    max_wall_hours: float = 72.0
    max_gpu_hours: float = 288.0
    max_failed_candidate_rate: float = 0.95
    minimum_candidates_per_minute: float = 0.05
    manual_audit_minimum: int = 40
    difficulty_ratios: dict[str, float] = field(default_factory=lambda: dict(zip(DIFFICULTY_ORDER, (0.25, 0.45, 0.30))))

    def validate(self, formal: bool = False) -> None:
        largest_remainder(self.target_count, self.source_ratios, SOURCE_ORDER)
        largest_remainder(self.target_count, self.behavior_ratios, BEHAVIOR_ORDER)
        largest_remainder(self.target_count, self.difficulty_ratios, DIFFICULTY_ORDER)
        if self.counterfactual_samples < 1:
            raise ValueError("counterfactual_samples must be positive")
        if not (0 <= self.low_threshold <= self.high_threshold <= 1):
            raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")
        if self.min_utility < 0 or self.min_utility > 1:
            raise ValueError("min_utility must be in [0,1]")
        if self.rationale_mode not in {"none", "short"}:
            raise ValueError("invalid rationale mode")
        if self.label_method not in {
            "rule_design", "empirical_counterfactual",
        }:
            raise ValueError("label_method must be rule_design or empirical_counterfactual")
        if self.model_backend not in {"none", "transformers", "openrouter"}:
            raise ValueError("model_backend must be none, transformers, or openrouter")
        if self.model_backend != "none" and self.label_method == "rule_design":
            raise ValueError("rule_design requires model_backend=none")
        if len(self.multi_bug_counts) != len(self.multi_bug_weights):
            raise ValueError("multi bug counts and weights differ")
        if len(set(self.multi_bug_counts)) != len(self.multi_bug_counts):
            raise ValueError("multi bug counts must be unique")
        if any(count < 2 or count > 4 for count in self.multi_bug_counts):
            raise ValueError("multi bug counts must be 2..4")
        if any(weight < 0 for weight in self.multi_bug_weights):
            raise ValueError("multi bug weights must be non-negative")
        if abs(sum(self.multi_bug_weights) - 1.0) > 1e-6:
            raise ValueError("multi bug weights must sum to 1")
        allowed_sources = {
            "synthetic_single", "synthetic_multi", "model_natural",
            "verified_reference", "model_natural_correct",
        }
        if any(source not in allowed_sources for source in self.candidate_sources):
            raise ValueError("candidate_sources contains an unsupported source")
        if self.manual_audit_minimum < 1:
            raise ValueError("manual_audit_minimum must be positive")
        if self.max_wall_hours <= 0 or self.max_gpu_hours <= 0:
            raise ValueError("resource budgets must be positive")
        if not 0 <= self.max_failed_candidate_rate <= 1:
            raise ValueError("max_failed_candidate_rate must be in [0,1]")
        if self.minimum_candidates_per_minute < 0:
            raise ValueError("minimum_candidates_per_minute must be non-negative")
        if self.sandbox_backend not in {"bwrap", "local"}:
            raise ValueError("sandbox_backend must be bwrap or local")
        if formal and self.sandbox_backend != "bwrap":
            raise ValueError("formal runs require the bwrap sandbox backend")
        if (
            formal
            and self.model_backend != "none"
            and (not self.model_revision or self.model_revision.lower() == "latest")
        ):
            raise ValueError(
                "formal runs with a model backend require a pinned model revision"
            )
        if (
            formal
            and self.model_backend == "none"
            and self.model_revision is not None
        ):
            raise ValueError(
                "model_backend=none must not carry a model revision"
            )

    def quotas(self) -> dict[str, dict[str, int]]:
        return {
            "source": largest_remainder(self.target_count, self.source_ratios, SOURCE_ORDER),
            "behavior": largest_remainder(self.target_count, self.behavior_ratios, BEHAVIOR_ORDER),
            "difficulty": largest_remainder(self.target_count, self.difficulty_ratios, DIFFICULTY_ORDER),
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("candidate_sources", "multi_bug_counts", "multi_bug_weights", "mutation_families"):
            value[name] = list(value[name])
        return value

    def digest(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + sha256(raw).hexdigest()


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> SynthesisConfig:
    values: dict[str, Any] = {}
    if path:
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith(".json"):
            values = json.loads(text)
        else:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to read YAML config files") from exc
            values = yaml.safe_load(text) or {}
    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})
    valid = {item.name for item in fields(SynthesisConfig)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for name in ("candidate_sources", "multi_bug_counts", "multi_bug_weights", "mutation_families"):
        if name in values:
            values[name] = tuple(values[name])
    cfg = SynthesisConfig(**values)
    cfg.validate()
    return cfg


def save_resolved_config(config: SynthesisConfig, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "resolved_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **config.to_dict(),
        "config_hash": config.digest(),
        "quotas": config.quotas(),
        "code_revision": code_revision(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path

