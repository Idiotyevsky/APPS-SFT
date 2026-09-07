from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .config import (
    BEHAVIOR_ORDER, DIFFICULTY_ORDER, SOURCE_ORDER,
    SynthesisConfig, load_config, save_resolved_config,
)
from .export_sft import export_messages, export_tokenized
from .generation import TransformersTextGenerator
from .io_utils import read_jsonl
from .pipeline import (
    generate_candidates, prepare_apps, synthesize,
)
from .qa import run_qa, write_qa_report
from .report import build_manifest
from .showcase import generate_showcase


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _csv(value))


def _float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in _csv(value))


def _ratio_csv(value: str, names: tuple[str, ...]) -> dict[str, float]:
    values = _float_csv(value)
    if len(values) != len(names):
        raise argparse.ArgumentTypeError(
            f"expected {len(names)} values in order: {','.join(names)}"
        )
    return dict(zip(names, values))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--apps-revision")
    parser.add_argument("--apps-sha256")
    parser.add_argument("--dtype", choices=("bf16", "fp16"))
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--counterfactual-samples", type=int)
    parser.add_argument("--high-threshold", type=float)
    parser.add_argument("--low-threshold", type=float)
    parser.add_argument("--min-utility", type=float)
    parser.add_argument(
        "--candidate-sources",
        help="synthetic_single,synthetic_multi,model_natural",
    )
    parser.add_argument("--max-single-mutants-per-problem", type=int)
    parser.add_argument("--max-multi-mutants-per-problem", type=int)
    parser.add_argument("--max-natural-failures-per-problem", type=int)
    parser.add_argument("--natural-generations-per-problem", type=int)
    parser.add_argument("--multi-bug-counts")
    parser.add_argument("--multi-bug-weights")
    parser.add_argument("--mutation-families")
    parser.add_argument("--max-actions-per-episode", type=int)
    parser.add_argument("--max-submit-calls", type=int)
    parser.add_argument("--max-run-calls", type=int)
    parser.add_argument("--candidate-timeout-sec", type=float)
    parser.add_argument("--candidate-memory-mb", type=int)
    parser.add_argument("--max-output-bytes", type=int)
    parser.add_argument("--max-input-bytes", type=int)
    parser.add_argument("--max-code-tokens", type=int)
    parser.add_argument("--rationale-mode", choices=("none", "short"))
    parser.add_argument("--target-count", type=int)
    parser.add_argument(
        "--source-ratios",
        help="order: " + ",".join(SOURCE_ORDER),
    )
    parser.add_argument("--sandbox-backend", choices=("bwrap", "local"))
    parser.add_argument(
        "--behavior-ratios",
        help="order: " + ",".join(BEHAVIOR_ORDER),
    )
    parser.add_argument(
        "--difficulty-ratios",
        help="order: " + ",".join(DIFFICULTY_ORDER),
    )
    parser.add_argument(
        "--strict-quota", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def _config_from_args(args: argparse.Namespace) -> SynthesisConfig:
    names = {
        field.name for field in __import__(
            "dataclasses"
        ).fields(SynthesisConfig)
    }
    ignored = {
        "command", "config", "output_dir", "apps_path", "func",
        "resume", "overwrite", "fail_fast", "tokenize",
        "show_raw_apps_record", "examples_per_origin",
        "examples_per_behavior",
    }
    values: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in names and key not in ignored and value is not None:
            values[key] = value
    for name in (
        "candidate_sources", "mutation_families",
    ):
        if name in values and isinstance(values[name], str):
            values[name] = _csv(values[name])
    if "multi_bug_counts" in values and isinstance(
        values["multi_bug_counts"], str
    ):
        values["multi_bug_counts"] = _int_csv(values["multi_bug_counts"])
    if "multi_bug_weights" in values and isinstance(
        values["multi_bug_weights"], str
    ):
        values["multi_bug_weights"] = _float_csv(
            values["multi_bug_weights"]
        )
    if "source_ratios" in values and isinstance(
        values["source_ratios"], str
    ):
        values["source_ratios"] = _ratio_csv(
            values["source_ratios"], SOURCE_ORDER,
        )
    if "behavior_ratios" in values and isinstance(
        values["behavior_ratios"], str
    ):
        values["behavior_ratios"] = _ratio_csv(
            values["behavior_ratios"], BEHAVIOR_ORDER,
        )
    if "difficulty_ratios" in values and isinstance(
        values["difficulty_ratios"], str
    ):
        values["difficulty_ratios"] = _ratio_csv(
            values["difficulty_ratios"], DIFFICULTY_ORDER,
        )
    # QA/export/showcase invocations commonly point at an existing run
    # without repeating all flags. Reuse its frozen config unless an explicit
    # --config or command-line override is supplied.
    base: dict[str, Any] = {}
    if not args.config:
        resolved = Path(args.output_dir) / "resolved_config.json"
        if resolved.exists():
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            payload.pop("config_hash", None)
            payload.pop("quotas", None)
            payload.pop("code_revision", None)
            base.update(payload)
    base.update(values)
    return load_config(args.config, base)


def _generator(config: SynthesisConfig):
    config.validate(formal=True)
    return TransformersTextGenerator(
        config.model, config.model_revision or "",
        temperature=config.temperature, top_p=config.top_p,
        max_new_tokens=config.max_new_tokens, dtype=config.dtype,
    )


def command_prepare(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    summary = prepare_apps(
        args.apps_path, args.output_dir, config,
        resume=args.resume, overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_candidates(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    generator = (
        _generator(config)
        if "model_natural" in config.candidate_sources else None
    )
    summary = generate_candidates(
        args.output_dir, config, generator,
        resume=args.resume, overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_synthesize(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    generator = _generator(config)
    manifest = synthesize(
        args.output_dir, config, generator,
        resume=args.resume, overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if config.strict_quota and manifest["quota_shortfalls"]:
        return 2
    return 0


def command_export(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    root = Path(args.output_dir)
    episodes = list(read_jsonl(root / "episodes.jsonl"))
    message_path = root / "sft_messages.jsonl"
    if message_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"artifact already exists: {message_path}; use --resume or --overwrite"
        )
    count = (
        sum(1 for _ in read_jsonl(message_path))
        if message_path.exists() and args.resume and not args.overwrite
        else export_messages(episodes, message_path)
    )
    result: dict[str, Any] = {"sft_messages": count}
    if args.tokenize:
        if not config.model_revision:
            raise ValueError("--tokenize requires --model-revision")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "install the model extra to tokenize"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            config.model, revision=config.model_revision,
        )
        tokenized_path = root / "sft_tokenized.jsonl"
        if tokenized_path.exists() and args.resume and not args.overwrite:
            tokenized = sum(1 for _ in read_jsonl(tokenized_path))
        elif tokenized_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"artifact already exists: {tokenized_path}; use --resume or --overwrite"
            )
        else:
            tokenized = export_tokenized(
                episodes, tokenizer, tokenized_path,
            )
        result["sft_tokenized"] = tokenized
    print(json.dumps(result, indent=2))
    return 0


def command_qa(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    report = run_qa(
        args.output_dir, config.target_count, config.strict_quota, config,
    )
    write_qa_report(report, args.output_dir)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.passed else 1


def command_report(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    manifest = build_manifest(args.output_dir, config.target_count)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def command_showcase(args: argparse.Namespace) -> int:
    path = generate_showcase(
        args.output_dir,
        examples_per_origin=args.examples_per_origin,
        examples_per_behavior=args.examples_per_behavior,
        show_raw_apps_record=args.show_raw_apps_record,
    )
    print(path)
    return 0


def command_gate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    config.target_count = 200
    config.validate(formal=True)
    if not config.apps_revision:
        raise ValueError("gate-200 requires --apps-revision")
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()) and not (
        args.resume or args.overwrite
    ):
        raise FileExistsError(
            "gate output is non-empty; use --resume or --overwrite"
        )
    save_resolved_config(config, root)
    generator = _generator(config)
    prepare_apps(
        args.apps_path, root, config,
        resume=args.resume, overwrite=args.overwrite,
    )
    generate_candidates(
        root, config, generator,
        resume=args.resume, overwrite=args.overwrite,
    )
    manifest = synthesize(
        root, config, generator,
        resume=args.resume, overwrite=args.overwrite,
    )
    episodes = list(read_jsonl(root / "episodes.jsonl"))
    export_tokenized(
        episodes, generator.tokenizer,
        root / "sft_tokenized.jsonl",
    )
    report = run_qa(root, 200, strict_quota=True, replay_config=config)
    write_qa_report(report, root)
    build_manifest(root, 200, manifest.get("quota_shortfalls", []))
    if report.passed:
        generate_showcase(root)
    print(json.dumps({
        "manifest": manifest,
        "qa": report.to_dict(),
    }, ensure_ascii=False, indent=2))
    return 0 if report.passed and not manifest["quota_shortfalls"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m synthesis.cli",
        description=(
            "Evidence-driven APPS coding-agent data synthesis. "
            "Ratios are sampling targets and never labels."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-apps",
        help="parse, split, clean, and verify APPS train",
    )
    add_common(prepare)
    prepare.add_argument("--apps-path", required=True)
    prepare.set_defaults(func=command_prepare)

    candidates = subparsers.add_parser(
        "generate-candidates",
        help="generate single, multi, and natural candidates",
    )
    add_common(candidates)
    candidates.set_defaults(func=command_candidates)

    synthesis = subparsers.add_parser(
        "synthesize",
        help="run paired counterfactual synthesis",
    )
    add_common(synthesis)
    synthesis.set_defaults(func=command_synthesize)

    export = subparsers.add_parser(
        "export-sft",
        help="strip secrets and export SFT messages/token labels",
    )
    add_common(export)
    export.add_argument("--tokenize", action="store_true")
    export.set_defaults(func=command_export)

    qa = subparsers.add_parser(
        "qa", help="validate existing artifacts without model generation",
    )
    add_common(qa)
    qa.set_defaults(func=command_qa)

    gate = subparsers.add_parser(
        "gate-200", help="run the full strict 200-episode gate",
    )
    add_common(gate)
    gate.add_argument("--apps-path", required=True)
    gate.set_defaults(func=command_gate)

    showcase = subparsers.add_parser(
        "showcase", help="render examples from QA-passing artifacts",
    )
    add_common(showcase)
    showcase.add_argument(
        "--show-raw-apps-record", action="store_true",
        help="reserved for controlled local review; raw secrets are never emitted",
    )
    showcase.add_argument("--examples-per-origin", type=int, default=1)
    showcase.add_argument("--examples-per-behavior", type=int, default=1)
    showcase.set_defaults(func=command_showcase)

    report = subparsers.add_parser(
        "report", help="rescan artifacts and rebuild manifest",
    )
    add_common(report)
    report.set_defaults(func=command_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        if getattr(args, "fail_fast", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
