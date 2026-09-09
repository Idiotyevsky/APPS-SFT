#!/usr/bin/env python3
"""Rule-based SFT debug handoff: real execution, no model generation.

Builds a small, fully deterministic SFT sample set for teammates to debug the
training/formatting pipeline. Every candidate is a grader-verified mutation of a
twice-verified APPS reference; every tool observation comes from a real local
execution; every trajectory ends in an accepted reference submission. Behavior
labels (direct repair / failure replay / active validation / direct
submission, plus one scripted multi-round) are assigned by a deterministic rule
design and recorded with label_method=rule_design. No counterfactual claims are
made and no model/API is used.

Usage:
  build_rule_sft_handoff.py curate    # offline scan + verify -> .pool.json cache
  build_rule_sft_handoff.py assemble  # build episodes + QA + export from cache
  build_rule_sft_handoff.py all
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from synthesis.apps_loader import (  # noqa: E402
    RejectedProblem, clean_problem, parse_apps_record_strict,
    stream_apps_jsonl,
)
from synthesis.config import SynthesisConfig, save_resolved_config  # noqa: E402
from synthesis.export_sft import export_messages  # noqa: E402
from synthesis.grader import PrivateGrader, failure_signature  # noqa: E402
from synthesis.io_utils import atomic_write_json, atomic_write_jsonl  # noqa: E402
from synthesis.mutation import (  # noqa: E402
    apply_edits, enumerate_supported_single_edits,
    sample_compatible_sets,
)
from synthesis.mutation.validate import validate_multi_execution  # noqa: E402
from synthesis.normalize import normalized_code_hash  # noqa: E402
from synthesis.qa import run_qa, write_qa_report  # noqa: E402
from synthesis.report import build_manifest  # noqa: E402
from synthesis.rule_synthesizer import (  # noqa: E402
    ACTIVE_VALIDATION, DIRECT_REPAIR, DIRECT_SUBMISSION, FAILURE_REPLAY,
    build_active_validation, build_direct_submission, build_multiround,
    build_post_submit_repair,
)
from synthesis.sandbox import SandboxConfig, SandboxedExecutor  # noqa: E402
from synthesis.schemas import CleanProblem  # noqa: E402
from synthesis.showcase import generate_showcase  # noqa: E402

APPS = ROOT_DIR / "data/raw/apps/train.jsonl"
OUT = ROOT_DIR / "data/work/legacy_builds/sft_debug_rule_handoff"
CACHE = ROOT_DIR / "data/.cache/rule_handoff_pool.json"

BEHAVIOR_NEEDS = {
    DIRECT_REPAIR: 3,
    FAILURE_REPLAY: 3,
    ACTIVE_VALIDATION: 2,
}
DIRECT_NEED = 4
DIFF_NEED = {"introductory": 2, "interview": 2, "competition": 1}


def make_grader() -> PrivateGrader:
    return PrivateGrader(SandboxedExecutor(
        SandboxConfig(timeout_sec=3, memory_mb=512, backend="local"),
    ))


# --------------------------------------------------------------------------
# Curation (offline, no API)
# --------------------------------------------------------------------------

def _grade(gr, problem, code):
    public = problem.public_problem
    private = problem.private_evaluation
    return gr.grade(
        code, private.inputs, private.outputs,
        public.io_mode, public.fn_name,
    )


def _curate_one(gr, problem):
    """Return verified ref, single mutants, multi mutants (with partial code)."""
    public = problem.public_problem
    private = problem.private_evaluation
    ref = private.selected_reference_code or ""
    ref_grade = _grade(gr, problem, ref)
    if not ref_grade.accepted:
        return None
    ref_artifact = {
        "id": problem.problem_id,
        "code": ref,
        "origin": "verified_reference",
        "code_hash": normalized_code_hash(ref),
        "seed_submit": {
            "status": ref_grade.status, "passed": ref_grade.passed,
            "total": ref_grade.total, "pass_rate": ref_grade.pass_rate,
            "failing_input": None,
        },
        "failure_signature": None,
        "mutation": None,
    }
    singles = []
    seen = set()
    for edit in enumerate_supported_single_edits(ref):
        if len(singles) >= 3:
            break
        try:
            mutant, records = apply_edits(ref, [edit])
            compile(mutant, "<m>", "exec")
        except Exception:
            continue
        first = _grade(gr, problem, mutant)
        if not (0 < first.pass_rate < 1):
            continue
        second = _grade(gr, problem, mutant)
        if failure_signature(first) != failure_signature(second):
            continue
        code_hash = normalized_code_hash(mutant)
        if code_hash in seen:
            continue
        seen.add(code_hash)
        singles.append({
            "id": problem.problem_id,
            "code": mutant,
            "origin": "synthetic_single",
            "code_hash": code_hash,
            "seed_submit": {
                "status": first.status, "passed": first.passed,
                "total": first.total, "pass_rate": first.pass_rate,
                "failing_input": first.failing_input,
            },
            "failure_signature": failure_signature(first),
            "mutation": {
                "is_mutated": True, "bug_count": 1,
                "semantic_edit_count": 1,
                "all_individually_harmful": True,
                "survivor_check_passed": True,
                "masked_mutation": False,
                "edits": [record.to_dict() for record in records],
            },
        })

    multis = []
    seen_multi = set()
    ref_ast_edits = enumerate_supported_single_edits(ref)
    for bug_count in (2, 3):
        for edit_set in sample_compatible_sets(
            ref_ast_edits, bug_count, seed=0x5EED + bug_count, limit=14,
        ):
            if len(multis) >= 2:
                break
            evidence = validate_multi_execution(problem, edit_set, gr)
            if not evidence.valid or evidence.combined_result is None:
                continue
            mutant, records = apply_edits(ref, edit_set)
            code_hash = normalized_code_hash(mutant)
            if code_hash in seen_multi:
                continue
            seen_multi.add(code_hash)
            partial, _ = apply_edits(ref, edit_set[1:])
            result = evidence.combined_result
            multis.append({
                "id": problem.problem_id,
                "code": mutant,
                "origin": "synthetic_multi",
                "code_hash": code_hash,
                "partial_code": partial,
                "seed_submit": {
                    "status": result.status, "passed": result.passed,
                    "total": result.total,
                    "pass_rate": result.pass_rate,
                    "failing_input": result.failing_input,
                },
                "failure_signature": failure_signature(result),
                "mutation": {
                    "is_mutated": True,
                    "bug_count": bug_count,
                    "semantic_edit_count": bug_count,
                    "all_individually_harmful": evidence.all_individually_harmful,
                    "survivor_check_passed": evidence.survivor_check_passed,
                    "masked_mutation": evidence.masked_mutation,
                    "individual_failure_signatures": evidence.individual_signatures,
                    "edits": [record.to_dict() for record in records],
                    "all_tests_failed": result.pass_rate == 0.0,
                },
            })
    return {"ref": ref_artifact, "singles": singles, "multis": multis}


def curate() -> None:
    started = time.monotonic()
    gr = make_grader()
    entries = []
    difficulty_have = {d: 0 for d in DIFF_NEED}
    scan = 0
    last_log = 0.0
    for _line, raw in stream_apps_jsonl(APPS):
        scan += 1
        if scan > 150_000:
            break
        try:
            rec = parse_apps_record_strict(raw)
        except ValueError:
            continue
        diff = rec["difficulty"]
        if difficulty_have[diff] >= DIFF_NEED[diff] * 4:
            continue
        io = rec["input_output"]
        if not (2 <= len(io["inputs"]) <= 7):
            continue
        if not any(len(code) <= 2600 for code in rec["solutions"]):
            continue
        problem = clean_problem(raw, gr)
        if isinstance(problem, RejectedProblem):
            continue
        cur = _curate_one(gr, problem)
        if cur is None:
            continue
        difficulty_have[diff] += 1
        entries.append({
            "problem": problem.to_dict(include_reference_code=True),
            "ref": cur["ref"],
            "singles": cur["singles"],
            "multis": cur["multis"],
        })
        singles_total = sum(len(e["singles"]) for e in entries)
        multis_total = sum(len(e["multis"]) for e in entries)
        if time.monotonic() - last_log > 5:
            last_log = time.monotonic()
            print(
                f"curate: problems={len(entries)} "
                f"diff={dict(difficulty_have)} singles={singles_total} "
                f"multis={multis_total} scan={scan} "
                f"elapsed={round(time.monotonic()-started,1)}s",
                flush=True,
            )
        done = (
            all(difficulty_have[d] >= DIFF_NEED[d] for d in DIFF_NEED)
            and len(entries) >= 6
            and singles_total >= 9
            and multis_total >= 1
        )
        if done:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CACHE, {"entries": entries})
    print(
        "curate done: problems=", len(entries),
        "diff=", dict(difficulty_have),
        "singles=", sum(len(e["singles"]) for e in entries),
        "multis=", sum(len(e["multis"]) for e in entries),
        "elapsed=", round(time.monotonic() - started, 1), flush=True,
    )


def _load_cache():
    value = json.loads(CACHE.read_text(encoding="utf-8"))
    pool = []
    for entry in value["entries"]:
        problem = CleanProblem.from_dict(entry["problem"])
        pool.append({
            "problem": problem,
            "ref": entry["ref"],
            "singles": entry["singles"],
            "multis": entry["multis"],
        })
    return pool


# --------------------------------------------------------------------------
# Probe for pre-submit active validation (rule-constructed diagnostic input)
# --------------------------------------------------------------------------

def probe_input(gr, problem: CleanProblem, code: str) -> str | None:
    public = problem.public_problem
    private = problem.private_evaluation
    if public.io_mode != "stdin":
        return None
    hidden = {str(value) for value in private.inputs}
    ref = private.selected_reference_code or ""
    patterns: list[str] = []

    def add_pattern(value: str) -> None:
        if value not in hidden:
            patterns.append(value)

    for value in range(0, 9):
        add_pattern(f"{value}\n")
    for values in (
        [1, 2, 3], [5, 4, 3, 2, 1], [1, 1, 1], [1, 2, 2],
        [2, 2, 3, 3], [3, 3, 3], [7, 7, 1, 2],
        [1, 1, 2, 2, 3, 3], [4, 4, 4, 4, 1], [2, 2, 2, 1],
    ):
        add_pattern(f"{len(values)}\n{' '.join(map(str, values))}\n")
    for text in (
        "aa", "ab", "aba", "abab", "baa", "aaab", "ba", "b",
        "a", "zz", "abba", "aabb", "cc",
    ):
        for n in (2, 3, 4, 5, 7):
            add_pattern(f"{n} {len(text)}\n{text}\n")
    for words in (
        ["aa", "bb"], ["ab", "ab"], ["a", "a"], ["x", "y"],
        ["hariton", "hkariton"], ["buoi", "boooi", "bui"],
        ["k", "h"], ["ab", "aa", "ab"], ["aaaa", "aaaa"],
        ["o", "u", "k"],
    ):
        add_pattern(f"{len(words)}\n" + "\n".join(words) + "\n")
    for grid in (
        ["..", ".."], [".R", "R."], ["..R", "..."], ["RR", "RR"],
        ["..", "RR"], ["R.", ".R"], ["...", ".R.", "..."],
        [".RR", "RR.", ".R."],
    ):
        add_pattern(
            f"{len(grid)} {len(grid[0])}\n" + "\n".join(grid) + "\n",
        )
    for pattern in patterns:
        bug_run, _ = gr.run_candidate(code, pattern, public.io_mode, public.fn_name)
        if bug_run.status in {"invalid_input", "timeout", "output_limit"}:
            continue
        ref_run, _ = gr.run_candidate(ref, pattern, public.io_mode, public.fn_name)
        if ref_run.status != "ok":
            continue
        if (bug_run.stdout, bug_run.status) != (ref_run.stdout, ref_run.status):
            return pattern
    return None


# --------------------------------------------------------------------------
# Assembly + export + QA
# --------------------------------------------------------------------------

def assemble() -> int:
    started = time.monotonic()
    if not CACHE.exists():
        raise RuntimeError("run curate first (missing .pool.json)")
    pool = _load_cache()
    gr = make_grader()

    problems = [p["problem"] for p in pool]
    verified = [p["ref"] for p in pool]
    singles = [c for p in pool for c in p["singles"]]
    multis = [c for p in pool for c in p["multis"]]
    print(
        f"assemble: problems={len(problems)} refs={len(verified)} "
        f"singles={len(singles)} multis={len(multis)}", flush=True,
    )

    config = SynthesisConfig(
        split_seed=42, target_count=13,
        sandbox_backend="local", label_method="rule_design",
        model_backend="none",
    )
    episodes: list[dict] = []
    used_hashes: set[str] = set()

    def register(episode: dict) -> None:
        code_hash = episode["metadata"]["candidate"]["code_hash"]
        if code_hash in used_hashes:
            raise RuntimeError("duplicate candidate hash across episodes")
        used_hashes.add(code_hash)
        episodes.append(episode)

    def problem_of(artifact) -> CleanProblem:
        return next(p for p in problems if p.problem_id == artifact["id"])

    diff_order = ("competition", "introductory", "interview")
    ref_pick: list[dict] = []
    for diff in diff_order:
        for artifact in verified:
            if problem_of(artifact).public_problem.difficulty == diff:
                ref_pick.append(artifact)
                break
    for artifact in verified:
        if len(ref_pick) >= DIRECT_NEED:
            break
        if artifact not in ref_pick:
            ref_pick.append(artifact)
    for artifact in ref_pick[:DIRECT_NEED]:
        register(build_direct_submission(
            problem_of(artifact), artifact, gr, config,
        ))

    rng = random.Random(20260905)
    candidates = list(singles)
    rng.shuffle(candidates)
    for behavior, need in BEHAVIOR_NEEDS.items():
        taken = 0
        for candidate in candidates:
            if taken >= need:
                break
            if candidate["code_hash"] in used_hashes:
                continue
            problem = problem_of(candidate)
            ref = problem.private_evaluation.selected_reference_code or ""
            try:
                if behavior == ACTIVE_VALIDATION:
                    probe = probe_input(gr, problem, candidate["code"])
                    if probe is None:
                        continue
                    episode = build_active_validation(
                        problem, candidate, ref, probe, gr, config,
                    )
                else:
                    episode = build_post_submit_repair(
                        problem, candidate, ref, behavior, gr, config,
                        use_run=behavior == FAILURE_REPLAY,
                    )
            except ValueError as exc:
                print(
                    f"skip {behavior} {problem.problem_id}: {exc}", flush=True,
                )
                continue
            register(episode)
            taken += 1
            print(f"episode {behavior} id={problem.problem_id}", flush=True)
        if taken < need:
            print(f"WARN: behavior {behavior} short by {need - taken}", flush=True)

    for candidate in multis:
        if candidate["code_hash"] in used_hashes:
            continue
        problem = problem_of(candidate)
        ref = problem.private_evaluation.selected_reference_code or ""
        try:
            episode = build_multiround(
                problem, candidate, ref,
                candidate["partial_code"], gr, config,
            )
        except ValueError as exc:
            print(
                f"skip multiround {problem.problem_id}: {exc}", flush=True,
            )
            continue
        register(episode)
        print(f"episode multiround id={problem.problem_id}", flush=True)
        break

    if not episodes:
        raise RuntimeError("no episodes produced")
    episodes.sort(key=lambda item: item["id"])
    print("episodes:", len(episodes), flush=True)

    # ---- persist artifacts -------------------------------------------------
    cleaned_rows = []
    for problem in problems:
        row = problem.to_dict(include_reference_code=True)
        row["pool"] = "sft"
        cleaned_rows.append(row)
    atomic_write_jsonl(OUT / "cleaned" / "problems.jsonl", cleaned_rows)
    atomic_write_json(OUT / "splits.json", {
        "sft": [p.problem_id for p in problems],
        "rl": [], "behavior_dev": [],
    })
    source_digest = "sha256:" + hashlib.sha256(APPS.read_bytes()).hexdigest()
    atomic_write_json(OUT / "source_manifest.json", {
        "dataset": "codeparrot/apps",
        "revision": None,
        "split": "train",
        "path": str(APPS.resolve()),
        "sha256": source_digest,
        "records": len(problems),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    candidate_rows = [
        {key: value for key, value in artifact.items() if key != "partial_code"}
        for artifact in (verified + singles + multis)
    ]
    atomic_write_jsonl(OUT / "candidates.jsonl", candidate_rows)
    atomic_write_jsonl(OUT / "episodes.jsonl", episodes)
    atomic_write_jsonl(OUT / "metadata.jsonl", (
        episode["metadata"] for episode in episodes
    ))
    export_messages(episodes, OUT / "sft_messages.jsonl")
    save_resolved_config(config, OUT)

    # ---- offline QA ---------------------------------------------------------
    report = run_qa(
        OUT, target_count=None, strict_quota=False, replay_config=config,
    )
    write_qa_report(report, OUT)
    print(
        "qa passed:", report.passed, "issues:", len(report.issues),
        flush=True,
    )
    for issue in report.issues:
        print("  issue:", issue.to_dict(), flush=True)

    manifest = build_manifest(OUT, len(episodes))
    manifest.update({
        "usage": "sft_integration_fixture",
        "label_method": "rule_design",
        "model_backend": "none",
        "formal_accepted": False,
        "counterfactual_verified": False,
        "natural_candidates": False,
        "behavior_by_rule": True,
        "sources": ["synthetic_single", "synthetic_multi", "verified_reference"],
        "qa_passed": report.passed,
        "wall_seconds": round(time.monotonic() - started, 1),
    })
    atomic_write_json(OUT / "dataset_manifest.json", manifest)
    write_readme(len(episodes), report.passed)
    if report.passed:
        generate_showcase(OUT, examples_per_origin=1, examples_per_behavior=1)
        try_tokenize()
    print("done ->", OUT, flush=True)
    return 0 if report.passed else 2


def write_readme(count: int, qa_passed: bool) -> None:
    (OUT / "README.md").write_text(
        "\n".join([
            "# SFT debug handoff (rule-based fixtures)",
            "",
            f"- episodes: {count}; generated entirely offline with real",
            "  APPS records and the project grader (local sandbox). No model",
            "  or API was used and no counterfactual measurement was",
            "  performed.",
            "- Every tool observation comes from a real execution of the",
            "  current candidate; every trajectory ends in a twice-verified",
            "  reference accepted by the grader.",
            "- Behavior labels are a deterministic rule/curriculum design",
            "  (`label_method=rule_design`). They are NOT empirical",
            "  counterfactual conclusions.",
            f"- offline QA passed: {qa_passed}",
            "- `sft_messages.jsonl` is the training-format view;",
            "  `episodes.jsonl` + `metadata.jsonl` carry the evidence records.",
            "- For teammates to debug the SFT reader / chat template /",
            "  loss-mask path, not for formal training.",
        ]) + "\n",
        encoding="utf-8",
    )


def try_tokenize() -> None:
    """Best-effort token-level loss labels with a real chat-tokenizer."""
    try:
        from transformers import AutoTokenizer
        from synthesis.export_sft import export_tokenized
        from synthesis.io_utils import read_jsonl

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct", revision="main",
        )
        episodes = list(read_jsonl(OUT / "episodes.jsonl"))
        export_tokenized(episodes, tokenizer, OUT / "sft_tokenized.jsonl")
        print("tokenized export written", flush=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"tokenize skipped: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in {"curate", "all"}:
        curate()
    if stage in {"assemble", "all"}:
        raise SystemExit(assemble())
