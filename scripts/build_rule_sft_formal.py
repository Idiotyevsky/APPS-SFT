#!/usr/bin/env python3
"""Rule-based formal SFT set, supply-driven (offline, deterministic).

Every row:
  - id      = the raw APPS problem id (no prefix, no synthesis type)
  - one row per problem (never reuse a problem)
  - tool observations come from real local executions of grader/sandbox
  - final submit is the twice-verified reference and is replayed by QA
  - label_method = rule_design (no counterfactual claims, no model/API)

Coverage: all four behaviors (direct_submission / post_submit_direct_repair /
post_submit_failure_replay / pre_submit_active_validation), all three sources
(synthetic_single / synthetic_multi / verified_reference) and all three APPS
difficulties, drawn from the SFT pool. Exact row count is set by --target
(default: all usable problems), NOT forced to a fixed quota; per-behavior soft
caps only bound how much of the trivial direct-submission kind is emitted.

Stages:
  curate    parallel, resumable scan+verify of the sft pool -> .sft_pool.jsonl
  assemble  build rows (progress bar) + offline QA + exports
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from synthesis.export_sft import export_messages  # noqa: E402
from synthesis.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl  # noqa: E402
from synthesis.qa import run_qa, write_qa_report  # noqa: E402
from synthesis.report import build_manifest  # noqa: E402
from synthesis.rule_synthesizer import (  # noqa: E402
    ACTIVE_VALIDATION, DIRECT_REPAIR, DIRECT_SUBMISSION, FAILURE_REPLAY,
    build_active_validation, build_direct_submission, build_multiround,
    build_post_submit_repair,
)

APPS = ROOT_DIR / "data/raw/apps/train.jsonl"
RL_SPLITS = ROOT_DIR / "data/rl/splits.json"
OUT = ROOT_DIR / "data/sft_v4_final"
POOL_CACHE = ROOT_DIR / "data/.cache/rule_sft_pool.jsonl"

DIFF_ORDER = ("introductory", "interview", "competition")

# soft caps: direct-submission rows are unlimited (every usable problem can
# yield one); active probing is OFF for the formal set (no external-candidate
# mechanism in the problem-only evaluator). Multi-round repair cap: convert up
# to MULTIROUND_TARGET multi-bug candidates into true iterative episodes.
CAP_DIRECT = 1000000
CAP_ACTIVE = 0
MULTIROUND_TARGET = 250


# ---------------------------------------------------------------------------
# Curation (parallel, resumable)
# ---------------------------------------------------------------------------

def _worker(raw: dict):
    import json as _json

    from synthesis.rule_common import clean, curate_problem, make_grader

    try:
        raw = _json.loads(_json.dumps(raw)) if not isinstance(raw, dict) else raw
    except Exception:
        pass
    gr = make_grader()
    try:
        problem = clean(gr, raw)
    except Exception as exc:
        return {"id": None, "ok": False, "reason": f"clean:{type(exc).__name__}"}
    if problem is None:
        return {"id": raw.get("id"), "ok": False, "reason": "rejected"}
    entry = curate_problem(gr, problem)
    if entry is None:
        return {"id": problem.problem_id, "ok": False, "reason": "no_verified_ref"}
    row = problem.to_dict(include_reference_code=True)
    row["pool"] = "sft"
    return {
        "id": problem.problem_id,
        "difficulty": problem.public_problem.difficulty,
        "ok": True,
        "problem": row,
        "ref": entry["ref"],
        "singles": entry["singles"],
        "multis": entry["multis"],
    }


def _raw_filter(raw: dict) -> bool:
    if raw.get("difficulty") not in DIFF_ORDER:
        return False
    try:
        io = json.loads(raw.get("input_output", ""))
    except Exception:
        return False
    if not isinstance(io, dict) or not isinstance(io.get("inputs"), list):
        return False
    if not (1 <= len(io["inputs"]) <= 60):
        return False
    try:
        solutions = json.loads(raw.get("solutions", ""))
    except Exception:
        return False
    if not isinstance(solutions, list) or not solutions:
        return False
    if not any(isinstance(code, str) and 10 <= len(code) <= 20000 for code in solutions):
        return False
    return True


def curate(workers: int, caps: dict[str, int], pool: str = "all") -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, int] = {d: 0 for d in DIFF_ORDER}
    existing: set[str] = set()
    if POOL_CACHE.exists():
        for row in read_jsonl(POOL_CACHE):
            if row.get("id") is not None:
                existing.add(str(row["id"]))
                if row.get("ok"):
                    done[row["difficulty"]] += 1

    allowed_ids: set[str] | None = None
    if pool != "all":
        splits = json.loads(RL_SPLITS.read_text(encoding="utf-8"))
        allowed_ids = {str(value) for value in splits[pool]}
    candidates: list[dict] = []
    for raw in open(APPS, encoding="utf-8"):
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        identifier = rec.get("id", rec.get("problem_id"))
        if identifier is None:
            continue
        sid = str(identifier)
        if allowed_ids is not None and sid not in allowed_ids:
            continue
        if sid in existing:
            continue
        if not _raw_filter(rec):
            continue
        difficulty = rec.get("difficulty")
        if done[difficulty] >= caps[difficulty]:
            continue
        candidates.append(rec)
        done[difficulty] += 1

    print(
        f"curate: existing={len(existing)} new_tasks={len(candidates)} "
        f"caps={caps} workers={workers}", flush=True,
    )
    accepted = 0
    from tqdm import tqdm

    def cache_row(result: dict) -> None:
        with POOL_CACHE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, rec) for rec in candidates]
        with tqdm(total=len(futures), desc="curate", unit="task",
                  leave=True, dynamic_ncols=True) as pbar:
            for future in futures:
                result = future.result()
                if result.get("ok"):
                    cache_row(result)
                    accepted += 1
                elif result.get("id") is not None:
                    cache_row({
                        "id": str(result["id"]), "ok": False,
                        "reason": result.get("reason"),
                    })
                pbar.update(1)
    print(
        f"curate done: +{accepted} accepted this run, "
        f"elapsed={round(time.monotonic()-started,1)}s", flush=True,
    )


# ---------------------------------------------------------------------------
# Assembly (supply-driven)
# ---------------------------------------------------------------------------

def _observation_stable(gr, problem, code: str, input_text: str) -> bool:
    if not isinstance(input_text, str):
        return False
    public = problem.public_problem
    one, _ = gr.run_candidate(code, input_text, public.io_mode, public.fn_name)
    two, _ = gr.run_candidate(code, input_text, public.io_mode, public.fn_name)
    return one.public_dict() == two.public_dict()


def load_pool():
    return [
        row for row in read_jsonl(POOL_CACHE) if row.get("ok")
    ]


def main(target: int) -> int:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    entries = load_pool()
    from synthesis.config import SynthesisConfig, save_resolved_config
    from synthesis.grader import PrivateGrader
    from synthesis.rule_common import probe_input
    from synthesis.sandbox import SandboxConfig, SandboxedExecutor
    from synthesis.schemas import CleanProblem
    from synthesis.showcase import generate_showcase
    from tqdm import tqdm

    gr = PrivateGrader(SandboxedExecutor(
        SandboxConfig(timeout_sec=3, memory_mb=512, backend="local"),
    ))
    problems = {row["id"]: CleanProblem.from_dict(row["problem"]) for row in entries}
    print(
        f"assemble: usable problems={len(entries)} target={target} "
        f"singles={sum(len(e['singles']) for e in entries)} "
        f"multis={sum(len(e['multis']) for e in entries)}", flush=True,
    )

    config = SynthesisConfig(
        split_seed=42, target_count=target,
        sandbox_backend="local", label_method="rule_design",
        model_backend="none",
    )

    def build_replay_or_dr(behavior, problem, candidate, ref):
        ep = build_post_submit_repair(
            problem, candidate, ref, behavior, gr, config,
            use_run=behavior == FAILURE_REPLAY,
        )
        if behavior == FAILURE_REPLAY and not _observation_stable(
            gr, problem, candidate["code"], candidate["seed_submit"].get("failing_input"),
        ):
            return None
        return ep

    rng = random.Random(20260905)
    order = list(entries)
    rng.shuffle(order)

    partial = OUT / ".episodes_partial.jsonl"
    episodes: list[dict] = []
    count = {"direct_submission": 0, DIRECT_REPAIR: 0,
             FAILURE_REPLAY: 0, ACTIVE_VALIDATION: 0}
    multi_count = 0
    used_ids: set[str] = set()
    if partial.exists():
        for ep in read_jsonl(partial):
            episodes.append(ep)
            used_ids.add(str(ep["id"]))
            count[ep["metadata"]["behavior_sequence"][0]] += 1
            if (
                ep["metadata"]["behavior_sequence"][0] == FAILURE_REPLAY
                and ep["metadata"].get("tool_sequence", []).count(
                    "run_candidate") >= 2
            ):
                multi_count += 1
        print(f"resume: {len(episodes)} rows already built "
              f"(multiround={multi_count})", flush=True)

    def persist(episode: dict) -> None:
        with partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    total_work = min(target, len(order))
    pbar = tqdm(total=total_work, initial=len(episodes), desc="rows", unit="row",
                leave=True, dynamic_ncols=True)
    for row in order:
        if len(episodes) >= target:
            break
        problem_id = str(row["id"])
        if problem_id in used_ids:
            continue
        problem = problems[problem_id]
        ref = problem.private_evaluation.selected_reference_code or ""

        def attempt(behavior):
            candidates = (
                row["singles"] + row["multis"]
                if behavior != ACTIVE_VALIDATION else row["singles"] + row["multis"]
            )
            for candidate in candidates:
                if behavior == ACTIVE_VALIDATION:
                    probe = probe_input(gr, problem, candidate["code"])
                    if probe is None:
                        continue
                    if not _observation_stable(gr, problem, candidate["code"], probe):
                        continue
                    try:
                        ep = build_active_validation(
                            problem, candidate, ref, probe, gr, config,
                        )
                    except ValueError:
                        continue
                    return ep
                try:
                    ep = build_replay_or_dr(behavior, problem, candidate, ref)
                except ValueError:
                    continue
                if ep is not None:
                    return ep
            return None

        def attempt_multiround():
            for candidate in row.get("multis", []):
                if not candidate.get("partial_code"):
                    continue
                if (candidate.get("mutation") or {}).get("bug_count", 0) < 2:
                    continue
                try:
                    ep = build_multiround(
                        problem, candidate, ref, candidate["partial_code"],
                        gr, config,
                    )
                except ValueError:
                    continue
                if ep is not None:
                    return ep
            return None

        chosen = None
        # multi-round repair first (P0), then replay/direct-repair; active is OFF.
        if multi_count < MULTIROUND_TARGET and row.get("multis"):
            chosen = attempt_multiround()
            if chosen is not None:
                multi_count += 1
                count[FAILURE_REPLAY] += 1
        if chosen is None and count[ACTIVE_VALIDATION] < CAP_ACTIVE:
            chosen = attempt(ACTIVE_VALIDATION)
            if chosen is not None:
                count[ACTIVE_VALIDATION] += 1
        if chosen is None:
            chosen = attempt(FAILURE_REPLAY)
            if chosen is not None:
                count[FAILURE_REPLAY] += 1
        if chosen is None:
            chosen = attempt(DIRECT_REPAIR)
            if chosen is not None:
                count[DIRECT_REPAIR] += 1
        if chosen is None and count[DIRECT_SUBMISSION] < CAP_DIRECT:
            try:
                chosen = build_direct_submission(problem, row["ref"], gr, config)
                if chosen is not None:
                    count[DIRECT_SUBMISSION] += 1
            except ValueError:
                chosen = None
        if chosen is None:
            continue
        episodes.append(chosen)
        used_ids.add(problem_id)
        persist(chosen)
        pbar.set_postfix(bh=chosen["metadata"]["behavior_sequence"][0])
        pbar.update(1)
    pbar.close()
    if partial.exists():
        partial.unlink()

    episodes.sort(key=lambda item: str(item["id"]))
    print(
        "built rows:", len(episodes), "behavior counts:", dict(count),
        "multiround:", multi_count, flush=True,
    )
    if not episodes:
        raise RuntimeError("no episodes produced")

    by_source = {}
    for e in episodes:
        o = e["metadata"]["candidate"]["origin"]
        by_source[o] = by_source.get(o, 0) + 1
    by_behavior = {}
    for e in episodes:
        b = e["metadata"]["behavior_sequence"][0]
        by_behavior[b] = by_behavior.get(b, 0) + 1
    by_diff = {}
    for e in episodes:
        d = e["metadata"]["difficulty"]
        by_diff[d] = by_diff.get(d, 0) + 1
    ids = [str(e["id"]) for e in episodes]
    assert len(ids) == len(set(ids)), "duplicate problem ids"
    print("by_source", by_source, "by_behavior", by_behavior, "by_diff", by_diff,
          flush=True)

    atomic_write_jsonl(OUT / "episodes.jsonl", episodes)
    atomic_write_jsonl(OUT / "metadata.jsonl", (e["metadata"] for e in episodes))
    export_messages(episodes, OUT / "sft_messages.jsonl")

    cleaned = []
    for row in entries:
        value = dict(row["problem"])
        value["pool"] = "sft"
        cleaned.append(value)
    atomic_write_jsonl(OUT / "cleaned" / "problems.jsonl", cleaned)
    atomic_write_json(OUT / "splits.json", json.loads(RL_SPLITS.read_text(encoding="utf-8")))
    source_digest = "sha256:" + hashlib.sha256(APPS.read_bytes()).hexdigest()
    atomic_write_json(OUT / "source_manifest.json", {
        "dataset": "codeparrot/apps", "split": "train",
        "path": str(APPS.resolve()), "sha256": source_digest,
        "records": len(entries),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    save_resolved_config(config, OUT)

    qa = run_qa(OUT, target_count=None, strict_quota=False, replay_config=config)
    unstable_codes = {"observation_replay", "seed_replay", "final_replay"}
    for _drop_round in range(3):
        bad = {i.episode_id for i in qa.issues
               if i.code in unstable_codes}
        if not bad:
            break
        print(f"dropping {len(bad)} replay-unstable rows: {sorted(bad)}",
              flush=True)
        episodes = [e for e in episodes if str(e["id"]) not in bad]
        atomic_write_jsonl(OUT / "episodes.jsonl", episodes)
        atomic_write_jsonl(OUT / "metadata.jsonl",
                           (e["metadata"] for e in episodes))
        export_messages(episodes, OUT / "sft_messages.jsonl")
        qa = run_qa(OUT, target_count=None, strict_quota=False,
                    replay_config=config)
    write_qa_report(qa, OUT)
    print("qa passed:", qa.passed, "issues:", len(qa.issues),
          "rows:", len(episodes), flush=True)
    for issue in qa.issues[:20]:
        print("  issue:", issue.to_dict(), flush=True)

    manifest = build_manifest(OUT, len(episodes))
    manifest.update({
        "usage": "rule_formal_supply_driven",
        "target_requested": target,
        "rows": len(episodes),
        "label_method": "rule_design",
        "model_backend": "none",
        "counterfactual_verified": False,
        "natural_candidates": False,
        "one_row_per_problem": True,
        "by_candidate_origin": by_source,
        "by_behavior": by_behavior,
        "by_difficulty": by_diff,
        "qa_passed": qa.passed,
        "wall_seconds": round(time.monotonic() - started, 1),
    })
    atomic_write_json(OUT / "dataset_manifest.json", manifest)

    passed = qa.passed and len(episodes) >= 1
    if qa.passed:
        try_tokenize()
        generate_showcase(OUT, examples_per_origin=1, examples_per_behavior=1)
    print("done ->", OUT, "rows:", len(episodes), "qa_passed:", qa.passed,
          flush=True)
    return 0 if passed else 2


def try_tokenize() -> None:
    try:
        from transformers import AutoTokenizer
        from synthesis.export_sft import export_tokenized

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct", revision="main",
        )
        episodes = list(read_jsonl(OUT / "episodes.jsonl"))
        export_tokenized(episodes, tokenizer, OUT / "sft_tokenized.jsonl")
        print("tokenized export written", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"tokenize skipped: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("curate", "assemble", "all"))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--caps", default="150,220,140",
        help="intro,interview,competition curation caps",
    )
    parser.add_argument("--target", type=int, default=1000000,
                        help="max rows (default: use every usable problem)")
    parser.add_argument(
        "--pool", choices=("all", "sft", "sft_dev"), default="all",
        help="which train problems to curate from (default: all 5000)",
    )
    args = parser.parse_args()
    if args.stage in ("curate", "all"):
        caps = dict(zip(DIFF_ORDER, (int(v) for v in args.caps.split(","))))
        curate(args.workers, caps, pool=args.pool)
    if args.stage in ("assemble", "all"):
        raise SystemExit(main(args.target))
