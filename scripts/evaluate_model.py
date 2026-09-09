#!/usr/bin/env python3
"""Evaluate a coding model on the ToolAPPS eval set (single-shot, greedy).

Protocol (adapted from RLEF-Code's evaluate.py conventions):
  * strict whole-question success: the generated program must pass EVERY test
    case of the problem through the local private grader (accepted);
  * greedy decoding (temperature 0) for a deterministic pass@1;
  * fixed per-difficulty denominators from data/eval/toolapps_eval.jsonl;
  * per-problem granular results are always saved (enables paired analyses),
    summaries are printed and stored as JSON.

Usage:
  PYTHONPATH=src python3 scripts/evaluate_model.py \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --revision <PIN> \
      --eval-path data/eval/toolapps_eval.jsonl \
      --output-dir data/eval/results \
      --limit 8          # quick smoke
Refuses to silently fall back: a missing/unloadable model is an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
warnings.simplefilter("ignore")

from synthesis.grader import PrivateGrader  # noqa: E402
from synthesis.generation import parse_complete_code  # noqa: E402
from synthesis.io_utils import atomic_write_json  # noqa: E402
from synthesis.sandbox import SandboxConfig, SandboxedExecutor  # noqa: E402

CODE_TAG = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)


def build_prompt(row: dict) -> list[dict]:
    starter = row.get("starter_code") or ""
    starter_text = starter if starter else "(none)"
    fn = row.get("fn_name")
    fn_line = f"\nFunction name: {fn}" if fn else ""
    io_line = (
        "Your program reads from standard input and writes to standard output."
        if row["io_mode"] == "stdin"
        else f"Implement the function and return the value (no main/printing)."
    )
    user = (
        f"Write a complete Python solution for the problem below.\n"
        f"Return ONLY one Python code block (```python ... ```).\n\n"
        f"Problem:\n{row['question']}\n\n"
        f"Starter code:\n{starter_text}\n"
        f"I/O mode: {row['io_mode']}{fn_line}\n{io_line}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert competitive programmer. Write correct, "
                "efficient Python code. Output exactly one Python code block."
            ),
        },
        {"role": "user", "content": user},
    ]


def extract_code(raw: str) -> str | None:
    tag = CODE_TAG.search(raw)
    if tag:
        return parse_complete_code(tag.group(1)) or parse_complete_code(raw)
    return parse_complete_code(raw)


def _grade_problem(payload: dict) -> dict:
    """Worker: grade one generated solution with the private grader."""
    row = payload["row"]
    raw = payload["raw"]
    save_codes = payload["save_codes"]
    from synthesis.grader import PrivateGrader
    from synthesis.sandbox import SandboxConfig, SandboxedExecutor

    grader = PrivateGrader(SandboxedExecutor(
        SandboxConfig(timeout_sec=3, memory_mb=512, backend="local"),
    ))
    record = {
        "problem_id": row["problem_id"],
        "difficulty": row["difficulty"],
    }
    code = extract_code(raw)
    record["parse_ok"] = code is not None
    if code is None:
        record.update({"solved": False, "pass_rate": 0.0,
                       "status": "unparseable", "code": None})
        return record
    grade = grader.grade(
        code, row["inputs"], row["outputs"],
        row["io_mode"], row.get("fn_name"),
    )
    record.update({
        "solved": grade.accepted,
        "pass_rate": grade.pass_rate,
        "status": grade.status,
        "code": code if save_codes else None,
        "raw": raw if save_codes else None,
    })
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--revision", default=None,
                        help="pinned HF revision (recommended for reproducibility)")
    parser.add_argument("--eval-path", default=ROOT_DIR / "data/eval/toolapps_eval.jsonl")
    parser.add_argument("--output-dir", default=ROOT_DIR / "data/eval/results")
    parser.add_argument("--limit", type=int, default=0, help="0 = full eval set")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "auto"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-completions", action="store_true",
                        help="include raw model output per problem in the report")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted run (per-problem progress file)")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf",
                        help="generation engine (vllm is much faster; needs vllm)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--grade-workers", type=int, default=32,
                        help="processes for parallel sandbox grading (vllm backend)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.eval_path, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"eval problems: {len(rows)}  model: {args.model} revision: {args.revision or 'main'}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "__")
    progress_path = out_dir / f"{slug}@{(args.revision or 'main')}_progress.jsonl"
    done_ids: set[str] = set()
    if args.resume and progress_path.exists():
        for line in progress_path.open(encoding="utf-8"):
            try:
                done_ids.add(str(json.loads(line)["problem_id"]))
            except Exception:
                pass
        print(f"resume: {len(done_ids)} problems already evaluated", flush=True)

    from transformers import AutoTokenizer

    revision = args.revision or "main"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer for {args.model}@{revision}; refusing to fall back. {exc}"
        ) from exc

    sample = args.temperature > 0
    grader = PrivateGrader(SandboxedExecutor(
        SandboxConfig(timeout_sec=3, memory_mb=512, backend="local"),
    ))

    results: list[dict] = []
    started = time.monotonic()
    from tqdm import tqdm
    pbar = tqdm(total=len(rows), desc="eval", unit="problem")

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams  # type: ignore

        llm = LLM(
            model=args.model,
            dtype="bfloat16" if args.dtype in ("bf16", "auto") else "float16",
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        sampling = SamplingParams(
            temperature=args.temperature, top_p=1.0,
            max_tokens=args.max_new_tokens,
        )
        # template prompts with the tokenizer only (no model weights needed)
        templated = [
            tokenizer.apply_chat_template(
                build_prompt(row), tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in rows
        ]
        print("vllm: generating outputs for all problems (batched)...", flush=True)
        outputs = llm.generate(templated, sampling, use_tqdm=True)
        generations = {
            str(row["problem_id"]): out.outputs[0].text
            for row, out in zip(rows, outputs)
        }
        gen_cache = out_dir / f"{slug}@{revision}_generations.jsonl"
        with gen_cache.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({
                    "problem_id": row["problem_id"],
                    "raw": generations[str(row["problem_id"])],
                }, ensure_ascii=False) + "\n")
        print("vllm: generation cached, grading in parallel...", flush=True)

        from concurrent.futures import ProcessPoolExecutor, as_completed

        pending = [row for row in rows if str(row["problem_id"]) not in done_ids]
        tasks = [{
            "row": row,
            "raw": generations[str(row["problem_id"])],
            "save_codes": args.save_completions,
        } for row in pending]
        with ProcessPoolExecutor(max_workers=args.grade_workers) as pool:
            futures = {pool.submit(_grade_problem, task): task["row"] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="grade", unit="problem"):
                record = future.result()
                results.append(record)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                pbar.update(1)
        pbar.close()
        results = [json.loads(line) for line in progress_path.open(encoding="utf-8")]
        payload = _build_payload(args, revision, sample, results, started)
        out_path = out_dir / f"{slug}@{revision}_eval.json"
        atomic_write_json(out_path, payload)
        print("\n=== EVALUATION RESULTS (pass@1, strict whole-question) ===")
        _print_summary(payload)
        print(f"saved: {out_path}  wall {payload['wall_seconds']:.0f}s")
        return 0

    # ── HF backend: load the model only here (vllm branch already returned) ──
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = None
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=revision, torch_dtype=torch_dtype,
            device_map="auto",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model {args.model}@{revision}; refusing to fall back. {exc}"
        ) from exc
    model.eval()
    device = model.device

    for row in rows:
        if str(row["problem_id"]) in done_ids:
            pbar.update(1)
            continue
        record = {
            "problem_id": row["problem_id"],
            "difficulty": row["difficulty"],
        }
        messages = build_prompt(row)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        encoded = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                do_sample=sample,
                temperature=max(args.temperature, 1e-6),
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            output[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        code = extract_code(raw)
        record["parse_ok"] = code is not None
        if code is None:
            record.update({"solved": False, "pass_rate": 0.0,
                           "status": "unparseable", "code": None})
            results.append(record)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.update(1)
            continue
        grade = grader.grade(
            code, row["inputs"], row["outputs"],
            row["io_mode"], row.get("fn_name"),
        )
        record.update({
            "solved": grade.accepted,
            "pass_rate": grade.pass_rate,
            "status": grade.status,
            "code": code if args.save_completions else None,
            "raw": raw if args.save_completions else None,
        })
        results.append(record)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        pbar.update(1)
    pbar.close()

    payload = _build_payload(args, revision, sample, results, started)
    out_path = out_dir / f"{slug}@{revision}_eval.json"
    atomic_write_json(out_path, payload)
    print("\n=== EVALUATION RESULTS (pass@1, strict whole-question) ===")
    _print_summary(payload)
    print(f"saved: {out_path}  wall {payload['wall_seconds']:.0f}s")
    return 0


def _build_payload(args, revision, sample, results, started) -> dict:
    if results:
        by_diff = {d: [r for r in results if r["difficulty"] == d]
                   for d in ["introductory", "interview", "competition"]}
        summary = {"total": len(results), "solved": sum(r["solved"] for r in results)}
        for difficulty, items in by_diff.items():
            summary[difficulty] = {
                "total": len(items),
                "solved": sum(r["solved"] for r in items),
                "pass_at_1": (sum(r["solved"] for r in items) / len(items))
                if items else None,
            }
        summary["pass_at_1_overall"] = (
            summary["solved"] / summary["total"] if summary["total"] else None
        )
    else:
        summary = {"total": 0, "solved": 0}
    return {
        "run": {
            "model": args.model,
            "revision": revision,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "eval_path": str(args.eval_path),
            "seed_fixed": not sample,
        },
        "summary": summary,
        "by_problem": results,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _print_summary(payload: dict) -> None:
    summary = payload["summary"]
    for key, value in summary.items():
        if isinstance(value, dict):
            rate = value["pass_at_1"]
            if rate is None:
                print(f"{key:14s} 0/0")
            else:
                print(
                    f"{key:14s} {value['solved']:4d}/{value['total']:<4d}  "
                    f"{rate:6.1%}"
                )
        elif key in {"total", "solved"}:
            print(f"{key:14s} {value}")


if __name__ == "__main__":
    raise SystemExit(main())
