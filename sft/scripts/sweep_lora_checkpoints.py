#!/usr/bin/env python3
"""First-submit executable-rate sweep for one base model and many LoRA checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synthesis.agent_eval import (
    EvalConfig,
    PROTOCOL_VERSION,
    ProgressStore,
    action_json_schema,
    evaluate_problem,
    file_digest,
    problem_id,
    select_rows,
    summarize,
)
from synthesis.io_utils import atomic_write_json, read_jsonl


def adapter_identity(path: Path) -> dict:
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    if not all(item.is_file() for item in required):
        raise ValueError(f"invalid LoRA checkpoint: {path}")
    config = json.loads(required[0].read_text())
    return {
        "path": str(path.resolve()),
        "rank": config.get("r"),
        "files": {item.name: file_digest(item) for item in required},
    }


def diagnostic_summary(rows: list[dict]) -> dict:
    observations = []
    for row in rows:
        turns = row.get("turn_history", [])
        observations.append((turns[0].get("observation") or {}).get("status") if turns else None)
    return {
        "syntax_valid": sum(status not in {None, "compile_error"} for status in observations),
        "compile_errors": observations.count("compile_error"),
        "truncated": sum(row["status"] == "truncated" for row in rows),
        "accepted": sum(row["solved"] for row in rows),
        "first_observation_status": {
            status: observations.count(status) for status in sorted(set(observations), key=str)
        },
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-path", default=str(ROOT / "data/eval/toolapps_eval.jsonl"))
    parser.add_argument("--per-difficulty", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    model = Path(args.model).resolve()
    adapter_root = Path(args.adapter_root).resolve()
    checkpoints = sorted(
        adapter_root.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    if not checkpoints:
        parser.error("no checkpoint-* adapters found")
    rows = select_rows(list(read_jsonl(args.eval_path)), 0, args.per_difficulty)
    if not rows:
        parser.error("empty evaluation selection")

    # Reuse the exact model hashes computed by the completed Base evaluation.
    # The absolute model path is checked before reuse, and the source manifest is
    # recorded so this shortcut remains auditable.
    base_manifest_path = ROOT / "data/eval/results/agent/base_v5_constrained_balanced30_t8k/run_manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    if Path(base_manifest["model"]["path"]).resolve() != model:
        raise ValueError("cached Base identity belongs to a different model path")
    base_identity = base_manifest["model"]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    config = EvalConfig(
        mode="problem-only", max_model_len=args.max_model_len, max_actions=1,
        max_submits=1, max_runs=1, max_new_tokens=args.max_new_tokens,
        max_total_new_tokens=args.max_new_tokens, save_completions=True,
    )
    llm = LLM(
        model=str(model), tokenizer=str(model), dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1, seed=42, enable_lora=True,
        max_loras=1, max_lora_rank=32,
    )
    schema = StructuredOutputsParams(json=action_json_schema(("submit",)))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    variants = [("base", None, None)] + [
        (checkpoint.name, checkpoint, LoRARequest(checkpoint.name, index, str(checkpoint)))
        for index, checkpoint in enumerate(checkpoints, 1)
    ]

    for label, adapter, request in variants:
        out_dir = output_root / label
        identity = base_identity if adapter is None else {
            "base": base_identity, "adapter": adapter_identity(adapter)
        }
        source_files = sorted((ROOT / "src/synthesis").rglob("*.py")) + [Path(__file__).resolve()]
        manifest = {
            "protocol": PROTOCOL_VERSION,
            "purpose": "first-submit-lora-checkpoint-sweep",
            "model": identity,
            "base_identity_source": str(base_manifest_path),
            "config": asdict(config),
            "eval_sha256": file_digest(args.eval_path),
            "selected_ids": [problem_id(row) for row in rows],
            "selection": {"per_difficulty": args.per_difficulty},
            "greedy": True,
            "seed": 42,
            "dtype": "bfloat16",
            "decoding": {"constrain_actions": True},
            "versions": {name: version(name) for name in ("transformers", "vllm", "torch")},
            "sources": {str(path.relative_to(ROOT)): file_digest(path) for path in source_files},
        }
        store = ProgressStore(out_dir, manifest, resume=False)
        started = time.monotonic()

        def generate(ids, budget, allowed_names):
            if tuple(allowed_names) != ("submit",):
                raise ValueError("first-submit sweep unexpectedly exposed another tool")
            sampling = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=budget, seed=42,
                skip_special_tokens=False, structured_outputs=schema,
            )
            completion = llm.generate(
                [{"prompt_token_ids": ids}], sampling, use_tqdm=False,
                lora_request=request,
            )[0].outputs[0]
            text = completion.text
            if tokenizer.eos_token and text.endswith(tokenizer.eos_token):
                text = text[:-len(tokenizer.eos_token)]
            return {
                "text": text,
                "token_count": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }

        for index, row in enumerate(rows, 1):
            print(f"{label}: starting {index}/{len(rows)} id={problem_id(row)}", flush=True)
            result = evaluate_problem(row, tokenizer, generate, config)
            store.append(result)
            print(
                f"{label}: {index}/{len(rows)} id={result['id']} status={result['status']} "+
                f"tokens={result['generated_tokens']}", flush=True,
            )
        completed = [store.records[problem_id(row)] for row in rows]
        payload = {
            "protocol": PROTOCOL_VERSION,
            "run_hash": store.hash,
            "summary": summarize(completed, len(rows)),
            "diagnostic": diagnostic_summary(completed),
            "by_problem": completed,
            "session_wall_seconds": round(time.monotonic() - started, 1),
        }
        atomic_write_json(out_dir / "results.json", payload)
        all_summaries[label] = {
            "summary": payload["summary"], "diagnostic": payload["diagnostic"],
            "session_wall_seconds": payload["session_wall_seconds"],
        }
        atomic_write_json(output_root / "summary.json", all_summaries)
        print(label, json.dumps(all_summaries[label], ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
