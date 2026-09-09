#!/usr/bin/env python3
"""Real, budgeted ToolAPPS evaluation (not single-shot code pass@1).

Example: python sft/scripts/evaluate_sft_agent.py --model /path/to/model \
  --per-difficulty 10 --output-dir data/eval/results/agent/sft-smoke
No model is loaded when importing this module or requesting --help.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synthesis.agent_eval import (
    AgentEnv, EvalConfig, MODES, PROTOCOL_VERSION, ProgressStore,
    action_json_schema, build_initial_messages, digest, evaluate_problem,
    file_digest, parse_action, problem_id, select_rows, summarize,
)
from synthesis.io_utils import atomic_write_json, read_jsonl


def model_identity(model, revision):
    path = Path(model)
    if path.is_dir():
        if (path / "adapter_config.json").exists():
            raise ValueError("use the merged model, not an unmerged LoRA adapter directory")
        files = sorted(p for p in path.rglob("*") if p.is_file() and
                       (p.suffix in {".json", ".jinja", ".safetensors", ".bin", ".model", ".txt"}))
        if not any(p.suffix in {".safetensors", ".bin"} for p in files):
            raise ValueError("local model has no weights")
        print("Fingerprinting local model weights and tokenizer for safe resume...", flush=True)
        return {"path": str(path.resolve()), "files": {
            str(p.relative_to(path)): file_digest(p) for p in files}}
    if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("remote models require an immutable 40-character --revision")
    return {"model": model, "revision": revision}


def adapter_identity(path):
    path = Path(path)
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    if not all(item.is_file() for item in required):
        raise ValueError(f"invalid LoRA adapter directory: {path}")
    config = json.loads(required[0].read_text())
    return {"path": str(path.resolve()), "rank": config.get("r"),
            "files": {item.name: file_digest(item) for item in required}}


def load_candidates(path, rows):
    values = {}
    for item in read_jsonl(path):
        key = problem_id(item)
        if key in values:
            raise ValueError(f"duplicate candidate id {key}")
        if not isinstance(item.get("code"), str) or not item["code"].strip():
            raise ValueError(f"invalid candidate code for {key}")
        values[key] = item
    selected = {}
    for row in rows:
        key = problem_id(row)
        if key not in values:
            raise ValueError(f"missing fixed candidate for {key}")
        item = values[key]
        if item.get("eval_row_sha256") != digest(row):
            raise ValueError(f"candidate must bind to exact eval row via eval_row_sha256: {key}")
        selected[key] = item["code"]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-adapter",
                        help="optional unmerged PEFT adapter; apply dynamically in vLLM")
    parser.add_argument("--revision")
    parser.add_argument("--eval-path", default=str(ROOT / "data/eval/toolapps_eval.jsonl"))
    parser.add_argument("--output-dir", required=True, help="one directory per model/protocol/config")
    parser.add_argument("--mode", choices=MODES, default="problem-only")
    parser.add_argument("--candidates", help="fixed JSONL: id, code, eval_row_sha256")
    parser.add_argument("--limit", type=int, default=0, help="ordered prefix; use --per-difficulty for balanced smoke")
    parser.add_argument("--per-difficulty", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--max-submits", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=None, help="default: model config (32768 for local Qwen 7B)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="optional cap; default: ALL remaining context each turn")
    parser.add_argument("--max-total-new-tokens", type=int, default=None,
                        help="optional trajectory cap; default: max_actions * context window")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16",
                        help="vLLM weight/compute dtype; use float16 for precision-preserving FP16 LoRA merges")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--save-completions", action="store_true")
    parser.add_argument("--constrain-actions", action="store_true",
                        help="constrain each model turn to the exact tool-action JSON schema")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if (args.mode != "problem-only") != bool(args.candidates):
        parser.error("candidate/repair modes require --candidates; problem-only must not use it")
    if not 0 < args.gpu_memory_utilization <= 1 or args.tensor_parallel_size < 1:
        parser.error("invalid GPU utilization or tensor parallel size")
    rows = select_rows(list(read_jsonl(args.eval_path)), args.limit, args.per_difficulty)
    if not rows:
        parser.error("empty evaluation set")
    candidates = load_candidates(args.candidates, rows) if args.candidates else {}
    for row in rows:
        if row.get("io_mode") not in {"stdin", "call"} or not row.get("inputs") or len(row["inputs"]) != len(row["outputs"]):
            raise ValueError(f"invalid eval test units: {problem_id(row)}")
    out_dir = Path(args.output_dir)
    if (out_dir / "run_manifest.json").exists() and not args.resume:
        parser.error("run exists; use --resume or a new --output-dir")

    from transformers import AutoConfig, AutoTokenizer
    base_identity = model_identity(args.model, args.revision)
    identity = ({"base": base_identity, "adapter": adapter_identity(args.lora_adapter)}
                if args.lora_adapter else base_identity)
    kwargs = {"revision": args.revision} if args.revision else {}
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=False, **kwargs)
    native = int(model_config.max_position_embeddings)
    max_len = args.max_model_len or native
    if max_len <= 0 or max_len > native:
        parser.error(f"max-model-len must be within configured window {native}; no implicit RoPE extension")
    config = EvalConfig(mode=args.mode, max_model_len=max_len, max_actions=args.max_actions,
        max_submits=args.max_submits, max_runs=args.max_runs, max_new_tokens=args.max_new_tokens,
        max_total_new_tokens=args.max_total_new_tokens, save_completions=args.save_completions)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, **kwargs)
    if not tokenizer.chat_template:
        raise ValueError("model tokenizer has no chat template")
    source_files = sorted((ROOT / "src/synthesis").rglob("*.py")) + [Path(__file__).resolve()]
    manifest = {"protocol": PROTOCOL_VERSION, "model": identity, "config": asdict(config),
        "eval_sha256": file_digest(args.eval_path), "selected_ids": [problem_id(r) for r in rows],
        "selection": {"limit": args.limit, "per_difficulty": args.per_difficulty},
        "candidates_sha256": file_digest(args.candidates) if args.candidates else None,
        "greedy": True, "seed": 42, "dtype": args.dtype, "sandbox_backend": "local",
        "decoding": {"constrain_actions": args.constrain_actions},
        "gpu_memory_utilization": args.gpu_memory_utilization, "tensor_parallel_size": args.tensor_parallel_size,
        "versions": {p: version(p) for p in ("transformers", "vllm", "torch")},
        "sources": {str(p.relative_to(ROOT)): file_digest(p) for p in source_files}}
    store = ProgressStore(out_dir, manifest, args.resume)
    selected_ids = set(manifest["selected_ids"])
    if not set(store.records) <= selected_ids:
        raise ValueError("progress contains unexpected problem ids")
    pending = [r for r in rows if problem_id(r) not in store.records]
    started = time.monotonic()

    def save_summary():
        results = [store.records[problem_id(r)] for r in rows if problem_id(r) in store.records]
        payload = {"protocol": PROTOCOL_VERSION, "run_hash": store.hash,
                   "summary": summarize(results, len(rows)), "by_problem": results,
                   "session_wall_seconds": round(time.monotonic()-started, 1)}
        atomic_write_json(out_dir / "results.json", payload)
        return payload

    if pending:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        lora_kwargs = {}
        lora_request = None
        if args.lora_adapter:
            from vllm.lora.request import LoRARequest
            rank = identity["adapter"].get("rank")
            if not isinstance(rank, int) or rank <= 0:
                raise ValueError("LoRA adapter config must contain a positive integer rank")
            lora_kwargs = {"enable_lora": True, "max_loras": 1, "max_lora_rank": rank}
            lora_request = LoRARequest("eval_adapter", 1, str(Path(args.lora_adapter).resolve()))
        llm = LLM(model=args.model, tokenizer=args.model, dtype=args.dtype,
                  max_model_len=max_len, gpu_memory_utilization=args.gpu_memory_utilization,
                  tensor_parallel_size=args.tensor_parallel_size, seed=42,
                  tokenizer_revision=args.revision, **kwargs, **lora_kwargs)
        # Feed exactly the token IDs used for budget accounting. Do not
        # re-tokenize rendered text and accidentally insert another BOS.
        constrained_cache = {}

        def generate(ids, budget, allowed_names):
            sampling_kwargs = dict(temperature=args.temperature, top_p=1.0, max_tokens=budget,
                                   seed=42, skip_special_tokens=False)
            if args.constrain_actions:
                key = tuple(allowed_names)
                if key not in constrained_cache:
                    constrained_cache[key] = StructuredOutputsParams(
                        json=action_json_schema(key))
                sampling_kwargs["structured_outputs"] = constrained_cache[key]
            else:
                sampling_kwargs.update(stop=["</tool_call>", "</response>"],
                                       include_stop_str_in_output=True)
            sampling = SamplingParams(**sampling_kwargs)
            completion = llm.generate([{"prompt_token_ids": ids}], sampling, use_tqdm=False,
                                      lora_request=lora_request)[0].outputs[0]
            text = completion.text
            # EOS is protocol termination, not part of the JSON action.
            if tokenizer.eos_token and text.endswith(tokenizer.eos_token):
                text = text[:-len(tokenizer.eos_token)]
            return {"text": text, "token_count": len(completion.token_ids),
                    "finish_reason": completion.finish_reason}
        save_summary()
        for row in pending:
            print(f"starting {len(store.records)+1}/{len(rows)} id={problem_id(row)}", flush=True)
            result = evaluate_problem(row, tokenizer, generate, config,
                                      candidates.get(problem_id(row)))
            store.append(result)
            payload = save_summary()
            print(f"{len(store.records)}/{len(rows)} id={result['id']} status={result['status']} "
                  f"solved={payload['summary']['solved']}", flush=True)
    payload = save_summary()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {out_dir / 'results.json'}", flush=True)
    bad = {"infrastructure_error", "generation_error", "candidate_invalid", "candidate_not_wrong"}
    return 1 if any(r["status"] in bad for r in store.records.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
