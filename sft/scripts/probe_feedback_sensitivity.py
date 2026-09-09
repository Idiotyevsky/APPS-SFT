#!/usr/bin/env python3
"""Counterfactual observation probe for an existing ToolAPPS trajectory.

This is a diagnostic only: it replays saved history, changes only the most
recent tool observation, and asks a dynamic-LoRA policy for one next action.
It never invokes or changes the evaluator/grader.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from difflib import SequenceMatcher, unified_diff
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synthesis.agent_eval import (  # noqa: E402
    action_json_schema, build_initial_messages, parse_action, problem_id,
)
from synthesis.io_utils import atomic_write_json, read_jsonl  # noqa: E402
from synthesis.tools import tool_schemas  # noqa: E402


def append_event(messages: list[dict], event: dict, observation: dict) -> None:
    action = event["action"]
    messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": action}]})
    messages.append({"role": "tool", "name": action["name"],
                     "content": json.dumps(observation, ensure_ascii=False,
                                           separators=(",", ":"))})


def contradictory(observation: dict) -> dict:
    failing_input = observation.get("failing_input")
    if observation.get("status") == "runtime_error":
        return {"status": "wrong_answer", "passed": 0, "total": 10,
                "pass_rate": 0.0, "failing_input": failing_input}
    return {"status": "runtime_error", "passed": 0,
            "total": observation.get("total", 1), "pass_rate": 0.0,
            "failing_input": failing_input,
            "error": {"type": "IndexError",
                      "message": "list index out of range at line 17"}}


def counterfactual_input(observation: dict, donor: dict) -> dict:
    changed = deepcopy(observation)
    for key in ("failing_input", "stdout", "stderr", "error"):
        if key in donor:
            changed[key] = deepcopy(donor[key])
    if changed == observation:
        changed["failing_input"] = "999\n1 2 3\n"
    return changed


def select_anchors(rows: list[dict], count: int) -> list[dict]:
    anchors = []
    # Balance feedback immediately after submit and immediately after run.
    for wanted_tool in ("submit", "run_candidate"):
        for row in rows:
            for index, event in enumerate(row.get("turn_history", [])):
                obs = event.get("observation")
                if (event.get("tool") == wanted_tool and isinstance(obs, dict)
                        and obs.get("status") != "accepted"):
                    anchors.append({"result": row, "event_index": index})
                    break
            if len([a for a in anchors if a["result"]["turn_history"][
                    a["event_index"]]["tool"] == wanted_tool]) >= count // 2:
                break
    return anchors[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-adapter", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    eval_rows = {problem_id(row): row for row in read_jsonl(args.eval_path)}
    source = json.loads(Path(args.source_results).read_text())
    anchors = select_anchors(source["by_problem"], args.anchors)
    if len(anchors) != args.anchors:
        raise ValueError(f"found only {len(anchors)} usable anchors")
    all_observations = [
        event["observation"] for row in source["by_problem"]
        for event in row.get("turn_history", [])
        if isinstance(event.get("observation"), dict)
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16",
              max_model_len=32768,
              gpu_memory_utilization=args.gpu_memory_utilization,
              seed=42, enable_lora=True, max_loras=1, max_lora_rank=32)
    request = LoRARequest("feedback_probe", 1,
                          str(Path(args.lora_adapter).resolve()))
    schema = StructuredOutputsParams(json=action_json_schema(
        ("run_candidate", "submit")))
    sampling = SamplingParams(temperature=0.0, top_p=1.0,
        max_tokens=args.max_new_tokens, seed=42, skip_special_tokens=False,
        structured_outputs=schema)

    records = []
    for anchor_number, anchor in enumerate(anchors):
        saved = anchor["result"]
        row = eval_rows[saved["id"]]
        index = anchor["event_index"]
        anchor_event = saved["turn_history"][index]
        real = anchor_event["observation"]
        donor = all_observations[(anchor_number + 7) % len(all_observations)]
        shuffled = all_observations[(anchor_number + 19) % len(all_observations)]
        variants = {
            "real": real,
            "null": {"status": "failed",
                     "message": "Execution finished, but no diagnostic details are available."},
            "contradictory": contradictory(real),
            "counterfactual_input": counterfactual_input(real, donor),
            "shuffled": shuffled,
        }
        base_messages = build_initial_messages(row, "problem-only")
        for prior in saved["turn_history"][:index]:
            if "action" in prior and "observation" in prior:
                append_event(base_messages, prior, prior["observation"])

        variant_records = []
        for label, observation in variants.items():
            messages = deepcopy(base_messages)
            append_event(messages, anchor_event, observation)
            ids = tokenizer.apply_chat_template(messages, tools=tool_schemas(),
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            output = llm.generate([{"prompt_token_ids": ids}], sampling,
                                  use_tqdm=False, lora_request=request)[0].outputs[0]
            action = parse_action(output.text)
            variant_records.append({
                "variant": label, "observation": observation,
                "raw": output.text, "action": action,
            })

        real_action = variant_records[0]["action"] or {}
        real_code = real_action.get("arguments", {}).get("code")
        for record in variant_records:
            action = record["action"] or {}
            code = action.get("arguments", {}).get("code")
            record["same_action_as_real"] = action.get("name") == real_action.get("name")
            record["same_code_as_real"] = code == real_code if code is not None and real_code is not None else None
            if code is not None and real_code is not None:
                record["code_similarity_to_real"] = SequenceMatcher(None, real_code, code).ratio()
                record["line_diff_from_real"] = list(unified_diff(
                    real_code.splitlines(), code.splitlines(), lineterm=""))[:80]
        records.append({"id": saved["id"], "anchor_tool": anchor_event["tool"],
                        "anchor_event_index": index, "variants": variant_records})

    comparisons = [v for r in records for v in r["variants"] if v["variant"] != "real"]
    summary = {
        "anchors": len(records), "generations": sum(len(r["variants"]) for r in records),
        "nonreal_comparisons": len(comparisons),
        "same_action_as_real": sum(v["same_action_as_real"] for v in comparisons),
        "submit_comparisons": sum(v["same_code_as_real"] is not None for v in comparisons),
        "same_code_as_real": sum(v["same_code_as_real"] is True for v in comparisons),
        "mean_code_similarity_to_real": (
            sum(v.get("code_similarity_to_real", 0) for v in comparisons
                if v["same_code_as_real"] is not None)
            / max(1, sum(v["same_code_as_real"] is not None for v in comparisons))),
    }
    atomic_write_json(args.output, {"summary": summary, "records": records})
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
