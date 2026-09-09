#!/usr/bin/env python3
"""Compare run10b memorization on single- versus multi-bug repair prefixes.

`prepare` uses the exact LLaMA-Factory training preprocessing to save prompt
token IDs. `infer` performs one raw greedy dynamic-LoRA generation per repair
prefix and compares the parsed submit code with both the teacher target and the
current candidate. No grader, constrained decoding, or retraining is involved.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import difflib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_ARTIFACT = ROOT / "sft/outputs/run10b_single_multi_learnability.json"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def state_type(row: dict) -> str:
    turns = row["conversations"]
    target = json.loads(turns[-1]["value"])["name"]
    return {
        (4, "submit"): "first_failure_direct_repair",
        (6, "submit"): "post_run_repair",
        (10, "submit"): "multiround_final_submit",
    }.get((len(turns), target), "other")


def previous_submit(row: dict) -> str:
    found = []
    for turn in row["conversations"][:-1]:
        if turn["from"] == "function_call":
            call = json.loads(turn["value"])
            if call["name"] == "submit":
                found.append(call["arguments"]["code"])
    if not found:
        raise RuntimeError(f"{row['sample_id']}: no previous submit")
    return found[-1]


def prepare(artifact: Path) -> None:
    import yaml
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams.data_args import DataArguments
    from llamafactory.hparams.model_args import ModelArguments
    from llamafactory.model.loader import load_tokenizer
    from transformers import Seq2SeqTrainingArguments

    config_path = ROOT / "sft/outputs/protocol_run10b_qwen25_instruct_weighted_repair.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["dataset"] = "coding_agent_train"
    cfg["eval_dataset"] = None
    cfg["tokenized_path"] = None
    model_args = ModelArguments(**{
        key: value for key, value in cfg.items()
        if key in {field.name for field in fields(ModelArguments)}})
    data_args = DataArguments(**{
        key: value for key, value in cfg.items()
        if key in {field.name for field in fields(DataArguments)}})
    training_args = Seq2SeqTrainingArguments(**{
        key: value for key, value in cfg.items()
        if key in {field.name for field in fields(Seq2SeqTrainingArguments)}})
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = (tokenizer_module["tokenizer"] if isinstance(tokenizer_module, dict)
                 else tokenizer_module.tokenizer)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset = get_dataset(template, model_args, data_args, training_args,
                          stage="sft", tokenizer=tokenizer)["train_dataset"]
    rows = json.loads((ROOT / "data/coding_sft_protocol_v2/train.json")
                      .read_text(encoding="utf-8"))
    manifest = {
        row["sample_id"]: row for row in map(
            json.loads,
            (ROOT / "data/coding_sft_protocol_v2/protocol_warmup_manifest.jsonl")
            .read_text(encoding="utf-8").splitlines())
    }
    wanted_ids = {
        manifest[row["sample_id"]]["problem_id"]
        for row in rows if state_type(row) != "other"
    }
    episodes = {}
    with (ROOT / "data/sft_v5_final/episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            if episode["id"] in wanted_ids:
                episodes[episode["id"]] = episode

    records = []
    for index, row in enumerate(rows):
        state = state_type(row)
        if state == "other":
            continue
        example = dataset[index]
        input_ids = list(example["input_ids"])
        labels = list(example["labels"])
        supervised = [position for position, label in enumerate(labels)
                      if label != -100]
        if not supervised:
            raise RuntimeError(f"{row['sample_id']}: no supervised target")
        start = supervised[0]
        target_call = json.loads(row["conversations"][-1]["value"])
        pid = manifest[row["sample_id"]]["problem_id"]
        metadata = episodes[pid]["metadata"]
        origin = metadata["candidate"]["origin"]
        original_k = (metadata.get("mutation") or {}).get("bug_count", 0)
        current_k = original_k - 1 if state == "multiround_final_submit" else original_k
        records.append({
            "index": index,
            "sample_id": row["sample_id"],
            "problem_id": pid,
            "state_type": state,
            "origin": origin,
            "original_bug_count": original_k,
            "current_bug_count": current_k,
            "complexity_group": "single" if current_k == 1 else "multi",
            "prompt_token_ids": input_ids[:start],
            "target_tool": target_call["name"],
            "target_code": target_call["arguments"]["code"],
            "candidate_code": previous_submit(row),
        })
    if len(records) != 170:
        raise RuntimeError(f"expected 170 repair records, got {len(records)}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({
        "config": str(config_path.resolve()),
        "records": records,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "records": len(records),
        "single": sum(r["complexity_group"] == "single" for r in records),
        "multi": sum(r["complexity_group"] == "multi" for r in records),
        "by_state_group": {
            f"{state}/{group}": sum(
                r["state_type"] == state and r["complexity_group"] == group
                for r in records)
            for state in sorted({r["state_type"] for r in records})
            for group in ("single", "multi")
        },
    }, indent=2), flush=True)


def code_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        a=TOKEN_RE.findall(left), b=TOKEN_RE.findall(right)).ratio()


def normalized_hash(code: str) -> str:
    from synthesis.normalize import normalized_code_hash
    return normalized_code_hash(code)


def aggregate(rows: list[dict]) -> dict:
    parsed = [r for r in rows if r["parsed_code"] is not None]
    return {
        "n": len(rows),
        "valid_action": sum(r["valid_action"] for r in rows),
        "valid_action_rate": sum(r["valid_action"] for r in rows) / len(rows),
        "target_tool_match": sum(r["target_tool_match"] for r in rows),
        "target_tool_match_rate": sum(r["target_tool_match"] for r in rows) / len(rows),
        "exact_target_code": sum(r["exact_target_code"] for r in rows),
        "exact_target_code_rate": sum(r["exact_target_code"] for r in rows) / len(rows),
        "duplicate_candidate": sum(r["duplicate_candidate"] for r in rows),
        "duplicate_candidate_rate": sum(r["duplicate_candidate"] for r in rows) / len(rows),
        "mean_target_similarity_when_parsed": (
            statistics.mean(r["target_similarity"] for r in parsed)
            if parsed else None),
        "mean_candidate_similarity_when_parsed": (
            statistics.mean(r["candidate_similarity"] for r in parsed)
            if parsed else None),
        "mean_movement_toward_target_when_parsed": (
            statistics.mean(r["movement_toward_target"] for r in parsed)
            if parsed else None),
        "moved_toward_target": sum(
            r["parsed_code"] is not None and r["movement_toward_target"] > 1e-12
            for r in rows),
        "moved_toward_target_rate": sum(
            r["parsed_code"] is not None and r["movement_toward_target"] > 1e-12
            for r in rows) / len(rows),
    }


def infer(artifact: Path, adapter: Path, output: Path) -> None:
    from synthesis.agent_eval import strict_action
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    records = payload["records"]
    llm = LLM(
        model="/home/nfs05/model/Qwen2.5-7B-Instruct",
        tokenizer="/home/nfs05/model/Qwen2.5-7B-Instruct",
        dtype="bfloat16", max_model_len=32768,
        gpu_memory_utilization=0.88, seed=42,
        enable_lora=True, max_loras=1, max_lora_rank=32,
    )
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=8192, seed=42,
        skip_special_tokens=False,
        stop=["</tool_call>", "</response>"], include_stop_str_in_output=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": r["prompt_token_ids"]} for r in records],
        sampling, use_tqdm=True,
        lora_request=LoRARequest("single_multi_probe", 1,
                                 str(adapter.resolve())),
    )
    results = []
    for record, output_item in zip(records, outputs):
        generated = output_item.outputs[0]
        text = generated.text
        try:
            action = strict_action(text)
            valid, error = True, None
        except Exception as exc:
            action, valid, error = None, False, str(exc)
        parsed_code = None
        parsed_tool = action.get("name") if action else None
        if action and parsed_tool == "submit":
            parsed_code = action.get("arguments", {}).get("code")
            if not isinstance(parsed_code, str):
                parsed_code = None
        target_similarity = (code_similarity(parsed_code, record["target_code"])
                             if parsed_code is not None else None)
        candidate_similarity = (code_similarity(parsed_code, record["candidate_code"])
                                if parsed_code is not None else None)
        baseline_similarity = code_similarity(
            record["candidate_code"], record["target_code"])
        results.append({
            **{k: v for k, v in record.items()
               if k not in {"prompt_token_ids", "target_code", "candidate_code"}},
            "valid_action": valid,
            "target_tool_match": parsed_tool == record["target_tool"],
            "parsed_tool": parsed_tool,
            "parsed_code": parsed_code,
            "parse_error": error,
            "generation": text,
            "generated_tokens": len(generated.token_ids),
            "finish_reason": generated.finish_reason,
            "exact_target_code": (
                parsed_code is not None
                and normalized_hash(parsed_code) == normalized_hash(record["target_code"])),
            "duplicate_candidate": (
                parsed_code is not None
                and normalized_hash(parsed_code) == normalized_hash(record["candidate_code"])),
            "target_similarity": target_similarity,
            "candidate_similarity": candidate_similarity,
            "candidate_target_similarity": baseline_similarity,
            "movement_toward_target": (
                target_similarity - baseline_similarity
                if target_similarity is not None else None),
        })

    groups = {group: aggregate([r for r in results
                                if r["complexity_group"] == group])
              for group in ("single", "multi")}
    by_state_group = {}
    for state in sorted({r["state_type"] for r in results}):
        for group in ("single", "multi"):
            subset = [r for r in results
                      if r["state_type"] == state
                      and r["complexity_group"] == group]
            if subset:
                by_state_group[f"{state}/{group}"] = aggregate(subset)
    summary = {
        "adapter": str(adapter.resolve()),
        "decoding": {
            "temperature": 0.0, "max_tokens": 8192,
            "constrained": False, "dynamic_lora": True,
        },
        "groups": groups,
        "by_state_group": by_state_group,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "records": results},
                                 ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    report = output.with_suffix(".md")
    lines = [
        "# Single-vs-multi repair learnability probe",
        "",
        "Raw greedy, unconstrained generation on exact LLaMA-Factory training",
        "prefix token IDs with Base + dynamic LoRA.",
        "",
        "| group | N | valid | exact target | duplicate candidate | moved toward target | mean target similarity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("single", "multi"):
        s = groups[group]
        lines.append(
            f"| {group} | {s['n']} | {100*s['valid_action_rate']:.1f}% | "
            f"{100*s['exact_target_code_rate']:.1f}% | "
            f"{100*s['duplicate_candidate_rate']:.1f}% | "
            f"{100*s['moved_toward_target_rate']:.1f}% | "
            f"{s['mean_target_similarity_when_parsed']} |")
    lines.extend(["", "## State-conditioned", ""])
    for key, s in by_state_group.items():
        lines.append(
            f"- {key}: n={s['n']}, valid={s['valid_action_rate']:.3f}, "
            f"exact_target={s['exact_target_code_rate']:.3f}, "
            f"duplicate={s['duplicate_candidate_rate']:.3f}, "
            f"moved_toward={s['moved_toward_target_rate']:.3f}, "
            f"target_similarity={s['mean_target_similarity_when_parsed']}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "infer"))
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--adapter", type=Path, default=(
        ROOT / "sft/outputs/protocol_run10b_qwen25_instruct_weighted_repair/checkpoint-12"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "sft/outputs/run10b_single_multi_learnability_results.json"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.artifact)
    else:
        infer(args.artifact, args.adapter, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
