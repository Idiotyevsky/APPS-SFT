#!/usr/bin/env python3
"""Raw greedy memorization sweep for every Protocol-v3 LoRA checkpoint.

The prompts are exact LLaMA-Factory training prefix token IDs.  Only repair
targets are included.  Generation is unconstrained and adapters are loaded
dynamically into one vLLM process, so the sweep measures whether each early
checkpoint independently learned the repair action rather than merely whether
the evaluator can coerce a valid JSON object.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

REPAIR_STATES = {
    "first_failure_direct_repair",
    "post_run_repair",
    "multiround_final_submit",
}


def normalized_hash(code: str) -> str | None:
    from synthesis.normalize import normalized_code_hash
    try:
        return normalized_code_hash(code)
    except (SyntaxError, IndentationError, tokenize.TokenError):
        return None


def prepare(config_path: Path, artifact: Path, expected_records: int | None = None,
            all_states: bool = False) -> None:
    import yaml
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams.data_args import DataArguments
    from llamafactory.hparams.model_args import ModelArguments
    from llamafactory.model.loader import load_tokenizer
    from transformers import Seq2SeqTrainingArguments

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

    data_dir = Path(cfg["dataset_dir"])
    rows = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    manifest = {
        row["sample_id"]: row for row in map(
            json.loads,
            (data_dir / "protocol_warmup_manifest.jsonl")
            .read_text(encoding="utf-8").splitlines())
    }
    if len(dataset) != len(rows):
        raise RuntimeError(f"LF/data length mismatch: {len(dataset)} != {len(rows)}")

    records = []
    for index, row in enumerate(rows):
        meta = manifest[row["sample_id"]]
        if not all_states and meta["state_type"] not in REPAIR_STATES:
            continue
        example = dataset[index]
        input_ids = list(example["input_ids"])
        labels = list(example["labels"])
        supervised = [i for i, label in enumerate(labels) if label != -100]
        if not supervised:
            raise RuntimeError(f"{row['sample_id']}: no supervised target")
        start = supervised[0]
        target = json.loads(row["conversations"][-1]["value"])
        if meta["state_type"] in REPAIR_STATES and target["name"] != "submit":
            raise RuntimeError(f"{row['sample_id']}: repair target is not submit")
        records.append({
            "sample_id": row["sample_id"],
            "problem_id": str(meta["problem_id"]),
            "state_type": meta["state_type"],
            "source": meta.get("source"),
            "prompt_token_ids": input_ids[:start],
            "target_tool": target["name"],
            "target_code": (target["arguments"].get("code")
                            if target["name"] == "submit" else None),
        })
    if expected_records is not None and len(records) != expected_records:
        raise RuntimeError(
            f"expected {expected_records} repair records, got {len(records)}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({
        "config": str(config_path.resolve()),
        "records": records,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "records": len(records),
        "by_state": {state: sum(r["state_type"] == state for r in records)
                     for state in sorted(REPAIR_STATES)},
        "by_source": {source: sum(r["source"] == source for r in records)
                      for source in sorted({r["source"] for r in records})},
    }, indent=2), flush=True)


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    exact_eligible = sum(r.get("exact_eligible", False) for r in rows)
    exact = sum(r["exact_target_code"] for r in rows)
    return {
        "n": n,
        "valid_action": sum(r["valid_action"] for r in rows),
        "valid_action_rate": sum(r["valid_action"] for r in rows) / n,
        "target_tool_match": sum(r["target_tool_match"] for r in rows),
        "target_tool_match_rate": sum(r["target_tool_match"] for r in rows) / n,
        "exact_target_code": exact,
        "exact_target_code_eligible": exact_eligible,
        "exact_target_code_rate": exact / exact_eligible if exact_eligible else None,
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / n,
    }


def infer(artifact: Path, adapter_root: Path | None, output: Path,
          model_path: str, base_only: bool = False,
          min_exact: float = 0.50) -> None:
    from synthesis.agent_eval import strict_action
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    records = json.loads(artifact.read_text(encoding="utf-8"))["records"]
    checkpoints = ([] if base_only else sorted(
        adapter_root.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])))
    if not checkpoints and not base_only:
        raise RuntimeError(f"no checkpoints under {adapter_root}")
    for checkpoint in checkpoints:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (checkpoint / name).is_file():
                raise RuntimeError(f"{checkpoint}: missing {name}")

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16", max_model_len=32768,
        gpu_memory_utilization=0.88, seed=42,
        enable_lora=True, max_loras=1, max_lora_rank=32,
    )
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=8192, seed=42,
        skip_special_tokens=False,
        stop=["</tool_call>", "</response>"], include_stop_str_in_output=True,
    )
    decoding = {"temperature": 0.0, "max_tokens": 8192,
                "constrained": False, "dynamic_lora": True}
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("decoding") != decoding:
            raise RuntimeError("existing sweep uses different decoding settings")
    else:
        payload = {"decoding": decoding, "checkpoints": {}}
    variants = [("base", None)] if base_only else [
        (checkpoint.name, checkpoint) for checkpoint in checkpoints]
    for adapter_id, (variant_name, checkpoint) in enumerate(variants, 1):
        if variant_name in payload["checkpoints"]:
            print(f"skipping completed {variant_name}", flush=True)
            continue
        print(f"starting {variant_name} ({len(records)} repair prefixes)", flush=True)
        outputs = llm.generate(
            [{"prompt_token_ids": r["prompt_token_ids"]} for r in records],
            sampling, use_tqdm=True,
            lora_request=(None if checkpoint is None else
                          LoRARequest(checkpoint.name, adapter_id,
                                      str(checkpoint.resolve()))),
        )
        result_rows = []
        for record, output_item in zip(records, outputs):
            generated = output_item.outputs[0]
            text = generated.text
            try:
                action = strict_action(text)
                valid, error = True, None
            except Exception as exc:
                action, valid, error = None, False, str(exc)
            parsed_tool = action.get("name") if action else None
            code = (action.get("arguments", {}).get("code")
                    if action and parsed_tool == "submit" else None)
            if not isinstance(code, str):
                code = None
            exact_eligible = (record["state_type"] in REPAIR_STATES and
                              isinstance(record.get("target_code"), str))
            result_rows.append({
                "sample_id": record["sample_id"],
                "problem_id": record["problem_id"],
                "state_type": record["state_type"],
                "source": record["source"],
                "valid_action": valid,
                "target_tool_match": parsed_tool == record["target_tool"],
                "parsed_tool": parsed_tool,
                "parse_error": error,
                "exact_eligible": exact_eligible,
                "exact_target_code": (exact_eligible and code is not None and
                    normalized_hash(code) == normalized_hash(record["target_code"])),
                "generated_tokens": len(generated.token_ids),
                "finish_reason": generated.finish_reason,
                "generation": text,
            })
        overall = aggregate(result_rows)
        by_state = {state: aggregate([r for r in result_rows
                                      if r["state_type"] == state])
                    for state in sorted({r["state_type"] for r in result_rows})}
        exact_rate = overall["exact_target_code_rate"]
        passed = (overall["valid_action_rate"] >= 0.95 and
                  overall["target_tool_match_rate"] >= 0.90 and
                  (exact_rate is None or exact_rate >= min_exact))
        payload["checkpoints"][variant_name] = {
            "adapter": None if checkpoint is None else str(checkpoint.resolve()),
            "gate_passed": passed,
            "overall": overall,
            "by_state": by_state,
            "records": result_rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(json.dumps({variant_name: {
            "gate_passed": passed, **overall}}, indent=2), flush=True)

    lines = [
        "# Protocol v3 checkpoint repair-prefix sweep", "",
        "Exact LLaMA-Factory training prefixes; Base + dynamic LoRA; raw greedy; unconstrained.", "",
        ("Gate: legal >= 95%, target tool >= 90%, exact repair target >= "
         f"{min_exact:.0%}."), "",
        "| checkpoint | legal | target tool | exact repair | truncated | mean tokens | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, item in payload["checkpoints"].items():
        s = item["overall"]
        lines.append(
            f"| {name} | {s['valid_action_rate']:.1%} | "
            f"{s['target_tool_match_rate']:.1%} | "
            f"{(s['exact_target_code_rate'] or 0):.1%} | "
            f"{s['truncated']} | {s['mean_generated_tokens']:.1f} | "
            f"{'PASS' if item['gate_passed'] else 'FAIL'} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer"))
    parser.add_argument("--config", type=Path,
                        default=ROOT / "sft/configs/sft_protocol_v3.yaml")
    parser.add_argument("--artifact", type=Path,
                        default=ROOT / "sft/outputs/protocol_v3_repair_prefixes.json")
    parser.add_argument("--adapter-root", type=Path, default=ROOT /
                        "sft/outputs/protocol_run11_qwen25_instruct_v3_final")
    parser.add_argument("--model", default="/home/nfs05/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--all-states", action="store_true")
    parser.add_argument("--min-exact", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=ROOT /
                        "sft/outputs/protocol_v3_checkpoint_sweep.json")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.config, args.artifact, args.expected_records, args.all_states)
    else:
        infer(args.artifact, args.adapter_root, args.output, args.model,
              args.base_only, args.min_exact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
