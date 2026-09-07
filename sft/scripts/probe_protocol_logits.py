#!/usr/bin/env python3
"""Token-level logits/rank probe on the 12 protocol training prefixes.

Compare Base / ordinary-40 (protocol_run03/checkpoint-40) /
selective-40 (protocol_run04_selective/checkpoint-40) at three structural
decision points inside the teacher-forced target window:

  A. first protocol token (entry into tool-call mode)
  B. tool name  (submit / run_candidate)
  C. first token of the closing </tool_call> (exit tool-call mode)

Metrics per position per model: top-1 acc, gold-token logprob, gold rank,
top1<->gold logit margin. Prefix/target IDs are the exact LLaMA-Factory
training tokenization (same get_dataset path used by protocol_template_probe).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "/home/zfs02/model/Qwen2.5-Coder-7B-Instruct"
ARTIFACT = ROOT / "sft/outputs/protocol_logits_probe.json"

ADAPTERS = {
    "base": None,
    "ordinary-40": ROOT / "sft/outputs/protocol_run03/checkpoint-40",
    "selective-40": ROOT / "sft/outputs/protocol_run04_selective/checkpoint-40",
}


def state_type(row):
    turns = row["conversations"]
    target = json.loads(turns[-1]["value"])["name"]
    return {
        (2, "submit"): "problem_submit",
        (4, "run_candidate"): "first_failure_run",
        (4, "submit"): "first_failure_direct_repair",
        (6, "submit"): "post_run_repair",
        (8, "run_candidate"): "second_failure_run",
        (10, "submit"): "multiround_final_submit",
    }[key] if (key := (len(turns), target)) in {
        (2, "submit"), (4, "run_candidate"), (4, "submit"),
        (6, "submit"), (8, "run_candidate"), (10, "submit")} else "other"


def load_training_tokens():
    """Return (tokenizer, rows) with exact LF training input_ids/labels for the
    first two prefixes of every state type (deterministic, mirrors the
    memorization gate probes)."""
    import yaml
    from llamafactory.hparams.data_args import DataArguments
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    from llamafactory.hparams.model_args import ModelArguments
    from llamafactory.model.loader import load_tokenizer
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from transformers import Seq2SeqTrainingArguments

    cfg = yaml.safe_load((ROOT / "sft/configs/sft_protocol.yaml").read_text())
    cfg["dataset"] = "coding_agent_train"
    cfg["eval_dataset"] = None
    cfg["tokenized_path"] = None
    model_args = ModelArguments(**{k: v for k, v in cfg.items()
                                    if k in {f.name for f in fields(ModelArguments)}})
    data_args = DataArguments(**{k: v for k, v in cfg.items()
                                  if k in {f.name for f in fields(DataArguments)}})
    finetuning_args = FinetuningArguments(**{
        k: v for k, v in cfg.items()
        if k in {f.name for f in fields(FinetuningArguments)}})
    train_fields = {f.name for f in fields(Seq2SeqTrainingArguments)}
    training_args = Seq2SeqTrainingArguments(**{
        k: v for k, v in cfg.items() if k in train_fields})
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = (tokenizer_module["tokenizer"] if isinstance(tokenizer_module, dict)
                 else tokenizer_module.tokenizer)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset = get_dataset(template, model_args, data_args, training_args,
                          stage="sft", tokenizer=tokenizer)["train_dataset"]
    rows = json.loads((ROOT / "data/coding_sft_protocol/train.json").read_text())

    chosen, counts = [], {}
    for index, row in enumerate(rows):
        kind = state_type(row)
        if counts.get(kind, 0) < 2:
            chosen.append((index, kind, row))
            counts[kind] = counts.get(kind, 0) + 1
        if len(chosen) == 12:
            break

    records = []
    for index, kind, row in chosen:
        example = dataset[index]
        input_ids = list(example["input_ids"])
        labels = list(example["labels"])
        supervised = [i for i, v in enumerate(labels) if v != -100]
        start = supervised[0]
        end = supervised[-1] + 1
        records.append({
            "index": index, "sample_id": row["sample_id"],
            "state_type": kind,
            "target_tool": json.loads(row["conversations"][-1]["value"])["name"],
            "prompt_ids": input_ids[:start],
            "target_ids": input_ids[start:end],
        })
    return tokenizer, records


def position_indexes(tokenizer, record):
    """Locate (row_of_distribution, gold_token_id) triples for the three gates
    plus their human-readable names. Rows are computed on the FULL
    prompt+target ids; logits row r predicts ids[r+1]."""
    ids = record["prompt_ids"] + record["target_ids"]
    q = len(record["prompt_ids"])  # first target token position (predicts ids[q])
    target_tool = record["target_tool"]

    decoded = tokenizer.decode(record["target_ids"], skip_special_tokens=False)
    tool_needle = f'"{target_tool}"'
    tool_offset = decoded.find(tool_needle)
    assert tool_offset >= 0, (record["sample_id"], tool_needle, decoded[:200])
    close_needle = "</tool_call>"
    close_offset = decoded.find(close_needle)
    assert close_offset >= 0, (record["sample_id"], close_needle, decoded[-200:])

    def token_index_at(char_offset, start_pos):
        acc = ""
        for j in range(start_pos, len(ids)):
            acc += tokenizer.decode([ids[j]], skip_special_tokens=False)
            if len(acc) > char_offset:
                return j
        raise AssertionError(("char-offset overflow", char_offset, acc[-200:]))

    t = token_index_at(tool_offset, q)
    c = token_index_at(close_offset, q)
    return [
        {"name": "A_action_start", "row": q - 1, "gold": ids[q],
         "gold_text": tokenizer.decode([ids[q]], skip_special_tokens=False)},
        {"name": "B_tool_name", "row": t - 1, "gold": ids[t],
         "gold_text": tokenizer.decode([ids[t]], skip_special_tokens=False)},
        {"name": "C_close_start", "row": c - 1, "gold": ids[c],
         "gold_text": tokenizer.decode([ids[c]], skip_special_tokens=False)},
    ]


def run_probe(adapter_path, tokenizer, records, device="cuda:0"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=False, attn_implementation="sdpa")
    label = "base"
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        label = adapter_path.parent.name + "/" + adapter_path.name
    model.eval()

    per_record = []
    with torch.inference_mode():
        for record in records:
            ids = torch.tensor([record["prompt_ids"] + record["target_ids"]],
                               dtype=torch.long, device=device)
            logits = model(input_ids=ids).logits[0].float()  # (N, V) fp32
            gates = position_indexes(tokenizer, record)
            row = {}
            for g in gates:
                logit_row = logits[g["row"]]
                logp_row = logit_row.log_softmax(dim=-1)
                top1 = int(logit_row.argmax())
                gold_lp = float(logp_row[g["gold"]].item())
                margin = float(logit_row[top1].item() - logit_row[g["gold"]].item())
                rank = int((logit_row >= logit_row[g["gold"]]).sum().item())
                row[g["name"]] = {
                    "top1_token_id": top1,
                    "top1_text": tokenizer.decode([top1], skip_special_tokens=False),
                    "top1_is_gold": top1 == g["gold"],
                    "gold_text": g["gold_text"],
                    "gold_logprob": gold_lp,
                    "gold_rank": rank,
                    "margin": margin,
                }
            per_record.append({"sample_id": record["sample_id"],
                               "state_type": record["state_type"],
                               "target_tool": record["target_tool"], **row})
    del model
    return per_record


def aggregate(records):
    agg = {}
    for gate in ("A_action_start", "B_tool_name", "C_close_start"):
        vals = [r[gate] for r in records]
        agg[gate] = {
            "top1_acc": sum(v["top1_is_gold"] for v in vals) / len(vals),
            "mean_gold_logprob": sum(v["gold_logprob"] for v in vals) / len(vals),
            "median_gold_rank": sorted(v["gold_rank"] for v in vals)[len(vals) // 2],
            "max_gold_rank": max(v["gold_rank"] for v in vals),
            "mean_margin": sum(v["margin"] for v in vals) / len(vals),
        }
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    tokenizer, records = load_training_tokens()

    print("tokenization sanity:",
          tokenizer.encode("<tool_call>\n", add_special_tokens=False),
          tokenizer.tokenize("<tool_call>\n"), flush=True)

    out = {"tokenization_sanity": {
        "tool_call_open_ids": tokenizer.encode("<tool_call>\n", add_special_tokens=False),
        "tool_call_open_tokens": tokenizer.tokenize("<tool_call>\n"),
        "close_ids": tokenizer.encode("</tool_call>", add_special_tokens=False)},
        "records": [], "by_model": {}}
    for name, adapter in ADAPTERS.items():
        rows = run_probe(adapter, tokenizer, records, device=args.device)
        out["records"].append({"model": name, "rows": rows})
        out["by_model"][name] = aggregate(rows)
        print("\n===", name, "===", flush=True)
        for r in rows:
            a, b, c = r["A_action_start"], r["B_tool_name"], r["C_close_start"]
            print(f"{r['sample_id']:>10} {r['state_type']:<26} "
                  f"A top1={a['top1_text']!r:<18} gold={a['gold_text']!r:<22} "
                  f"rank={a['gold_rank']:>4} llh={a['gold_logprob']:7.2f} "
                  f"margin={a['margin']:7.2f} | "
                  f"B rank={b['gold_rank']:>3} acc={int(b['top1_is_gold'])} | "
                  f"C rank={c['gold_rank']:>4} acc={int(c['top1_is_gold'])} "
                  f"top1={c['top1_text']!r:<14}", flush=True)
        print("AGG", json.dumps(out["by_model"][name]), flush=True)
    ARTIFACT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("saved", ARTIFACT, flush=True)


if __name__ == "__main__":
    main()
