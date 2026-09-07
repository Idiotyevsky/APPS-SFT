#!/usr/bin/env python3
"""Tiny (2-sample) overfit diagnostic gate.

Train: problem_submit 271:m2 + first_failure_run 357:m4, LoRA r32 q/k/v/o,
200 steps (protocol_run05_tiny2). Here: greedy-decode the SAME two training
prefixes (exact LLaMA-Factory tokenization) through each saved checkpoint and
report legal tool call and target-tool match per sample.

Goal: decide whether the attention-only LoRA + LLaMA-Factory path can memorize
a tool-call target at all (distinguishes optimization/representational failure
from protocol-frequency/balance failure).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "/home/zfs02/model/Qwen2.5-Coder-7B-Instruct"
CFG = ROOT / "sft/configs/sft_protocol_tiny2.yaml"
RUN = ROOT / "sft/outputs/protocol_run05_tiny2"
ARTIFACT = RUN / "tiny2_probe.json"

# Two rows only: first problem_submit, then first_failure_run (file order).
STATE_OF = {"271:m2": "problem_submit", "357:m4": "first_failure_run"}
TOOL_OF = {"271:m2": "submit", "357:m4": "run_candidate"}


def load_training_tokens():
    import yaml
    from llamafactory.hparams.data_args import DataArguments
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    from llamafactory.hparams.model_args import ModelArguments
    from llamafactory.model.loader import load_tokenizer
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from transformers import Seq2SeqTrainingArguments

    cfg = yaml.safe_load(CFG.read_text())
    cfg["dataset"] = "coding_agent_train"
    cfg["eval_dataset"] = None
    cfg["tokenized_path"] = None
    model_args = ModelArguments(**{k: v for k, v in cfg.items()
                                    if k in {f.name for f in fields(ModelArguments)}})
    data_args = DataArguments(**{k: v for k, v in cfg.items()
                                  if k in {f.name for f in fields(DataArguments)}})
    FinetuningArguments(**{k: v for k, v in cfg.items()
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
    file_rows = json.loads((ROOT / "data/tiny_protocol_t2/train.json").read_text())
    assert len(file_rows) == len(dataset)
    records = []
    for index in range(len(dataset)):
        ex = dataset[index]
        input_ids = list(ex["input_ids"])
        labels = list(ex["labels"])
        sup = [i for i, v in enumerate(labels) if v != -100]
        records.append({"sample_id": file_rows[index]["sample_id"],
                        "prompt_ids": input_ids[:sup[0]],
                        "target_ids": input_ids[sup[0]:sup[-1] + 1]})
    return tokenizer, records


def probe(adapter_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from synthesis.agent_eval import strict_action

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=False, attn_implementation="sdpa")
    if adapter_dir != "base":
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    _, records = load_training_tokens()
    results = []
    with torch.inference_mode():
        for rec in records:
            ids = torch.tensor([rec["prompt_ids"]], dtype=torch.long, device="cuda:0")
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 do_sample=False, max_new_tokens=1024,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(out[0][ids.shape[1]:].tolist(),
                                    skip_special_tokens=False)
            head_tokens = tokenizer.convert_ids_to_tokens(
                out[0][ids.shape[1]:].tolist()[:8])
            target_tool = TOOL_OF.get(rec["sample_id"])
            try:
                action = strict_action(text)
                legal, parsed, err = True, action["name"], None
            except Exception as exc:
                legal, parsed, err = False, None, str(exc)[:300]
            results.append({
                "sample_id": rec["sample_id"],
                "state_type": STATE_OF.get(rec["sample_id"]),
                "legal": legal, "parsed_name": parsed, "parse_error": err,
                "target_tool": target_tool,
                "target_name_match": parsed == target_tool,
                "head_tokens": head_tokens,
                "gen_head": text[:260],
            })
    del model
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(RUN))
    args = parser.parse_args()

    payload = json.loads(ARTIFACT.read_text()) if ARTIFACT.exists() else {"results": {}}
    name = "base" if args.adapter == "base" else Path(args.adapter).name
    rows = probe(args.adapter)
    payload["results"][name] = rows
    ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("=== adapter:", args.adapter, "===", flush=True)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    legal = sum(r["legal"] for r in rows)
    match = sum(r["target_name_match"] for r in rows)
    print(f"SUMMARY legal {legal}/{len(rows)} target-match {match}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
