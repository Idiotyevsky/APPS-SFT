#!/usr/bin/env python3
"""Prepare a LLaMA-Factory tokenized dataset whose ONLY supervised tokens are
the two protocol gates of 271:m2 (problem_submit):

  A: <tool_call> (id 151657)
  C: </tool_call> span

Everything else (prompt history, JSON keys, code body, <|im_end|>) is -100.
Output: DatasetDict {"train": [...]} saved to TOK_PATH for
`llamafactory-cli train` via `tokenized_path:` (official training path).
Sidecar JSON keeps prompt ids + gate positions for rank tracking.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ID = "271:m2"
TOK_PATH = ROOT / "sft/outputs/struct271_tokenized"
SIDE = ROOT / "sft/outputs/struct271_meta.json"


def main():
    import yaml
    from llamafactory.hparams.data_args import DataArguments
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    from llamafactory.hparams.model_args import ModelArguments
    from llamafactory.model.loader import load_tokenizer
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from transformers import Seq2SeqTrainingArguments

    cfg = yaml.safe_load((ROOT / "sft/configs/archive/7b_protocol/sft_protocol.yaml").read_text())
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
    rows = json.loads((ROOT / "data/coding_sft_protocol/train.json").read_text())
    idx = next(i for i, r in enumerate(rows) if r["sample_id"] == SAMPLE_ID)
    ex = dataset[idx]
    input_ids = list(ex["input_ids"])
    labels = [-100] * len(input_ids)
    sup = [i for i, v in enumerate(ex["labels"]) if v != -100]
    q = sup[0]
    labels[q] = input_ids[q]  # A
    decoded = tokenizer.decode(input_ids[sup[0]:sup[-1] + 1], skip_special_tokens=False)
    close_offset = decoded.find("</tool_call>")
    assert close_offset >= 0
    acc, c_first, c_last = "", None, None
    end_char = close_offset + len("</tool_call>") - 1
    for j in range(q, sup[-1] + 1):
        acc += tokenizer.decode([input_ids[j]], skip_special_tokens=False)
        if c_first is None and len(acc) > close_offset:
            c_first = j
        if c_first is not None and len(acc) > end_char:
            c_last = j
            break
    if c_last is None:
        c_last = sup[-1]
    for j in range(c_first, c_last + 1):
        labels[j] = input_ids[j]  # C
    supervised = [i for i, v in enumerate(labels) if v != -100]
    assert len(supervised) == (1 + c_last - c_first + 1), supervised
    (SIDE).write_text(json.dumps({
        "sample_id": SAMPLE_ID, "input_ids": input_ids, "supervised": supervised,
        "A": q, "close_span": [c_first, c_last],
        "n_supervised": len(supervised),
        "supervised_tokens": [tokenizer.decode([input_ids[i]],
                              skip_special_tokens=False) for i in supervised],
    }, ensure_ascii=False, indent=1) + "\n")
    ds = Dataset.from_dict({"input_ids": [input_ids],
                            "attention_mask": [[1] * len(input_ids)],
                            "labels": [labels]})
    DatasetDict({"train": ds}).save_to_disk(str(TOK_PATH))
    print("saved", TOK_PATH, "n_supervised", len(supervised),
          "supervised_tokens", [tokenizer.decode([input_ids[i]],
           skip_special_tokens=False) for i in supervised], flush=True)


if __name__ == "__main__":
    main()
