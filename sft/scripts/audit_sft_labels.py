#!/usr/bin/env python3
"""audit_sft_labels.py — 用 LLaMA-Factory 真实模板审计 labels。

只加载 tokenizer（不加载模型权重），按 sft_lora.yaml 的 template/tool_format/
mask_history/cutoff_len 走官方 get_dataset 预处理，检查：
  * 每样本至少一个监督 token；监督块连续且等于末尾目标
  * 按目标工具统计监督 token 数（对照 sample_manifest/train.json）
  * 结构集(structure)：历史中的失败 submit 不出现在 labels
  * 长度分布（P50/P95/P99/max）

用法：
  SFT_VENV/bin/python sft/scripts/audit_sft_labels.py \
      --config sft/configs/sft_lora.yaml \
      --dataset coding_agent_train \
      --report-dir sft/outputs/data_audit
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    return ordered[hi] if hi == lo else (ordered[lo] + ordered[hi]) / 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="coding_agent_train",
                        choices=["coding_agent_train", "coding_agent_dev",
                                 "coding_agent_smoke", "coding_agent_structure"])
    parser.add_argument("--report-dir", default=str(ROOT / "sft/outputs/data_audit"))
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg["dataset"] = args.dataset
    cfg["eval_dataset"] = None
    cfg["tokenized_path"] = None
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    from llamafactory.hparams.data_args import DataArguments
    from llamafactory.hparams.finetuning_args import FinetuningArguments
    from llamafactory.hparams.model_args import ModelArguments
    from llamafactory.model.loader import load_tokenizer

    model_args = ModelArguments(**{
        k: v for k, v in cfg.items()
        if k in {f.name for f in fields(ModelArguments)}
    })
    data_args = DataArguments(**{
        k: v for k, v in cfg.items()
        if k in {f.name for f in fields(DataArguments)}
    })
    finetuning_args = FinetuningArguments(**{
        k: v for k, v in cfg.items()
        if k in {f.name for f in fields(FinetuningArguments)}
    })

    from transformers import Seq2SeqTrainingArguments
    train_fields = {f.name for f in fields(Seq2SeqTrainingArguments)}
    training_args = Seq2SeqTrainingArguments(**{
        k: v for k, v in cfg.items() if k in train_fields
    })

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = (
        tokenizer_module["tokenizer"]
        if isinstance(tokenizer_module, dict) else tokenizer_module.tokenizer
    )

    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    module = get_dataset(
        template, model_args, data_args, training_args,
        stage="sft", tokenizer=tokenizer,
    )
    dataset = module["train_dataset"]
    print(f"dataset={args.dataset} rows={len(dataset)}", flush=True)

    tool_by_index: list[str | None] = []
    if args.dataset in {
        "coding_agent_train", "coding_agent_dev",
        "coding_agent_smoke", "coding_agent_structure",
    }:
        name = args.dataset.replace("coding_agent_", "")
        path = ROOT / "data/coding_sft" / f"{name}.json"
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                last = row["conversations"][-1]
                tool_by_index.append(
                    json.loads(last["value"]).get("name"))
        else:
            tool_by_index = [None] * len(dataset)

    supervised_tokens = Counter()
    supervised_by_tool = defaultdict(list)
    lengths: list[int] = []
    problems = []
    for index, example in enumerate(dataset):
        input_ids = example["input_ids"]
        labels = example["labels"]
        lengths.append(len(input_ids))
        if len(input_ids) != len(labels):
            problems.append(f"row {index}: length mismatch")
            continue
        spans = [i for i, label in enumerate(labels) if label != -100]
        if not spans:
            problems.append(f"row {index}: no supervised token")
            continue
        if spans != list(range(spans[0], spans[-1] + 1)):
            problems.append(f"row {index}: supervised span not contiguous")
        if spans[-1] != len(labels) - 1:
            problems.append(f"row {index}: supervised span not at sequence end")
        supervised_tokens["total"] += len(spans)
        tool = tool_by_index[index] if index < len(tool_by_index) else None
        if tool:
            supervised_by_tool[tool].append(len(spans))
        if index < 20 and args.dataset.startswith("coding_agent_structure"):
            decoded = tokenizer.decode(
                [input_ids[i] for i in spans], skip_special_tokens=False)
            rows = json.loads(
                (ROOT / "data/coding_sft/structure.json")
                .read_text(encoding="utf-8"))
            history = rows[index]["conversations"][:-1]
            final_value = rows[index]["conversations"][-1]["value"]
            final_call = json.loads(final_value)
            for item in history:
                if item["from"] != "function_call":
                    continue
                call = json.loads(item["value"])
                if item["value"] == final_value:
                    continue  # legitimate repeated run on same input
                is_failed_submit = (
                    call["name"] == "submit"
                    and not (final_call["name"] == "submit"
                             and call["arguments"].get("code")
                             == final_call["arguments"].get("code"))
                )
                if is_failed_submit and item["value"] in decoded:
                    problems.append(
                        f"row {index}: failed submit code leaked into labels")

    for tool, values in supervised_by_tool.items():
        supervised_tokens[tool] = sum(values)
    supervised_tokens["per_sample_mean"] = (
        supervised_tokens["total"] / len(dataset) if dataset else 0)

    stats = {
        "dataset": args.dataset,
        "rows": len(dataset),
        "lengths": {
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths) if lengths else None,
        },
        "supervised": dict(supervised_tokens),
        "supervised_by_tool_mean": {
            tool: sum(vals) / len(vals) if vals else None
            for tool, vals in supervised_by_tool.items()
        },
        "problems": problems,
    }
    report = report_dir / f"audit_{args.dataset}.json"
    report.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print("problems:", len(problems), flush=True)
    for p in problems[:10]:
        print("  ", p, flush=True)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
