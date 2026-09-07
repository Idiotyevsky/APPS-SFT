#!/usr/bin/env python3
"""Merge a LoRA in FP32, then export FP16 weights without BF16 rounding loss."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args()

    output = Path(args.output_dir)
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        parser.error("output or incomplete directory already exists")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    print("Loading Base explicitly in float32...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    first = next(model.parameters())
    if first.dtype != torch.float32:
        raise RuntimeError(f"Base loaded as {first.dtype}, expected float32")

    print("Loading and merging LoRA in float32...", flush=True)
    peft_model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model = peft_model.merge_and_unload(safe_merge=True)
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("LoRA merge did not remain in float32")

    print("Converting merged model to float16...", flush=True)
    model = model.to(dtype=torch.float16)
    model.config.torch_dtype = torch.float16
    incomplete.mkdir(parents=True)
    print(f"Writing FP16 shards to {incomplete}...", flush=True)
    model.save_pretrained(
        incomplete, max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(incomplete)
    incomplete.rename(output)
    print(f"Saved complete FP16 merged model: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
