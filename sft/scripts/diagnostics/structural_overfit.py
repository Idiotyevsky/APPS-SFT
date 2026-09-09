#!/usr/bin/env python3
"""Structural-only single-sample overfit on 271:m2 (problem_submit).

Supervise ONLY the two tool-mode gates in the label stream:
  A: the <tool_call> open token (id 151657)
  C: the token(s) of </tool_call>
everything else in the target is masked (-100); prompt tokens also -100.

Question: can q/k/v/o LoRA push A from rank ~50k to rank 1 under direct
structural supervision? If not, rerun with +MLP and +lm_head to isolate the
expressive bottleneck (frozen lm_head + tie_word_embeddings=false).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = "/home/zfs02/model/Qwen2.5-Coder-7B-Instruct"
SAMPLE_ID = "271:m2"

MODULES = {
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qkvo_mlp": ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
    "qkvo_lmhead": ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
}


def load_single(modules_key: str):
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
    input_ids = torch.tensor(ex["input_ids"], dtype=torch.long)
    labels = torch.tensor([-100] * len(input_ids), dtype=torch.long)
    # labels A/C tokens only
    sup = [i for i, v in enumerate(ex["labels"]) if v != -100]
    q = sup[0]
    labels[q] = input_ids[q]  # A: <tool_call> id 151657
    decoded = tokenizer.decode(input_ids[sup[0]:sup[-1] + 1], skip_special_tokens=False)
    close_offset = decoded.find("</tool_call>")
    assert close_offset >= 0
    acc, c_first, c_last = "", None, None
    for j in range(q, sup[-1] + 1):
        piece = tokenizer.decode([input_ids[j].item()], skip_special_tokens=False)
        acc += piece
        if c_first is None and len(acc) > close_offset:
            c_first = j
        if c_first is not None and "</tool_call>" not in acc:
            c_last = j - 1
            break
    if c_last is None:
        c_last = sup[-1]
    labels[c_first:c_last + 1] = input_ids[c_first:c_last + 1]  # C: </tool_call> span
    assert (labels != -100).sum().item() >= 2, "at least A+C must be supervised"
    return tokenizer, input_ids, labels, q, c_first, c_last


def gate_ranks(model, tokenizer, input_ids, q, c_first, c_last):
    with torch.inference_mode():
        logits = model(input_ids=input_ids.unsqueeze(0),
                       use_cache=False).logits[0].float()
        out = {}
        for key, (row, gold) in {
            "A": (q - 1, input_ids[q].item()),
            "C": (c_first - 1, input_ids[c_first].item()),
        }.items():
            row_logits = logits[row]
            gold_lp = row_logits.log_softmax(dim=-1)[gold].item()
            rank = int((row_logits >= row_logits[gold]).sum().item())
            top1 = int(row_logits.argmax())
            out[key] = {"rank": rank, "top1": tokenizer.decode([top1],
                         skip_special_tokens=False), "gold_logprob": gold_lp}
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", default="qkvo",
                        choices=list(MODULES))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers.optimization import get_cosine_schedule_with_warmup
    from transformers.models.qwen2 import modeling_qwen2 as _mqwen

    # transformers' kernel-hub-fused apply_rotary_pos_emb errors through
    # autograd (28 vs 128 head-dim mismatch). Fall back to the vanilla
    # implementation that also tolerates the (S, dim, heads) layout and an
    # over-long cos cache seen under autograd.
    def _vanilla_rope(q, k, cos, sin, unsqueeze_dim=1):
        transposed = q.ndim == 3
        if transposed:
            q = q.permute(2, 0, 1).unsqueeze(0)   # (S,d,H) -> (1,H,S,d)
            k = k.permute(2, 0, 1).unsqueeze(0)
            unsqueeze_dim = 1
        S = q.shape[2]
        cos = cos[:, :S, :]
        sin = sin[:, :S, :]
        q_embed = (q * cos.unsqueeze(unsqueeze_dim)
                   ) + (_mqwen.rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
        k_embed = (k * cos.unsqueeze(unsqueeze_dim)
                   ) + (_mqwen.rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
        if transposed:
            q_embed = q_embed[0].permute(1, 2, 0)
            k_embed = k_embed[0].permute(1, 2, 0)
        return q_embed, k_embed
    _mqwen.apply_rotary_pos_emb = _vanilla_rope

    device = "cuda:0"
    tokenizer, input_ids, labels, q, c_first, c_last = load_single(args.modules)
    n_tok = len(input_ids)
    sup_rows = [q - 1] + list(range(c_first - 1, c_last))
    run = ROOT / f"sft/outputs/struct271_{args.modules}"
    run.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=False, attn_implementation="eager")
    lora = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05,
                      target_modules=MODULES[args.modules],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5,
                                            num_training_steps=args.steps)
    ids = input_ids.to(device)
    trace = []
    start = gate_ranks(model, tokenizer, ids, q, c_first, c_last)
    trace.append({"step": 0, **{f"{k}_rank": v["rank"] for k, v in start.items()},
                  **{f"{k}_top1": v["top1"] for k, v in start.items()}})
    print("step", 0, json.dumps(trace[-1]), flush=True)

    for step in range(1, args.steps + 1):
        opt.zero_grad()
        logits = model(input_ids=ids, use_cache=False).logits[0]
        logits_shift = logits[:-1].float()
        labels_shift = labels[1:].to(device)
        loss = torch.nn.functional.cross_entropy(
            logits_shift, labels_shift, ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 5 == 0 or step == args.steps:
            g = gate_ranks(model, tokenizer, ids, q, c_first, c_last)
            rec = {"step": step, "loss": loss.item(),
                   **{f"{k}_rank": v["rank"] for k, v in g.items()},
                   **{f"{k}_top1": v["top1"] for k, v in g.items()}}
            trace.append(rec)
            print("step", step, json.dumps(rec), flush=True)

    model.save_pretrained(str(run / "final_adapter"))
    (run / "trace.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in trace) + "\n")
    print("saved", run, flush=True)


if __name__ == "__main__":
    main()
