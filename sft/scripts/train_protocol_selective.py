#!/usr/bin/env python3
"""Train protocol SFT with protocol-aware supervision over submit code bodies.

The raw dataset and LLaMA-Factory formatter remain unchanged. This launcher
wraps the official SFT dataset loader and changes only train labels:

* run_candidate targets remain fully supervised;
Two strategies are supported:

* ``random-submit`` preserves the original experiment: an exact deterministic
  fraction of submit targets remain fully supervised and the rest supervise
  only the wrapper;
* ``state-delta`` fully supervises run actions, uses wrapper-only supervision
  for problem-only submits, and supervises either the complete repair or only
  the changed code regions (plus local context) for repair submits.

Use ``--audit-only`` to execute the real tokenization/masking path without
loading model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
IGNORE_INDEX = -100
CODE_FIELD = '"code": "'
SUBMIT_SUFFIX = '"}}\n</tool_call>'
REPAIR_SUBMIT_STATES = {
    "first_failure_direct_repair",
    "post_run_repair",
    "multiround_final_submit",
}


@dataclass
class AuditComplete(Exception):
    report: dict


def _target_tool(row):
    return json.loads(row["conversations"][-1]["value"])["name"]


def _choose_full_submit(rows, fraction, seed):
    submit = [(i, row["sample_id"]) for i, row in enumerate(rows)
              if _target_tool(row) == "submit"]
    count = round(len(submit) * fraction)
    ranked = sorted(submit, key=lambda item: hashlib.sha256(
        f"{seed}:{item[1]}".encode()).hexdigest())
    return {index for index, _ in ranked[:count]}, len(submit), count


def _flat_ids(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _mask_code_body(feature, tokenizer, keep_body_ranges=(), allow_no_mask=False):
    labels = list(feature["labels"])
    supervised = [i for i, token in enumerate(labels) if token != IGNORE_INDEX]
    if not supervised or supervised != list(range(supervised[0], supervised[-1] + 1)):
        raise ValueError("input target must be one contiguous supervised suffix before selective masking")
    target_ids = [labels[i] for i in supervised]
    target = tokenizer.decode(target_ids, skip_special_tokens=False)
    if not target.startswith("<tool_call>\n") or not target.endswith("</tool_call><|im_end|>"):
        raise ValueError(f"unexpected function target wrapper: {target[:120]!r}")
    marker = target.find(CODE_FIELD)
    body_start = marker + len(CODE_FIELD)
    body_end = target.rfind(SUBMIT_SUFFIX)
    if marker < 0 or body_end <= body_start:
        raise ValueError("cannot locate submit code JSON-string boundaries")
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    encoded_ids = _flat_ids(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets and len(offsets) == 1:
        offsets = offsets[0]
    offsets = [tuple(item) for item in offsets]
    if encoded_ids != target_ids or len(offsets) != len(target_ids):
        raise ValueError("target re-tokenization mismatch; refusing approximate masking")
    keep_body_ranges = tuple(keep_body_ranges)
    masked = []
    code_tokens = 0
    kept_code_tokens = 0
    for relative, (start, end) in enumerate(offsets):
        # Keep boundary-crossing tokens so structural quotes are never masked.
        if start >= body_start and end <= body_end and end > start:
            code_tokens += 1
            body_relative = (start - body_start, end - body_start)
            keep = any(body_relative[0] < right and left < body_relative[1]
                       for left, right in keep_body_ranges)
            if keep:
                kept_code_tokens += 1
            else:
                labels[supervised[relative]] = IGNORE_INDEX
                masked.append(supervised[relative])
    if not masked and not allow_no_mask:
        raise ValueError("submit code body produced no maskable tokens")
    kept = [i for i, token in enumerate(labels) if token != IGNORE_INDEX]
    if not kept or kept[-1] != len(labels) - 1:
        raise ValueError("selective labels must retain the target suffix through EOS")
    result = dict(feature)
    result["labels"] = labels
    return result, len(supervised), len(masked), code_tokens, kept_code_tokens, target


def _weighted_feature(feature, tokenizer, body_weight, edit_ranges=(),
                      neighborhood_ranges=(), edit_weight=1.0,
                      neighborhood_weight=0.5, wrapper_weight=1.0):
    """Attach token weights while preserving the full teacher-forced target."""
    labels = list(feature["labels"])
    supervised = [i for i, token in enumerate(labels) if token != IGNORE_INDEX]
    if not supervised or supervised != list(range(supervised[0], supervised[-1] + 1)):
        raise ValueError("input target must be one contiguous supervised suffix")
    target_ids = [labels[i] for i in supervised]
    target = tokenizer.decode(target_ids, skip_special_tokens=False)
    weights = [0.0] * len(labels)
    for index in supervised:
        weights[index] = wrapper_weight

    marker = target.find(CODE_FIELD)
    if marker < 0:  # run_candidate: the complete short target has weight 1.
        result = dict(feature)
        result["loss_weights"] = weights
        return result, {"wrapper_tokens": len(supervised), "body_tokens": 0,
                        "base_body_tokens": 0, "neighborhood_tokens": 0,
                        "edit_tokens": 0, "weight_sum": sum(weights)}

    body_start = marker + len(CODE_FIELD)
    body_end = target.rfind(SUBMIT_SUFFIX)
    if body_end <= body_start:
        raise ValueError("cannot locate submit code JSON-string boundaries")
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    encoded_ids = _flat_ids(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and len(offsets) == 1:
        offsets = offsets[0]
    offsets = [tuple(item) for item in offsets]
    if encoded_ids != target_ids or len(offsets) != len(target_ids):
        raise ValueError("target re-tokenization mismatch; refusing approximate weighting")

    counts = Counter(wrapper_tokens=len(supervised), body_tokens=0,
                     base_body_tokens=0, neighborhood_tokens=0, edit_tokens=0)
    for relative, (start, end) in enumerate(offsets):
        if not (start >= body_start and end <= body_end and end > start):
            continue
        counts["body_tokens"] += 1
        left, right = start - body_start, end - body_start
        weight = body_weight
        bucket = "base_body_tokens"
        if any(left < b and a < right for a, b in neighborhood_ranges):
            weight = neighborhood_weight
            bucket = "neighborhood_tokens"
        if any(left < b and a < right for a, b in edit_ranges):
            weight = edit_weight
            bucket = "edit_tokens"
        weights[supervised[relative]] = weight
        counts[bucket] += 1
    if counts["body_tokens"] == 0:
        raise ValueError("submit code body produced no weightable tokens")
    counts["wrapper_tokens"] -= counts["body_tokens"]
    result = dict(feature)
    result["loss_weights"] = weights
    counts["weight_sum"] = sum(weights)
    return result, dict(counts)


def _escaped_boundaries(code):
    """Map Python string character boundaries into its JSON-string body."""
    boundaries = [0]
    for char in code:
        escaped = json.dumps(char, ensure_ascii=False)[1:-1]
        boundaries.append(boundaries[-1] + len(escaped))
    return boundaries


def _merge_ranges(ranges):
    merged = []
    for left, right in sorted(ranges):
        if left >= right:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def _delta_body_ranges(previous, target, context_chars):
    """Return JSON-escaped target ranges around code edits."""
    boundaries = _escaped_boundaries(target)
    char_ranges = []
    matcher = difflib.SequenceMatcher(a=previous, b=target, autojunk=False)
    for tag, _a0, _a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = max(0, b0 - context_chars)
        right = min(len(target), b1 + context_chars)
        if left == right and target:
            left = max(0, left - 1)
            right = min(len(target), right + 1)
        char_ranges.append((left, right))
    if not char_ranges:
        raise ValueError("repair target is identical to previous candidate")
    return _merge_ranges((boundaries[left], boundaries[right])
                         for left, right in _merge_ranges(char_ranges))


def _previous_submit_code(row):
    previous = []
    for message in row["conversations"][:-1]:
        if message.get("from") != "function_call":
            continue
        action = json.loads(message["value"])
        if action.get("name") == "submit":
            previous.append(action["arguments"]["code"])
    if not previous:
        raise ValueError(f"repair sample has no prior submit: {row['sample_id']}")
    return previous[-1]


def _dataset_file(data_args):
    info_path = Path(data_args.dataset_dir) / "dataset_info.json"
    info = json.loads(info_path.read_text())
    names = [name.strip() for name in data_args.dataset if name.strip()]
    if len(names) != 1:
        raise ValueError("selective protocol launcher requires exactly one train dataset")
    return Path(data_args.dataset_dir) / info[names[0]]["file_name"]


def _load_states(data_args, rows):
    path = _dataset_file(data_args).with_name("protocol_warmup_manifest.jsonl")
    if not path.is_file():
        raise ValueError(f"missing protocol manifest: {path}")
    by_id = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        by_id[item["sample_id"]] = item["state_type"]
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in by_id]
    if missing:
        raise ValueError(f"manifest missing sample ids: {missing[:5]}")
    return [by_id[row["sample_id"]] for row in rows]


def _choose_full_repairs(rows, states, fraction, seed):
    selected = set()
    counts = Counter()
    selected_counts = Counter()
    for state in sorted(REPAIR_SUBMIT_STATES):
        candidates = [(i, row["sample_id"]) for i, (row, current) in
                      enumerate(zip(rows, states)) if current == state]
        counts[state] = len(candidates)
        count = round(len(candidates) * fraction)
        ranked = sorted(candidates, key=lambda item: hashlib.sha256(
            f"{seed}:{state}:{item[1]}".encode()).hexdigest())
        chosen = ranked[:count]
        selected.update(index for index, _ in chosen)
        selected_counts[state] = len(chosen)
    return selected, counts, selected_counts


def _install_dataset_wrapper(report_path, fraction, selection_seed, audit_only,
                             strategy="random-submit", full_repair_fraction=0.20,
                             delta_context_chars=16, problem_code_weight=0.05,
                             unchanged_weight=0.1, neighborhood_weight=0.5,
                             edit_weight=1.0):
    import llamafactory.train.sft.workflow as workflow

    original_get_dataset = workflow.get_dataset

    def selective_get_dataset(template, model_args, data_args, training_args,
                              stage, **tokenizer_module):
        module = original_get_dataset(template, model_args, data_args,
                                      training_args, stage, **tokenizer_module)
        if stage != "sft" or not training_args.do_train:
            return module
        tokenizer = tokenizer_module["tokenizer"]
        rows = json.loads(_dataset_file(data_args).read_text())
        states = _load_states(data_args, rows)
        dataset = module["train_dataset"]
        if len(dataset) != len(rows):
            raise ValueError(f"raw/tokenized row mismatch: {len(rows)} != {len(dataset)}")
        submit_count = sum(_target_tool(row) == "submit" for row in rows)
        repair_counts = Counter()
        selected_repair_counts = Counter()
        if strategy == "random-submit":
            full_indices, submit_count, full_count = _choose_full_submit(
                rows, fraction, selection_seed)
        elif strategy == "state-delta":
            full_indices, repair_counts, selected_repair_counts = _choose_full_repairs(
                rows, states, full_repair_fraction, selection_seed)
            full_count = len(full_indices)
        else:
            full_indices, full_count = set(), 0
            repair_counts.update(state for state in states if state in REPAIR_SUBMIT_STATES)

        totals = Counter()
        problems = []

        def transform(feature, index):
            row = rows[index]
            state = states[index]
            tool = _target_tool(row)
            labels = list(feature["labels"])
            supervised_before = sum(token != IGNORE_INDEX for token in labels)
            decoded = tokenizer.decode(
                [token for token in labels if token != IGNORE_INDEX],
                skip_special_tokens=False)
            expected_name = f'"name": "{tool}"'
            if expected_name not in decoded:
                raise ValueError(f"row alignment failure at {index} {row['sample_id']}")
            totals["samples"] += 1
            totals[f"state_{state}_samples"] += 1
            totals[f"{tool}_samples"] += 1
            totals["supervised_before"] += supervised_before
            totals[f"{tool}_supervised_before"] += supervised_before
            if strategy == "weighted-repair":
                if tool == "run_candidate":
                    result, weight_stats = _weighted_feature(
                        feature, tokenizer, body_weight=1.0)
                    policy = "run_weighted_full"
                elif state == "problem_submit":
                    result, weight_stats = _weighted_feature(
                        feature, tokenizer, body_weight=problem_code_weight)
                    policy = "problem_submit_weighted"
                elif state in REPAIR_SUBMIT_STATES:
                    target_action = json.loads(row["conversations"][-1]["value"])
                    target_code = target_action["arguments"]["code"]
                    previous_code = _previous_submit_code(row)
                    edit_ranges = _delta_body_ranges(previous_code, target_code, 0)
                    neighborhood_ranges = _delta_body_ranges(
                        previous_code, target_code, delta_context_chars)
                    result, weight_stats = _weighted_feature(
                        feature, tokenizer, body_weight=unchanged_weight,
                        edit_ranges=edit_ranges,
                        neighborhood_ranges=neighborhood_ranges,
                        edit_weight=edit_weight,
                        neighborhood_weight=neighborhood_weight)
                    policy = "repair_submit_weighted"
                else:
                    raise ValueError(f"unsupported weighted state {state}")
                masked = 0
                code_tokens = weight_stats["body_tokens"]
                kept_code_tokens = code_tokens
                for key, value in weight_stats.items():
                    totals[f"weighted_{key}"] += value
            elif tool == "run_candidate":
                result = dict(feature)
                policy = "run_full"
                masked = 0
                code_tokens = 0
                kept_code_tokens = 0
            elif index in full_indices:
                result = dict(feature)
                policy = ("submit_full" if strategy == "random-submit"
                          else "repair_submit_full")
                masked = 0
                code_tokens = 0
                kept_code_tokens = 0
            elif strategy == "state-delta" and state in REPAIR_SUBMIT_STATES:
                target_action = json.loads(row["conversations"][-1]["value"])
                target_code = target_action["arguments"]["code"]
                previous_code = _previous_submit_code(row)
                keep_ranges = _delta_body_ranges(
                    previous_code, target_code, delta_context_chars)
                escaped = json.dumps(target_code, ensure_ascii=False)[1:-1]
                result, _, masked, code_tokens, kept_code_tokens, decoded = \
                    _mask_code_body(feature, tokenizer, keep_ranges, allow_no_mask=True)
                marker = decoded.find(CODE_FIELD) + len(CODE_FIELD)
                body_end = decoded.rfind(SUBMIT_SUFFIX)
                if decoded[marker:body_end] != escaped:
                    raise ValueError(
                        f"JSON code-body mismatch at {index} {row['sample_id']}")
                policy = "repair_submit_delta"
            else:
                result, _, masked, code_tokens, kept_code_tokens, _ = \
                    _mask_code_body(feature, tokenizer)
                policy = ("submit_wrapper_only" if strategy == "random-submit"
                          else "problem_submit_wrapper_only")
            supervised_after = sum(token != IGNORE_INDEX for token in result["labels"])
            totals[f"{policy}_samples"] += 1
            totals["supervised_after"] += supervised_after
            totals[f"{tool}_supervised_after"] += supervised_after
            totals["masked_code_tokens"] += masked
            totals["code_tokens_seen_by_masker"] += code_tokens
            totals["delta_code_tokens_kept"] += kept_code_tokens
            if supervised_after <= 0:
                problems.append(f"row {index}: no supervised token after masking")
            return result

        transformed = dataset.map(
            transform, with_indices=True, load_from_cache_file=False,
            desc="Applying protocol-aware selective labels")
        if strategy == "state-delta":
            wrapper_only_count = totals["problem_submit_wrapper_only_samples"]
            delta_count = totals["repair_submit_delta_samples"]
        elif strategy == "weighted-repair":
            wrapper_only_count = 0
            delta_count = 0
        else:
            wrapper_only_count = submit_count - full_count
            delta_count = 0
        report = {
            "dataset": str(_dataset_file(data_args)),
            "fraction": fraction,
            "strategy": strategy,
            "full_repair_fraction": full_repair_fraction,
            "delta_context_chars": delta_context_chars,
            "weights": {"problem_submit_code": problem_code_weight,
                        "repair_unchanged": unchanged_weight,
                        "repair_neighborhood": neighborhood_weight,
                        "repair_edit": edit_weight, "wrapper": 1.0,
                        "run_candidate": 1.0},
            "selection_seed": selection_seed,
            "submit_samples": submit_count,
            "selected_full_submit": full_count,
            "selected_non_full_submit": submit_count - full_count,
            "selected_wrapper_only_submit": wrapper_only_count,
            "selected_delta_submit": delta_count,
            "repair_state_counts": dict(repair_counts),
            "selected_full_repair_by_state": dict(selected_repair_counts),
            "stats": dict(totals),
            "problems": problems,
            "full_submit_sample_ids": [rows[i]["sample_id"] for i in sorted(full_indices)],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if problems:
            raise ValueError(f"selective label audit failed with {len(problems)} problems")
        module = dict(module)
        module["train_dataset"] = transformed
        if audit_only:
            raise AuditComplete(report)
        return module

    workflow.get_dataset = selective_get_dataset


def _install_weighted_runtime():
    """Teach LLaMA-Factory's collator/trainer to consume loss_weights."""
    import torch
    import torch.nn.functional as F
    import llamafactory.train.sft.workflow as workflow

    base_collator = workflow.SFTDataCollatorWith4DAttentionMask
    base_trainer = workflow.CustomSeq2SeqTrainer

    class WeightedCollator(base_collator):
        def __call__(self, features, return_tensors=None):
            raw_weights = []
            stripped = []
            for feature in features:
                item = dict(feature)
                weights = item.pop("loss_weights", None)
                if weights is None:
                    weights = [1.0 if label != IGNORE_INDEX else 0.0
                               for label in item["labels"]]
                if len(weights) != len(item["labels"]):
                    raise ValueError("loss weight/label length mismatch")
                raw_weights.append(list(weights))
                stripped.append(item)
            # LLaMA-Factory's multimodal collator owns tensor conversion and
            # exposes __call__(features), unlike HF DataCollatorForSeq2Seq.
            batch = super().__call__(stripped)
            length = batch["labels"].shape[1]
            padded = []
            for weights in raw_weights:
                pad = [0.0] * (length - len(weights))
                padded.append(pad + weights if self.tokenizer.padding_side == "left"
                              else weights + pad)
            batch["loss_weights"] = torch.tensor(padded, dtype=torch.float32)
            if batch["loss_weights"].shape != batch["labels"].shape:
                raise AssertionError("loss_weights.shape != labels.shape")
            if not torch.all(batch["loss_weights"][batch["labels"].eq(IGNORE_INDEX)] == 0):
                raise AssertionError("ignored/padding labels must have zero loss weight")
            if not getattr(self, "_weighted_preflight_printed", False):
                print("[weighted-preflight] shapes equal: true; labels==-100 weights==0: true",
                      flush=True)
                self._weighted_preflight_printed = True
            return batch

    class WeightedTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            inputs = dict(inputs)
            weights = inputs.pop("loss_weights")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            shift_weights = weights[:, 1:].to(shift_logits.device)
            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction="none", ignore_index=IGNORE_INDEX,
            ).view_as(shift_labels)
            active_weights = shift_weights * shift_labels.ne(IGNORE_INDEX)
            denominator = active_weights.sum()
            if denominator <= 0:
                raise ValueError("batch has no positive supervised loss weight")
            loss = (per_token * active_weights).sum() / denominator
            if not getattr(self, "_weighted_preflight_printed", False):
                print("[weighted-preflight] alignment: logits[:-1], labels[1:], weights[1:]",
                      flush=True)
                print(f"[weighted-preflight] denominator=sum(valid weights)={denominator.item():.6f}",
                      flush=True)
                self._weighted_preflight_printed = True
            return (loss, outputs) if return_outputs else loss

    workflow.SFTDataCollatorWith4DAttentionMask = WeightedCollator
    workflow.CustomSeq2SeqTrainer = WeightedTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--full-submit-fraction", type=float, default=0.20)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("random-submit", "state-delta", "weighted-repair"),
                        default="random-submit")
    parser.add_argument("--full-repair-fraction", type=float, default=0.20)
    parser.add_argument("--delta-context-chars", type=int, default=16)
    parser.add_argument("--problem-code-weight", type=float, default=0.05)
    parser.add_argument("--unchanged-weight", type=float, default=0.1)
    parser.add_argument("--neighborhood-weight", type=float, default=0.5)
    parser.add_argument("--edit-weight", type=float, default=1.0)
    parser.add_argument("--report", default=str(
        ROOT / "sft/outputs/protocol_selective_label_audit.json"))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.full_submit_fraction < 1.0:
        parser.error("full-submit-fraction must be between 0 and 1")
    if not 0.0 <= args.full_repair_fraction < 1.0:
        parser.error("full-repair-fraction must be in [0, 1)")
    if args.delta_context_chars < 0:
        parser.error("delta-context-chars must be non-negative")
    weights = (args.problem_code_weight, args.unchanged_weight,
               args.neighborhood_weight, args.edit_weight)
    if not all(weight > 0 for weight in weights):
        parser.error("all weighted-repair weights must be positive")
    if not (args.problem_code_weight <= args.unchanged_weight
            <= args.neighborhood_weight <= args.edit_weight):
        parser.error("weights must satisfy problem <= unchanged <= neighborhood <= edit")
    _install_dataset_wrapper(Path(args.report), args.full_submit_fraction,
                             args.selection_seed, args.audit_only,
                             args.strategy, args.full_repair_fraction,
                             args.delta_context_chars, args.problem_code_weight,
                             args.unchanged_weight, args.neighborhood_weight,
                             args.edit_weight)
    if args.strategy == "weighted-repair":
        _install_weighted_runtime()
    from llamafactory.train.tuner import run_exp
    sys.argv = [sys.argv[0], args.config]
    try:
        run_exp()
    except AuditComplete:
        print("selective label audit complete; model weights were not loaded", flush=True)


if __name__ == "__main__":
    main()
