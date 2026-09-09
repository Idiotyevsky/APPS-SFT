#!/usr/bin/env python3
"""prepare_sft.py — 把 ToolAPPS episodes 转成 LLaMA-Factory ShareGPT 前缀样本。

遵循 docs/LLAMA_FACTORY_SFT_IMPLEMENTATION_SPEC.md §3、§6、§7：
  1. 以 QA 通过的 episodes.jsonl 为准，校验 metadata 一一对应；
  2. 按题目(problem id)做 90/10 train/dev 划分（seed 42，题目级互斥）；
  3. 每条原始轨迹中每个 trainable 的 assistant 工具动作展开为一条
     “前缀 + 末轮监督”ShareGPT 样本（conversations/system/tools 顶层字段）；
  4. 产出 train/dev/smoke.json、dataset_info.json、split_manifest.json、
     sample_manifest.jsonl；
  5. 可选 --extra-episodes 读取额外轨迹（如含 masked 失败提交的多轮 fixture），
     转为 structure.json 供 labels 审计，不进 train/dev。

用法：
  python scripts/prepare_sft.py \
      --episodes ../../data/sft_final/episodes.jsonl \
      --metadata ../../data/sft_final/metadata.jsonl \
      --out ../../data/work/exports/coding_sft_current \
      --extra-episodes ../../data/archive/datasets/sft_debug_rule_handoff/episodes.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def record_hash(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def export_prefix_samples(record: dict) -> list[dict]:
    """Spec §3.3 converter: one prefix sample per trainable assistant action.

    Tool availability mirrors the real evaluator: until the first ``submit``
    appears in history only ``submit`` is exposed; afterwards both
    ``run_candidate`` and ``submit`` are exposed.
    """
    messages = record["messages"]
    if not messages or messages[0]["role"] != "system":
        raise ValueError("expected leading system message")
    system = messages[0]["content"]
    if system is None or messages[0].get("trainable") is not False:
        raise ValueError("system must carry trainable=false and content")
    history: list[dict] = []
    samples: list[dict] = []
    submitted = False

    def filtered_tools():
        names = {"submit"}
        if submitted:
            names.add("run_candidate")
        return [
            tool for tool in record["tools"]
            if tool.get("function", {}).get("name") in names
        ]

    def role_expected(length: int) -> set[str]:
        return {"human", "observation"} if length % 2 == 0 else {"function_call"}

    for index, message in enumerate(messages[1:], start=1):
        role = message["role"]
        if type(message.get("trainable")) is not bool:
            raise ValueError(f"m{index}: missing boolean trainable")
        if role in {"user", "tool"}:
            if message["trainable"]:
                raise ValueError(f"m{index}: context must not be trainable")
            converted = {
                "from": "human" if role == "user" else "observation",
                "value": message["content"],
            }
        elif role == "assistant":
            calls = message.get("tool_calls", [])
            if message.get("content") or len(calls) != 1:
                raise ValueError(f"m{index}: unsupported assistant shape")
            call = calls[0]
            name, arguments = call["name"], call["arguments"]
            key = {"run_candidate": "input", "submit": "code"}.get(name)
            if key is None or not isinstance(arguments, dict):
                raise ValueError(f"m{index}: invalid tool")
            if set(arguments) != {key} or not isinstance(arguments[key], str):
                raise ValueError(f"m{index}: invalid tool arguments")
            converted = {
                "from": "function_call",
                "value": json.dumps(
                    {"name": name, "arguments": arguments}, ensure_ascii=False
                ),
            }
            if role == "assistant" and message["trainable"]:
                sample_tools = json.dumps(filtered_tools(), ensure_ascii=False)
        else:
            raise ValueError(f"unsupported role: {role}")
        if converted["from"] not in role_expected(len(history)):
            raise ValueError(f"m{index}: invalid role sequence")
        history.append(converted)
        if role == "assistant" and message["trainable"]:
            samples.append({
                "sample_id": f"{record['id']}:m{index}",
                "conversations": copy.deepcopy(history),
                "system": system,
                "tools": sample_tools,
            })
        if role == "assistant" and name == "submit":
            submitted = True
    return samples


def load_rows(episodes_path: Path, metadata_path: Path):
    episodes = [json.loads(line) for line in open(episodes_path, encoding="utf-8")]
    metadata = {str(json.loads(line)["id"]): json.loads(line)
                for line in open(metadata_path, encoding="utf-8")}
    missing = [e["id"] for e in episodes if str(e["id"]) not in metadata]
    if missing:
        raise ValueError(f"{len(missing)} episodes missing metadata, e.g. {missing[:3]}")
    for e in episodes:
        m = metadata[str(e["id"])]
        if str(m.get("episode_id")) != str(e["id"]):
            raise ValueError(f"metadata id mismatch for {e['id']}")
        e["_meta"] = m
    return episodes


def sample_meta(sample: dict, source_hash: str, record_id: str,
                meta: dict, split: str, index_in_record: int,
                source_file: str) -> dict:
    target_message_index = int(sample["sample_id"].split(":m", 1)[1])
    last = sample["conversations"][-1]
    target_tool = json.loads(last["value"])["name"]
    return {
        "sample_id": sample["sample_id"],
        "source_file_hash": source_hash,
        "source_record_id": str(record_id),
        "problem_key": str(record_id),
        "difficulty": (meta.get("difficulty") or "unknown"),
        "io_mode": (meta.get("io_mode") or "unknown"),
        "behavior": (meta.get("behavior_sequence") or ["unknown"])[0],
        "target_message_index": target_message_index,
        "target_tool": target_tool,
        "split": split,
        "targets_in_record": index_in_record + 1,
        "source_file": source_file,
    }


def write_json(out: Path, rows: list[dict]) -> None:
    out.write_text(
        json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, type=Path,
                        default=Path("data/sft_final/episodes.jsonl"))
    parser.add_argument("--metadata", required=True, type=Path,
                        default=Path("data/sft_final/metadata.jsonl"))
    parser.add_argument("--out", required=True, type=Path,
                        default=Path("data/work/exports/coding_sft_current"))
    parser.add_argument("--extra-episodes", type=Path, default=None,
                        help="optional extra trajectories (e.g. masked-submit fixture)")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-n", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_hash = "sha256:" + hashlib.sha256(
        Path(args.episodes).read_bytes()).hexdigest()

    episodes = load_rows(args.episodes, args.metadata)
    by_problem = defaultdict(list)
    for e in episodes:
        by_problem[str(e["id"])].append(e)
    problems = sorted(by_problem)
    rng = random.Random(args.seed)
    rng.shuffle(problems)
    n_dev = int(round(len(problems) * args.dev_ratio))
    dev_problems = set(problems[:n_dev])
    train_problems = set(problems[n_dev:])
    assert not (dev_problems & train_problems)

    def convert(problem_ids, split, smoke_only=False):
        samples, manifest = [], []
        for pid in problem_ids:
            for record in by_problem[pid]:
                converted = export_prefix_samples(record)
                for index, sample in enumerate(converted):
                    samples.append(sample)
                    manifest.append(sample_meta(
                        sample, source_hash, record["id"], record["_meta"],
                        split, index, str(args.episodes)))
        return samples, manifest

    train_samples, train_manifest = convert(train_problems, "train")
    dev_samples, dev_manifest = convert(dev_problems, "dev")
    print(f"problems: total={len(problems)} train={len(train_problems)} "
          f"dev={len(dev_problems)}")
    print(f"samples: train={len(train_samples)} dev={len(dev_samples)}")

    # smoke: round-robin run/submit, prefer io diversity, distinct problems
    by_tool = defaultdict(list)
    for sample, meta in zip(train_samples, train_manifest):
        by_tool[meta["target_tool"]].append((sample, meta))
    chosen = []
    seen = set()
    n_run = (args.smoke_n + 1) // 2
    n_submit = args.smoke_n - n_run
    for tool, want in (("run_candidate", n_run), ("submit", n_submit)):
        taken = 0
        io_seen = defaultdict(int)
        for sample, meta in by_tool.get(tool, []):
            if taken >= want:
                break
            if meta["sample_id"] in seen:
                continue
            if io_seen[meta["io_mode"]] >= max(1, want // 2):
                continue
            chosen.append((sample, meta))
            seen.add(meta["sample_id"])
            io_seen[meta["io_mode"]] += 1
            taken += 1
        if taken < want:
            for sample, meta in by_tool.get(tool, []):
                if taken >= want or meta["sample_id"] in seen:
                    continue
                chosen.append((sample, meta))
                seen.add(meta["sample_id"])
                taken += 1
    smoke_samples = [s for s, _ in chosen]
    smoke_manifest = [m for _, m in chosen]
    print(f"smoke: {len(smoke_samples)}")

    write_json(out_dir / "train.json", train_samples)
    write_json(out_dir / "dev.json", dev_samples)
    write_json(out_dir / "smoke.json", smoke_samples)

    dataset_info = {}
    for name in ("train", "dev", "smoke"):
        dataset_info[f"coding_agent_{name}"] = {
            "file_name": f"{name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system",
                        "tools": "tools"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt",
                "observation_tag": "observation", "function_tag": "function_call",
            },
        }
    if args.extra_episodes:
        dataset_info["coding_agent_structure"] = {
            "file_name": "structure.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system",
                        "tools": "tools"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt",
                "observation_tag": "observation", "function_tag": "function_call",
            },
        }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    split_manifest = {
        "episodes_source": str(args.episodes),
        "source_hash": source_hash,
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "total_problems": len(problems),
        "train_problem_keys": sorted(train_problems),
        "dev_problem_keys": sorted(dev_problems),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "by_problem_sample_count": {
            "max": max(Counter(meta["problem_key"] for meta in train_manifest + dev_manifest).values())
        },
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    all_manifest = train_manifest + dev_manifest + smoke_manifest
    structure_samples: list[dict] = []
    if args.extra_episodes:
        extra_rows = [json.loads(line) for line in
                      open(args.extra_episodes, encoding="utf-8")]
        for e in extra_rows:
            for sample in export_prefix_samples(e):
                structure_samples.append(sample)
                all_manifest.append({
                    "sample_id": sample["sample_id"],
                    "source_file_hash": record_hash(e),
                    "source_record_id": str(e["id"]),
                    "problem_key": str(e["id"]),
                    "difficulty": "unknown",
                    "io_mode": "unknown",
                    "behavior": "multiround_fixture",
                    "target_message_index": int(
                        sample["sample_id"].split(":m", 1)[1]),
                    "target_tool": json.loads(
                        sample["conversations"][-1]["value"])["name"],
                    "split": "structure",
                    "targets_in_record": None,
                })
        write_json(out_dir / "structure.json", structure_samples)
    with (out_dir / "sample_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for meta in all_manifest:
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")

    summary = {
        "train_problems": len(train_problems),
        "dev_problems": len(dev_problems),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "smoke_samples": len(smoke_samples),
        "structure_samples": len(structure_samples),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
