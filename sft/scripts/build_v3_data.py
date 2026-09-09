#!/usr/bin/env python3
"""build_v3_data.py — 把 SFT 失败态重序列化为真实 assistant→tool history。

背景：V1/V2 数据把“候选+失败反馈”塞进 user 文本；推理时模型看到的是
assistant submit → tool failure 的真实多轮历史。二者结构错位 → 重序列化。

规则（按 V3 设计）：
  - C   (direct_submission):                problem -> submit(correct) [T]
  - A   (post_submit_direct_repair):        problem -> submit(bad)[F] -> tool fail
                                             -> submit(repair)[T] -> accepted
  - B1  (post_submit_failure_replay):       上述 A 之后插入 run_candidate(failing)[T]
                                             -> tool run(原观测) -> submit(repair)[T]
  - B2  (pre_submit_active_validation):     本版剔除（无 external candidate 机制）
坏 submit 只作为不可训练上下文（trainable=false）。

信息全部取自原 episodes 文本（candidate / Previous submit result JSON / 原 run
观测），不重新执行 grader，保证确定性与原始 truth 一致。

用法：
  PYTHONPATH=src python3 sft/scripts/build_v3_data.py
产出：
  data/work/legacy_builds/coding_sft_v3_episodes/{episodes,metadata}.jsonl
  （供 prepare_sft 复用）
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_EP = ROOT / "data/sft_final/episodes.jsonl"
SRC_META = ROOT / "data/sft_final/metadata.jsonl"
OUT_DIR = ROOT / "data/work/legacy_builds/coding_sft_v3_episodes"

CAND = re.compile(r"Current candidate:\n```python\n(.*?)\n```", re.DOTALL)
FEEDBACK = re.compile(r"Previous submit result:\n(\{.*\})", re.DOTALL)

BEHAVIOR_C = "direct_submission"
BEHAVIOR_A = "post_submit_direct_repair"
BEHAVIOR_B1 = "post_submit_failure_replay"
BEHAVIOR_B2 = "pre_submit_active_validation"


def split_problem_and_candidate(user: str):
    problem = user.split("\n\nCurrent candidate:", 1)[0]
    candidate = None
    match = CAND.search(user)
    if match:
        candidate = match.group(1) + "\n"
    fb = FEEDBACK.search(user)
    feedback = None
    if fb:
        try:
            feedback = json.loads(fb.group(1))
        except json.JSONDecodeError:
            feedback = None
    return problem, candidate, feedback


def tool_msg(name, arguments, trainable):
    return {"role": "assistant", "tool_calls": [{"name": name,
                                                 "arguments": arguments}],
            "trainable": trainable}


def rebuild(record: dict):
    behavior = record["metadata"]["behavior_sequence"][0]
    if behavior == BEHAVIOR_B2:
        return None
    msgs = record["messages"]
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    if behavior == BEHAVIOR_C:
        problem = user  # problem-only state already
        new = [{"role": "system", "content": system, "trainable": False},
               {"role": "user", "content": problem, "trainable": False}]
        for message in msgs[2:]:
            new.append({k: v for k, v in message.items()})
        return {"id": record["id"], "tools": record["tools"], "messages": new}

    problem, candidate, feedback = split_problem_and_candidate(user)
    if candidate is None or feedback is None:
        raise ValueError(f"{record['id']}: cannot parse candidate/feedback")
    new = [{"role": "system", "content": system, "trainable": False},
           {"role": "user", "content": problem, "trainable": False}]
    # failed submit as untrainable context
    new.append(tool_msg("submit", {"code": candidate}, False))
    new.append({"role": "tool", "name": "submit",
                "content": json.dumps(feedback, ensure_ascii=False,
                                      separators=(",", ":")),
                "trainable": False})
    index = 2
    if behavior == BEHAVIOR_B1:
        run_msg = msgs[index]
        run_tool = msgs[index + 1]
        index += 2
        # run_candidate trainable (kept as trainable=true target/context by
        # prefix export; in the later prefix sample it appears as context)
        new.append(tool_msg("run_candidate", run_msg["tool_calls"][0]["arguments"], True))
        new.append({"role": "tool", "name": "run_candidate",
                    "content": run_tool["content"], "trainable": False})
    # final submit trainable + accepted observation
    final = msgs[index]
    final_tool = msgs[index + 1]
    if final["tool_calls"][0]["name"] != "submit":
        raise ValueError(f"{record['id']}: unexpected final message")
    new.append(tool_msg("submit", final["tool_calls"][0]["arguments"], True))
    new.append({"role": "tool", "name": "submit",
                "content": final_tool["content"], "trainable": False})
    return {"id": record["id"], "tools": record["tools"], "messages": new}


def main() -> int:
    import collections
    episodes = [json.loads(l) for l in open(SRC_EP, encoding="utf-8")]
    meta = {str(json.loads(l)["id"]): json.loads(l)
            for l in open(SRC_META, encoding="utf-8")}
    kept, dropped = [], []
    counts = collections.Counter()
    for record in episodes:
        behavior = record["metadata"]["behavior_sequence"][0]
        rebuilt = rebuild(record)
        if rebuilt is None:
            dropped.append(record["id"])
            counts["B2_dropped"] += 1
            continue
        kept.append(rebuilt)
        counts[behavior] += 1
    if not kept:
        raise SystemExit("no episodes kept")
    # attach updated metadata copy (behavior unchanged)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for record in kept:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (OUT_DIR / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for record in kept:
            row = dict(meta[str(record["id"])])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "kept": len(kept),
        "by_behavior": dict(counts),
        "dropped_ids": dropped,
        "out": str(OUT_DIR),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
