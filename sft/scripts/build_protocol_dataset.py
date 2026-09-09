#!/usr/bin/env python3
"""build_protocol_dataset.py — Minimal Protocol SFT warm-up set (target ~300).

从完整 native prefix-sample 池（prepare_sft 产物）中按“状态覆盖优先、target
短且干净、保护 base coding ability”原则筛选出 300 条 protocol warm-up 样本。

配比（允许 ±浮动，总量 280~320，run router 数 >= repair submit 数）：
  problem_submit                30
  first_failure_run            140   <- 最重要（router 池最大）
  first_failure_direct_repair   35
  post_run_repair               55
  second_failure_run            25
  multiround_final_submit       15

规则：
  * 单位 = prefix sample（prepare_sft 导出），不是 episode；
  * state_type 由 history 结构推断（先 submit 失败? run 次数? 目标工具?）；
  * SELECTION_ORDER = rare-state-first：先占 second_failure_run /
    multiround_final_submit，避免被前面 bucket 用光 problem cap（每 problem ≤2）；
  * 核心行为下限是 hard gate（RuntimeError），不满足绝不输出；
  * 难度偏向 intro/interview，competition 降权；
  * target 代码长 & 怪风格（lambda 堆叠等）降权；
  * 最近一条 observation 有诊断价值 +1 / timeout·output_limit -1 / truncated -2；
  * 输出 data/coding_sft_protocol/{train,dev}.json + dataset_info +
    manifest + stats + README；不修改完整数据池。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "data/work/pools/coding_sft_v5"
OUT = ROOT / "data/coding_sft_protocol"

QUOTAS = {
    "problem_submit": 30,
    "first_failure_run": 140,
    "first_failure_direct_repair": 35,
    "post_run_repair": 55,
    "second_failure_run": 25,
    "multiround_final_submit": 15,
}
SELECTION_ORDER = [
    "second_failure_run",
    "multiround_final_submit",
    "first_failure_direct_repair",
    "post_run_repair",
    "problem_submit",
    "first_failure_run",
]
TOTAL_TARGET = sum(QUOTAS.values())  # 300
MAX_PER_PROBLEM = 2

HARD_GATES = {
    "total_min": 280, "total_max": 320,
    "first_failure_run_min": 100,
    "second_failure_run_min": 20,
    "multiround_final_submit_min": 10,
    "run_ge_submit": True,
}

CLEAR_RUNTIME_ERRORS = (
    "indexerror", "valueerror", "typeerror", "runtimeerror",
    "attributeerror", "keyerror", "zerodivisionerror", "nameerror",
    "assertionerror", "recursionerror", "stoperror", "overflowerror",
    "syntaxerror", "importerror",
)

LAMBDA = re.compile(r"\blambda\s")


def classify(sample: dict) -> str:
    """state_type from conversation history (excluding the target turn)."""
    conv = sample["conversations"]
    history = conv[:-1]
    target = json.loads(conv[-1]["value"])
    target_name = target["name"]

    submit_failures = 0
    runs = 0
    pending = None  # last function call awaiting its observation
    for item in history:
        if item["from"] == "function_call":
            pending = json.loads(item["value"])
            if pending["name"] == "run_candidate":
                runs += 1
        elif item["from"] == "observation" and pending is not None:
            if pending["name"] == "submit":
                try:
                    obs = json.loads(item["value"])
                except json.JSONDecodeError:
                    obs = {}
                if obs.get("status") != "accepted":
                    submit_failures += 1
            pending = None

    if target_name == "submit" and submit_failures == 0:
        return "problem_submit"
    if submit_failures == 1:
        if runs == 0 and target_name == "run_candidate":
            return "first_failure_run"
        if runs == 0 and target_name == "submit":
            return "first_failure_direct_repair"
        if runs >= 1 and target_name == "submit":
            return "post_run_repair"
        if runs == 1 and target_name == "run_candidate":
            return "other"  # pathological
        return "other"
    if submit_failures >= 2:
        if target_name == "run_candidate":
            return "second_failure_run"
        if target_name == "submit" and runs >= 2:
            return "multiround_final_submit"
        return "post_run_repair"
    return "other"


def target_info(sample: dict) -> tuple[str, str, str, int]:
    """(tool, code_text, input_text, submit_code_tokens_approx)"""
    value = json.loads(sample["conversations"][-1]["value"])
    name = value["name"]
    args = value["arguments"]
    if name == "submit":
        return name, args.get("code", ""), "", 0
    return name, "", args.get("input", ""), 0


def token_estimate(text: str) -> int:
    # 粗估：用于筛选排序即可（真实 token 由 label audit 提供）
    return max(1, len(text) // 3)


def code_style_penalty(code: str) -> int:
    penalty = 0
    if len(LAMBDA.findall(code)) >= 3:
        penalty += 2
    if len(code) < 80:  # likely golfed one-liner
        penalty += 1
    if "\\n" in code or code.count("\n") < 2:
        penalty += 1
    return penalty


def last_observation(sample: dict) -> dict:
    """Most recent observation preceding the target (empty dict if none)."""
    if len(sample["conversations"]) >= 2:
        prev = sample["conversations"][-2]
        if prev["from"] == "observation":
            try:
                obs = json.loads(prev["value"])
                return obs if isinstance(obs, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def obs_diagnostic_score(obs: dict) -> int:
    """Execution feedback value: clean ok / crisp runtime error +1;
    timeout/output_limit -1; truncated -2."""
    if not obs:
        return 0
    status = str(obs.get("status", "")).lower()
    if obs.get("truncated") is True:
        return -2
    if status in ("ok",):
        return 1 if not obs.get("truncated") else -2
    if status in ("timeout", "output_limit", "output_limit_exceeded"):
        return -1
    if status == "runtime_error" or obs.get("exit_code") not in (None, 0):
        error = str(obs.get("error") or obs.get("exception") or "").lower()
        stderr = str(obs.get("stderr") or "")
        if any(e in error or e in stderr.lower() for e in CLEAR_RUNTIME_ERRORS):
            return 1 if len(stderr) < 2000 else 0
        return 0
    return 0  # wrong_answer 等：有 failing_input 但无运行诊断，中性


def main() -> int:
    train_rows = json.loads((POOL_DIR / "train.json").read_text(encoding="utf-8"))
    manifest_index = {}
    for line in open(POOL_DIR / "sample_manifest.jsonl", encoding="utf-8"):
        row = json.loads(line)
        manifest_index[row["sample_id"]] = row

    entries = []
    for sample in train_rows:
        sample_id = sample["sample_id"]
        meta = manifest_index.get(sample_id)
        if meta is None:
            continue
        state = classify(sample)
        tool, code, probe, _ = target_info(sample)
        code_tokens = token_estimate(code)
        difficulty = meta.get("difficulty", "unknown")
        behavior = meta.get("behavior", "unknown")

        score = 0.0
        if tool == "run_candidate":
            score += 3
        if state == "second_failure_run":
            score += 2
        if state == "problem_submit":
            score += 1
        if code_tokens < 500:
            score += 1
        if code_tokens > 1000:
            score -= 3
        if difficulty == "competition":
            score -= 1
        if state == "problem_submit":
            score -= code_style_penalty(code)
        if state != "problem_submit":
            score += obs_diagnostic_score(last_observation(sample))
        entries.append({
            "sample": sample,
            "sample_id": sample_id,
            "problem_id": meta.get("problem_key") or meta.get("source_record_id"),
            "state_type": state,
            "target_tool": tool,
            "behavior": behavior,
            "difficulty": difficulty,
            "submit_code_tokens_est": code_tokens,
            "score": score,
        })

    by_state = Counter(e["state_type"] for e in entries)
    print("pool by state_type:", dict(by_state), flush=True)

    used_problem: Counter[str] = Counter()
    selected: list[dict] = []
    shortfall = []

    def take(e):
        pid = str(e["problem_id"])
        if used_problem[pid] >= MAX_PER_PROBLEM:
            return False
        used_problem[pid] += 1
        selected.append(e)
        return True

    # rare-state-first: multi-round samples are the scarcest and most valuable;
    # take them before abundant buckets exhaust a problem's 2-sample cap.
    for state in SELECTION_ORDER:
        want = QUOTAS[state]
        bucket = [e for e in entries if e["state_type"] == state]
        bucket.sort(key=lambda e: -e["score"])
        got = 0
        for e in bucket:
            if got >= want:
                break
            if take(e):
                got += 1
        if got < want:
            shortfall.append({"state_type": state, "want": want, "got": got,
                              "available": len(bucket)})

    run_total = sum(
        1 for e in selected if e["target_tool"] == "run_candidate")
    submit_total = sum(
        1 for e in selected if e["target_tool"] == "submit")
    n = len(selected)
    state_count = Counter(e["state_type"] for e in selected)

    # Hard gates: quota may drift, core behavior floors must hold. 失败即中止，
    # 绝不带着低于下限的集合进入下一步训练。
    errors = []
    if not HARD_GATES["total_min"] <= n <= HARD_GATES["total_max"]:
        errors.append(f"total {n} outside "
                      f"[{HARD_GATES['total_min']}, {HARD_GATES['total_max']}]")
    if HARD_GATES["run_ge_submit"] and run_total < submit_total:
        errors.append(f"run {run_total} < submit {submit_total}")
    if state_count["first_failure_run"] < HARD_GATES["first_failure_run_min"]:
        errors.append(f"first_failure_run {state_count['first_failure_run']} < "
                      f"{HARD_GATES['first_failure_run_min']}")
    if state_count["second_failure_run"] < HARD_GATES["second_failure_run_min"]:
        errors.append(f"second_failure_run {state_count['second_failure_run']} < "
                      f"{HARD_GATES['second_failure_run_min']}")
    if state_count["multiround_final_submit"] < HARD_GATES["multiround_final_submit_min"]:
        errors.append(
            f"multiround_final_submit {state_count['multiround_final_submit']} < "
            f"{HARD_GATES['multiround_final_submit_min']}")
    if errors:
        raise RuntimeError("hard gate failed: " + "; ".join(errors))
    if shortfall:
        print("NOTE quota shortfall (non-blocking, floors still held):",
              shortfall, flush=True)

    # build dev: stratified sanity split — 每类 state 4 条（24 total），problem 与
    # train 及 dev 内部均不相交。只用于早期 checkpoint 的 protocol-loss 观察。
    DEV_QUOTAS = {state: 4 for state in QUOTAS}
    train_problems = {str(e["problem_id"]) for e in selected}
    dev_entries = []
    dev_problems = set()
    for state in SELECTION_ORDER:
        want = DEV_QUOTAS[state]
        bucket = sorted((e for e in entries if e["state_type"] == state),
                        key=lambda e: -e["score"])
        got = 0
        for e in bucket:
            if got >= want:
                break
            pid = str(e["problem_id"])
            if pid in train_problems or pid in dev_problems:
                continue
            dev_problems.add(pid)
            dev_entries.append(e)
            got += 1
    if any(len([1 for e in dev_entries if e["state_type"] == s]) < 2
           for s in QUOTAS):
        raise RuntimeError(
            "dev stratified selection under-supplied a state; "
            "pool lacks problem-disjoint dev candidates")
    if len(dev_entries) != sum(DEV_QUOTAS.values()):
        raise RuntimeError(
            f"dev expected {sum(DEV_QUOTAS.values())} got {len(dev_entries)}")

    OUT.mkdir(parents=True, exist_ok=True)
    train_json = [e["sample"] for e in selected]
    dev_json = [e["sample"] for e in dev_entries]
    (OUT / "train.json").write_text(json.dumps(train_json, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    (OUT / "dev.json").write_text(json.dumps(dev_json, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    dataset_info = {}
    for name in ("train", "dev"):
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
    (OUT / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    manifest = []
    for e in selected:
        meta = manifest_index.get(e["sample_id"], {})
        manifest.append({
            "sample_id": e["sample_id"],
            "problem_id": e["problem_id"],
            "state_type": e["state_type"],
            "target_tool": e["target_tool"],
            "behavior": e["behavior"],
            "difficulty": e["difficulty"],
            "target_message_index": meta.get("target_message_index"),
            "submit_code_tokens_est": e["submit_code_tokens_est"],
            "selection_score": e["score"],
            "reason_selected": "bucket top-score within per-problem cap 2",
            "source_episode_id": e["sample_id"].rsplit(":m", 1)[0],
        })
    with (OUT / "protocol_warmup_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in manifest:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_problem = Counter(str(m["problem_id"]) for m in manifest)
    stats = {
        "total_samples": n,
        "distinct_problems": len(per_problem),
        "by_state_type": dict(Counter(m["state_type"] for m in manifest)),
        "by_target_tool": dict(Counter(m["target_tool"] for m in manifest)),
        "by_difficulty": dict(Counter(m["difficulty"] for m in manifest)),
        "by_behavior": dict(Counter(m["behavior"] for m in manifest)),
        "run_candidate_sample_count": run_total,
        "submit_sample_count": submit_total,
        "submit_code_tokens_est_total": sum(
            m["submit_code_tokens_est"] for m in manifest
            if m["target_tool"] == "submit"),
        "problems_with_1_sample": sum(
            1 for v in per_problem.values() if v == 1),
        "problems_with_2_samples": sum(
            1 for v in per_problem.values() if v == 2),
        "quota_shortfalls": shortfall,
        "dev_samples": len(dev_entries),
    }
    (OUT / "protocol_warmup_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "\n".join([
            "# Minimal Protocol SFT warm-up dataset",
            "",
            "- 单位：prefix sample（prepare_sft 导出）。目标：让 Qwen2.5-Coder-7B",
            "  稳定学会工具协议后进入 RL，不追求提升 APPS solve rate。",
            f"- 样本：train {n} / dev {len(dev_entries)}；同 problem 至多 2 条。",
            "- 结构：ShareGPT（conversations/system/tools），tools 按 candidate",
            "  是否已存在动态提供（首决策仅 submit）。",
            f"- 配比：{QUOTAS}",
            f"- 状态：{json.dumps(Counter(m['state_type'] for m in manifest))}",
            "- 统计见 protocol_warmup_stats.json；token 级审计用 LLaMA-Factory 真实",
            "  template 复核（mask_history=true），详见 README 顶层流程。",
        ]) + "\n", encoding="utf-8")

    print(json.dumps({
        "train": n, "dev": len(dev_entries), "problems": len(per_problem),
        "by_state_type": dict(Counter(m["state_type"] for m in manifest)),
        "run_submit": {"run": run_total, "submit": submit_total},
        "by_difficulty": stats["by_difficulty"],
        "shortfalls": shortfall,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
