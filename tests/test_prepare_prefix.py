"""Unit tests for prepare_sft.export_prefix_samples (native V4 histories).

Covers:
  - C     problem -> submit(correct)[T]          tools = [submit]
  - B1    problem -> submit(bad)[F] -> fail -> run[T] -> obs -> submit[T]
          router sample ends at run with tools = [submit, run_candidate];
          repair sample ends at submit with tools = [submit, run_candidate]
  - multiround emits one sample per trainable action, history intact.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_sft", ROOT / "sft" / "scripts" / "prepare_sft.py")
prepare_sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_sft)

TOOLS = [
    {"type": "function", "function": {"name": "run_candidate",
                                      "description": "run", "parameters": {}}},
    {"type": "function", "function": {"name": "submit",
                                      "description": "submit", "parameters": {}}},
]


def names(sample):
    return [json.loads(c["value"])["name"] for c in sample["conversations"]
            if c["from"] == "function_call"]


def tool_names(sample):
    return sorted(t["function"]["name"] for t in json.loads(sample["tools"]))


def msg(role, name=None, code=None, trainable=False, content=None):
    base = {"role": role, "trainable": trainable}
    if role == "assistant":
        if name == "run_candidate":
            arguments = {"input": code}
        else:
            arguments = {"code": code}
        base["tool_calls"] = [{"name": name, "arguments": arguments}]
    elif role == "tool":
        base["name"] = name
        base["content"] = content or "{}"
    else:
        base["content"] = content or "problem"
    return base


def test_c_sample_only_submit_tool():
    record = {
        "id": "c1", "tools": TOOLS,
        "messages": [
            {"role": "system", "content": "sys", "trainable": False},
            {"role": "user", "content": "Problem", "trainable": False},
            msg("assistant", "submit", "correct", True),
            msg("tool", "submit", content='{"status":"accepted"}', trainable=False),
        ],
    }
    samples = prepare_sft.export_prefix_samples(record)
    assert len(samples) == 1
    assert names(samples[0]) == ["submit"]
    assert tool_names(samples[0]) == ["submit"]


def test_b1_router_and_repair_samples():
    record = {
        "id": "b1", "tools": TOOLS,
        "messages": [
            {"role": "system", "content": "sys", "trainable": False},
            {"role": "user", "content": "Problem", "trainable": False},
            msg("assistant", "submit", "bad", False),
            msg("tool", "submit", content='{"status":"wrong_answer"}', trainable=False),
            msg("assistant", "run_candidate", "fail_case", True),
            msg("tool", "run_candidate", content='{"status":"ok"}', trainable=False),
            msg("assistant", "submit", "good", True),
            msg("tool", "submit", content='{"status":"accepted"}', trainable=False),
        ],
    }
    samples = prepare_sft.export_prefix_samples(record)
    assert len(samples) == 2
    router, repair = samples
    assert names(router) == ["submit", "run_candidate"]
    assert names(repair) == ["submit", "run_candidate", "submit"]
    # router target must be present as the last conversation item
    assert json.loads(router["conversations"][-1]["value"])["name"] == \
        "run_candidate"
    assert json.loads(repair["conversations"][-1]["value"])["name"] == "submit"
    assert tool_names(router) == ["run_candidate", "submit"]
    assert tool_names(repair) == ["run_candidate", "submit"]
