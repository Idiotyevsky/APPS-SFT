#!/usr/bin/env python3
"""Single-problem agent probe (HF eager) with step logs. Usage: probe_one.py <problem_id>"""
import json, sys, time
sys.path.insert(0, "src")
print("import libs", flush=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from synthesis.agent_eval import AgentEnv, build_initial_messages, strict_action, ActionError
from synthesis.tools import tool_schemas

MODEL = "sft/exports/sft_exp2_merged"
PID = sys.argv[1] if len(sys.argv) > 1 else "4004"

m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                         device_map="cuda:0",
                                         attn_implementation="eager")
print("loaded", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
row = next(json.loads(l) for l in open("data/eval/toolapps_eval.jsonl")
           if str(json.loads(l)["problem_id"]) == PID)
print("row", PID, row["difficulty"], "qchars", len(row["question"]), flush=True)
msgs = build_initial_messages(row, mode="problem-only")
text = tok.apply_chat_template(msgs, tools=tool_schemas(), tokenize=False,
                               add_generation_prompt=True)
e = tok(text, return_tensors="pt").to("cuda:0")
m.eval()
t = time.monotonic()
print("gen start", flush=True)
with torch.no_grad():
    o = m.generate(**e, do_sample=False, max_new_tokens=700,
                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
raw = tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=False)
print("A gen done", round(time.monotonic() - t, 1), "s len", len(raw), flush=True)
if __import__("os").environ.get("NOPARSE") == "1":
    with open("sft/outputs/probe_raw_%s.txt" % PID, "w", encoding="utf-8") as fh:
        fh.write(raw)
    print("NOPARSE saved", len(raw), flush=True)
    sys.exit(0)
print("A1 pre-parse", flush=True)
try:
    act = strict_action(raw)
    print("B parse ok", act["name"], flush=True)
except ActionError as exc:
    print("B parse fail", exc, flush=True)
    sys.exit(0)
print("C before submit", flush=True)
env = AgentEnv(row)
obs = env.submit(act["arguments"]["code"])
print("D submit status", obs.get("status"), obs.get("pass_rate"), flush=True)
