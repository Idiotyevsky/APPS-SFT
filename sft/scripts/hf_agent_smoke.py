#!/usr/bin/env python3
"""HF-backend agent smoke: 3 problems (one per difficulty), flush-per-problem logs.

Deterministic greedy; real sandbox grader; renders prompts via the SAME
prefix as training (verified token-identical). Use when vLLM is unreliable.
"""
import json, sys, time, collections, signal, faulthandler
faulthandler.register(signal.SIGUSR1, all_threads=True)
sys.path.insert(0, "src")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from synthesis.tools import tool_schemas
from synthesis.agent_eval import AgentEnv, build_initial_messages, strict_action, ActionError

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sft/exports/sft_exp2_merged"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/eval/results/agent/sft_exp2_hf3.json"

allrows = [json.loads(l) for l in open("data/eval/toolapps_eval.jsonl")]
rows = []
for diff in ("introductory", "interview", "competition"):
    rows.append(next(r for r in allrows if r["difficulty"] == diff))
print("problems", [(r["problem_id"], r["difficulty"]) for r in rows], flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
print("tokenizer ok", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0",
                                             attn_implementation="eager")
model.eval()
print("model loaded", flush=True)
tools = tool_schemas()


def gen(messages, max_new=700):
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)


res = []
t0 = time.monotonic()
for row in rows:
    msgs = build_initial_messages(row, mode="problem-only")
    env = AgentEnv(row)
    solved = False
    why = ""
    submits = runs = turns = 0
    for _ in range(4):
        print(f"  [{row['problem_id']}] turn {_ + 1}: before_gen", flush=True)
        raw = gen(msgs)
        print(f"  [{row['problem_id']}] turn {_ + 1}: after_gen len={len(raw)} "
              f"head={raw[:60]!r}", flush=True)
        try:
            action = strict_action(raw)
        except ActionError as exc:
            why = f"invalid:{exc}"
            print(f"  [{row['problem_id']}] parse fail: {exc}", flush=True)
            break
        turns += 1
        if action["name"] == "submit":
            submits += 1
            print(f"  [{row['problem_id']}] submit -> grading", flush=True)
            obs = env.submit(action["arguments"]["code"])
            print(f"  [{row['problem_id']}] submit obs status={obs.get('status')} "
                  f"rate={obs.get('pass_rate')}", flush=True)
            msgs.append({"role": "assistant", "content": raw})
            msgs.append({"role": "tool", "name": "submit",
                         "content": json.dumps(obs, ensure_ascii=False)})
            if obs["status"] == "accepted":
                solved = True
                break
            if submits >= 2:
                why = "submit_budget"
                break
        else:
            runs += 1
            obs = env.run(action["arguments"]["input"])
            msgs.append({"role": "assistant", "content": raw})
            msgs.append({"role": "tool", "name": "run_candidate",
                         "content": json.dumps(obs, ensure_ascii=False)})
    rec = {"problem_id": row["problem_id"], "difficulty": row["difficulty"],
           "solved": solved, "why": why, "submits": submits, "runs": runs,
           "turns": turns}
    res.append(rec)
    print(row["problem_id"], row["difficulty"],
          "solved" if solved else why,
          f"submits={submits} runs={runs} turns={turns}", flush=True)

print("TOTAL solved", sum(r["solved"] for r in res),
      "wall", round(time.monotonic() - t0, 1), flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
print("saved", OUT, flush=True)
