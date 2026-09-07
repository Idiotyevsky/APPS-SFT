import copy
import json

import pytest

from synthesis.agent_eval import (
    ActionError, AgentEnv, EvalConfig, ProgressStore, action_json_schema,
    digest, evaluate_problem, executable_action, select_rows, strict_action,
    summarize,
)


ROW = {"problem_id": "1", "difficulty": "introductory", "io_mode": "stdin",
       "question": "Echo one integer", "starter_code": "",
       "inputs": ["1\n", "2\n"], "outputs": ["1\n", "2\n"]}
GOOD = "print(input())\n"
BAD = "print(0)\n"


def action(name="submit", value=GOOD):
    return {"name": name, "arguments": {"code" if name == "submit" else "input": value}}


class Tokenizer:
    def __init__(self, length=100):
        self.length = length
        self.histories = []
        self.toolsets = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] and kwargs["add_generation_prompt"]
        self.histories.append(copy.deepcopy(messages))
        self.toolsets.append([tool["function"]["name"] for tool in kwargs["tools"]])
        return list(range(self.length + 10 * (len(messages)-2)))


class Generator:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.budgets = []
        self.allowed = []

    def __call__(self, ids, budget, allowed_names):
        self.budgets.append(budget)
        self.allowed.append(tuple(allowed_names))
        value = next(self.actions)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, str):
            return {"text": value, "token_count": 10, "finish_reason": "stop"}
        return {"text": json.dumps(value), "token_count": 10, "finish_reason": "stop"}


class Env:
    def __init__(self, row, candidate=None):
        self.current_candidate = candidate

    def submit(self, code):
        self.current_candidate = code
        good = code == GOOD
        return {"status": "accepted" if good else "wrong_answer", "passed": 2 if good else 0,
                "total": 2, "pass_rate": 1.0 if good else 0.0, "failing_input": None if good else "1\n"}

    def run(self, input_text):
        if self.current_candidate is None:
            return {"status": "invalid_input", "error": {"type": "NoCandidate"}}
        return {"status": "ok", "stdout": "0\n", "error": None}


def test_action_constraint_has_exact_tool_union():
    schema = action_json_schema()
    assert [b["properties"]["name"]["const"] for b in schema["oneOf"]] == [
        "run_candidate", "submit"]
    assert schema["oneOf"][0]["properties"]["arguments"]["required"] == ["input"]
    assert schema["oneOf"][1]["properties"]["arguments"]["required"] == ["code"]
    only_submit = action_json_schema(("submit",))
    assert len(only_submit["oneOf"]) == 1
    assert only_submit["oneOf"][0]["properties"]["name"]["const"] == "submit"
    with pytest.raises(ValueError):
        action_json_schema(("unknown",))


@pytest.mark.parametrize("wrap", [False, True])
def test_strict_single_call(wrap):
    text = json.dumps(action())
    if wrap:
        text = "<tool_call>\n" + text + "\n</tool_call>"
    assert strict_action(text) == action()


@pytest.mark.parametrize("text", [
    '<tool_call>{"name":"submit","arguments":{"code":"a"}}</tool_call>' * 2,
    json.dumps(action()) + json.dumps(action()),
    'explanation ' + json.dumps(action()),
    '{"name":"submit","name":"run_candidate","arguments":{"input":"1"}}',
    json.dumps({"name": "run_candidate", "arguments": {"input": "1", "code": "bad"}}),
    json.dumps({"name": [], "arguments": {}}),
    json.dumps({"name": "submit", "arguments": {"code": " "}}),
])
def test_reject_extra_or_malformed_calls(text):
    with pytest.raises(ActionError):
        strict_action(text)


def test_tag_text_in_code_is_not_a_second_action():
    value = action(value='print("<tool_call></tool_call>")')
    assert strict_action('<tool_call>'+json.dumps(value)+'</tool_call>') == value


def test_response_wrapper_is_executable_but_not_strict():
    text = "<response>\n" + json.dumps(action()) + "\n</response><|im_end|>"
    with pytest.raises(ActionError):
        strict_action(text)
    assert executable_action(text) == (action(), "response_wrapper")


def test_json_fence_is_executable_but_not_strict():
    text = "```json\n" + json.dumps(action()) + "\n```"
    with pytest.raises(ActionError):
        strict_action(text)
    assert executable_action(text) == (action(), "json_fence")


def test_response_wrapper_executes_without_inflating_format_metric():
    text = "<response>" + json.dumps(action()) + "</response>"
    result = evaluate_problem(ROW, Tokenizer(), Generator([text]), EvalConfig(), env_factory=Env)
    assert result["solved"] and result["executable_actions"] == 1
    assert result["valid_actions"] == 0
    summary = summarize([result], 1)
    assert summary["tool_action_parse_rate"] == 1.0
    assert summary["tool_format_valid_rate"] == 0.0


def test_dynamic_full_context_budget_and_exact_history():
    tok = Tokenizer(); gen = Generator([action(value=BAD), action("run_candidate", "1\n"), action()])
    r = evaluate_problem(ROW, tok, gen, EvalConfig(), env_factory=Env)
    assert r["solved"] and r["submissions"] == 2 and r["runs"] == 1
    assert r["repair_eligible"] and r["first_submit_success"] is False
    assert gen.budgets == [32768-100, 32768-120, 32768-140]
    assert gen.allowed == [("submit",), ("run_candidate", "submit"),
                           ("run_candidate", "submit")]
    assert tok.toolsets[0] == ["submit"]
    assert tok.toolsets[1] == ["run_candidate", "submit"]
    last = tok.histories[-1]
    assert last[2]["tool_calls"][0]["function"] == action(value=BAD)
    assert json.loads(last[3]["content"])["status"] == "wrong_answer"
    assert last[4]["tool_calls"][0]["function"]["name"] == "run_candidate"
    assert json.loads(last[5]["content"])["stdout"] == "0\n"
    assert tok.histories[0][0]["role"] == "system"
    assert tok.histories[0][1]["role"] == "user"
    # problem-only user text must match training (build_problem_only_state), no extra hint
    assert "Problem:" in tok.histories[0][1]["content"]


def test_truncation_is_not_parsing_failure_or_executed_submit():
    def gen(ids,budget,allowed_names):
        return {"text":json.dumps(action()), "token_count":budget, "finish_reason":"length"}
    r=evaluate_problem(ROW,Tokenizer(),gen,EvalConfig(),env_factory=Env)
    assert r["status"] == "truncated" and r["submissions"] == 0


def test_context_exhaustion_no_generation_or_history_truncation():
    gen=Generator([])
    r=evaluate_problem(ROW,Tokenizer(32768),gen,EvalConfig(),env_factory=Env)
    assert r["status"] == "context_exhausted" and gen.budgets == []


def test_generation_and_independent_tool_budgets():
    r=evaluate_problem(ROW,Tokenizer(),Generator([action(value=BAD),action()]),
                       EvalConfig(max_submits=1),env_factory=Env)
    assert r["status"] == "submit_budget" and r["submissions"] == 1
    r=evaluate_problem(ROW,Tokenizer(),Generator([action("run_candidate","1")]),
                       EvalConfig(mode="candidate",max_runs=0),BAD,env_factory=Env)
    assert r["status"] == "run_budget" and r["runs"] == 0
    gen=Generator([action(value=BAD)])
    r=evaluate_problem(ROW,Tokenizer(),gen,EvalConfig(max_total_new_tokens=10),env_factory=Env)
    assert r["status"] == "generation_budget" and gen.budgets == [10]
    assert EvalConfig(max_new_tokens=5000).available_tokens(100,0)==5000


@pytest.mark.parametrize("mode", ["candidate","repair"])
def test_fixed_candidate_setup_does_not_spend_submit_budget(mode):
    tok=Tokenizer();gen=Generator([action("run_candidate","1\n"),action()])
    r=evaluate_problem(ROW,tok,gen,EvalConfig(mode=mode),BAD,env_factory=Env)
    assert r["solved"] and r["runs"] == 1 and r["submissions"] == 1
    user=tok.histories[0][1]["content"]
    assert BAD in user
    assert ("Previous submit result" in user) == (mode == "repair")
    assert "expected" not in user
    r=evaluate_problem(ROW,Tokenizer(),Generator([]),EvalConfig(mode=mode),GOOD,env_factory=Env)
    assert r["status"] == "candidate_not_wrong"


def test_unavailable_tool_and_backend_error_are_distinct():
    r=evaluate_problem(ROW,Tokenizer(),Generator([action("run_candidate","1")]),
                       EvalConfig(),env_factory=Env)
    assert r["status"] == "invalid_action" and r["no_candidate_calls"] == 0
    assert r["turn_history"][0]["parse_error"] == "unavailable_tool"
    failed=evaluate_problem(ROW,Tokenizer(),Generator([RuntimeError("backend unavailable")]),
                            EvalConfig(),env_factory=Env)
    assert failed["status"] == "infrastructure_error"
    summary=summarize([r,failed],2)
    assert summary["budgeted_agent_success_rate"] == 0.0
    assert summary["terminal_status"]["infrastructure_error"] == 1


def test_resume_rejects_config_changes_and_recovers_partial_tail(tmp_path):
    manifest={"model":"hash1","selected_ids":["1","2"]}
    store=ProgressStore(tmp_path,manifest)
    store.append({"id":"1","solved":True})
    with store.path.open("ab") as f:f.write(b'{"id":')
    resumed=ProgressStore(tmp_path,manifest,resume=True)
    assert set(resumed.records)=={"1"}
    resumed.append({"id":"2","solved":False})
    assert len(ProgressStore(tmp_path,manifest,resume=True).records)==2
    with pytest.raises(ValueError,match="fingerprint mismatch"):
        ProgressStore(tmp_path,dict(manifest,model="changed"),resume=True)
    with pytest.raises(FileExistsError):ProgressStore(tmp_path,manifest)


def test_resume_rejects_complete_corruption(tmp_path):
    store=ProgressStore(tmp_path,{"model":"hash"})
    store.path.write_text('{oops}\n')
    with pytest.raises(ValueError,match="corrupt"):
        ProgressStore(tmp_path,{"model":"hash"},True)


def test_balanced_selection_is_deterministic():
    rows=[dict(ROW,problem_id=str(i),difficulty=d) for i,d in enumerate(
        ["interview"]*3+["introductory"]*3+["competition"]*3)]
    selected=select_rows(rows,per_difficulty=2)
    assert len(selected)==6 and selected==select_rows(rows,per_difficulty=2)
    with pytest.raises(ValueError):select_rows(rows,limit=3,per_difficulty=1)
    with pytest.raises(ValueError):select_rows(rows+[rows[0]])


def test_real_grader_two_test_units_and_candidate_run():
    env=AgentEnv(ROW)
    assert env.run("1\n")["error"]["type"]=="NoCandidate"
    bad=env.submit(BAD)
    assert bad["total"]==2 and bad["passed"]==0 and bad["failing_input"]=="1\n"
    assert env.run("2\n")["stdout"].strip()=="0"
    good=env.submit(GOOD)
    assert good["status"]=="accepted" and good["passed"]==good["total"]==2
    assert set(good)=={"status","passed","total","pass_rate","failing_input"}


def test_compile_feedback_and_duplicate_detection():
    broken = "print(input()))\n"
    feedback = AgentEnv(ROW).submit(broken)
    assert feedback["status"] == "compile_error"
    assert feedback["error"]["type"] == "SyntaxError"
    assert feedback["error"]["line"] == 1
    gen = Generator([action(value=broken), action(value=broken), action()])
    result = evaluate_problem(ROW, Tokenizer(), gen, EvalConfig(), env_factory=AgentEnv)
    assert result["solved"] and result["duplicate_submissions"] == 1
    duplicate = result["turn_history"][1]["observation"]
    assert duplicate["duplicate"] is True
    assert "already submitted" in duplicate["message"]


def test_real_call_mode_and_feedback_loop():
    row=dict(ROW,io_mode="call",fn_name="double",inputs=[[1],[3]],outputs=[2,6])
    good="def double(x):\n    return x*2\n"
    gen=Generator([action(value="def double(x):\n    return x\n"),action("run_candidate","[3]"),action(value=good)])
    r=evaluate_problem(row,Tokenizer(),gen,EvalConfig())
    assert r["solved"] and r["submissions"]==2 and r["runs"]==1
    assert r["turn_history"][1]["observation"]["stdout"].strip()=="3"
