"""Budgeted tool evaluation; model-independent and testable without a GPU."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

from .feedback_projection import project_submit_feedback
from .grader import PrivateGrader
from .io_utils import atomic_write_json, atomic_write_jsonl
from .prompting.state_builder import (
    build_problem_only_state, build_pre_submit_state, build_post_submit_state,
)
from .sandbox import SandboxConfig, SandboxedExecutor
from .schemas import PublicProblem
from .tools import tool_schemas, validate_arguments

PROTOCOL_VERSION = "tool-agent-v5"
MODES = ("problem-only", "candidate", "repair")


def action_json_schema(allowed_names=None):
    """Exact action union used by constrained, state-aware decoding."""
    all_names = ("run_candidate", "submit")
    allowed = tuple(allowed_names or all_names)
    if not allowed or any(name not in all_names for name in allowed) or len(set(allowed)) != len(allowed):
        raise ValueError("invalid allowed action set")

    def branch(name, argument):
        return {
            "type": "object",
            "properties": {
                "name": {"const": name},
                "arguments": {
                    "type": "object",
                    "properties": {argument: {"type": "string"}},
                    "required": [argument],
                    "additionalProperties": False,
                },
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
    arguments = {"run_candidate": "input", "submit": "code"}
    return {"oneOf": [branch(name, arguments[name]) for name in allowed]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def problem_id(row):
    value = row.get("id", row.get("problem_id"))
    if value is None or isinstance(value, bool):
        raise ValueError("missing problem id")
    return str(value)


def select_rows(rows, limit=0, per_difficulty=0):
    if limit < 0 or per_difficulty < 0 or (limit and per_difficulty):
        raise ValueError("use either --limit or --per-difficulty, both nonnegative")
    ids = [problem_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate eval problem ids")
    if per_difficulty:
        selected = []
        for difficulty in ("introductory", "interview", "competition"):
            bucket = [row for row in rows if row["difficulty"] == difficulty]
            if len(bucket) < per_difficulty:
                raise ValueError(f"not enough {difficulty} problems")
            selected.extend(bucket[:per_difficulty])
        return selected
    return rows[:limit] if limit else rows


class ActionError(ValueError):
    pass


def _strip_marker_edges(raw):
    markers = ("<|im_start|>", "<|im_end|>")
    text = raw
    while True:
        stripped = text.strip()
        changed = False
        for marker in markers:
            if stripped.startswith(marker):
                stripped = stripped[len(marker):]
                changed = True
            if stripped.endswith(marker):
                stripped = stripped[: -len(marker)]
                changed = True
        if not changed:
            return stripped
        text = stripped


def _decode_action(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=pairs)
    try:
        action, end = decoder.raw_decode(text)
    except (ValueError, TypeError) as exc:
        reason = "multiple_tool_calls" if (
            "<tool_call>" in text or "</tool_call>" in text
        ) else "invalid_json"
        raise ActionError(reason) from exc
    if text[end:].strip():
        raise ActionError("multiple_tool_calls")
    if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
        raise ActionError("invalid_tool_schema")
    try:
        validate_arguments(action["name"], action["arguments"])
    except (ValueError, TypeError) as exc:
        raise ActionError("invalid_tool_schema") from exc
    return action


def strict_action(raw):
    """Accept exactly one protocol-valid JSON action.

    Qwen template markers are tolerated at the edges. The action may be bare
    JSON or wrapped in one ``<tool_call>`` pair; prose and extra calls fail.
    """
    text = _strip_marker_edges(raw)
    if text.startswith("<tool_call>") and text.endswith("</tool_call>"):
        text = text[len("<tool_call>"): -len("</tool_call>")].strip()
    return _decode_action(text)


def executable_action(raw):
    """Return an executable action and its protocol format.

    ``response_wrapper`` is recoverable but not protocol-valid. This measures
    code correctness without hiding the model's format error. Arbitrary prose
    or embedded JSON remains rejected.
    """
    try:
        return strict_action(raw), "strict"
    except ActionError as strict_error:
        text = _strip_marker_edges(raw)
        if text.startswith("<response>") and text.endswith("</response>"):
            inner = text[len("<response>"): -len("</response>")].strip()
            return _decode_action(inner), "response_wrapper"
        if text.startswith("```json") and text.endswith("```"):
            inner = text[len("```json"): -len("```")].strip()
            return _decode_action(inner), "json_fence"
        raise strict_error


def parse_action(raw):
    """Compatibility helper; the evaluator itself retains the failure reason."""
    try:
        return strict_action(raw)
    except ActionError:
        return None


@dataclass(frozen=True)
class EvalConfig:
    mode: str = "problem-only"
    max_model_len: int = 32768
    max_actions: int = 4
    max_submits: int = 3
    max_runs: int = 3
    max_new_tokens: int | None = None  # None means all remaining context.
    max_total_new_tokens: int | None = None
    save_completions: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("invalid mode")
        if min(self.max_model_len, self.max_actions, self.max_submits) <= 0 or self.max_runs < 0:
            raise ValueError("invalid action/context budget")
        for value in (self.max_new_tokens, self.max_total_new_tokens):
            if value is not None and value <= 0:
                raise ValueError("token budgets must be positive")

    @property
    def total_token_budget(self):
        return self.max_total_new_tokens or self.max_actions * self.max_model_len

    def available_tokens(self, prompt_tokens, generated_tokens):
        return max(0, min(self.max_model_len - prompt_tokens,
                          self.total_token_budget - generated_tokens,
                          self.max_new_tokens or self.max_model_len))


class AgentEnv:
    def __init__(self, row, candidate=None, grader=None):
        self.row = row
        self.current_candidate = candidate
        self.grader = grader or PrivateGrader(SandboxedExecutor(
            SandboxConfig(timeout_sec=3, memory_mb=512, backend="local")))

    def submit(self, code):
        self.current_candidate = code
        r = self.row
        try:
            compile(code, "<candidate>", "exec")
        except (SyntaxError, ValueError, TypeError) as exc:
            error = {"type": type(exc).__name__,
                     "message": getattr(exc, "msg", str(exc))}
            if isinstance(exc, SyntaxError):
                error.update(line=exc.lineno, offset=exc.offset)
            return {"status": "compile_error", "passed": 0,
                    "total": len(r["inputs"]), "pass_rate": 0.0,
                    "failing_input": None, "error": error}
        return project_submit_feedback(self.grader.grade(
            code, r["inputs"], r["outputs"], r["io_mode"], r.get("fn_name")))

    def run(self, input_text):
        if self.current_candidate is None:
            return {"status": "invalid_input", "error": {
                "type": "NoCandidate", "message": "No candidate yet; submit complete code first."}}
        result, _ = self.grader.run_candidate(self.current_candidate, input_text,
                                               self.row["io_mode"], self.row.get("fn_name"))
        return result.public_dict()


def build_initial_messages(row, mode="problem-only", candidate=None, feedback=None):
    problem = PublicProblem(
        question=row["question"], starter_code=row.get("starter_code") or "",
        difficulty=row["difficulty"], io_mode=row["io_mode"], fn_name=row.get("fn_name"),
        input_format_note=("Pass one complete stdin string to run_candidate(input)."
            if row["io_mode"] == "stdin" else
            "Pass a canonical JSON argument list to run_candidate(input)."))
    if mode == "problem-only":
        state = build_problem_only_state(problem)
    elif mode == "candidate":
        state = build_pre_submit_state(problem, candidate)
    elif mode == "repair":
        state = build_post_submit_state(problem, candidate, feedback)
    else:
        raise ValueError("unknown mode")
    return [{k: v for k, v in message.items() if k != "trainable"} for message in state]


def evaluate_problem(row, tokenizer, generate, config, candidate=None, env_factory=AgentEnv):
    """generate(ids, max_tokens, allowed_names) returns one model turn."""
    started = time.monotonic()
    result = {"id": problem_id(row), "difficulty": row["difficulty"], "mode": config.mode,
              "solved": False, "status": "action_budget", "actions": 0,
              "submissions": 0, "runs": 0, "generated_tokens": 0,
              "first_submit_success": None, "repair_eligible": False,
              "valid_actions": 0, "executable_actions": 0,
              "no_candidate_calls": 0, "tool_timeouts": 0,
              "duplicate_submissions": 0,
              "turn_history": []}
    try:
        env = env_factory(row, candidate)
        feedback = None
        if config.mode != "problem-only":
            if not isinstance(candidate, str) or not candidate.strip():
                result["status"] = "candidate_invalid"
                return result
            # Verify the fixed erroneous candidate using this exact grader.
            # This setup call is not charged as a model submission.
            feedback = env.submit(candidate)
            result["seed_feedback"] = feedback
            if feedback["status"] == "accepted":
                result["status"] = "candidate_not_wrong"
                return result
            result["repair_eligible"] = True
        messages = build_initial_messages(row, config.mode, candidate, feedback)
        submitted_feedback = ({candidate: feedback}
                              if isinstance(candidate, str) and feedback is not None else {})
        for _ in range(config.max_actions):
            allowed_names = (("submit",) if env.current_candidate is None
                             else ("run_candidate", "submit"))
            available_tools = [tool for tool in tool_schemas()
                               if tool["function"]["name"] in allowed_names]
            ids = tokenizer.apply_chat_template(messages, tools=available_tools,
                    tokenize=True, add_generation_prompt=True,
                    enable_thinking=False)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            budget = config.available_tokens(len(ids), result["generated_tokens"])
            if budget == 0:
                result["status"] = ("context_exhausted" if len(ids) >= config.max_model_len
                                    else "generation_budget")
                break
            event = {"prompt_tokens": len(ids), "max_new_tokens": budget,
                     "available_tools": list(allowed_names)}
            result["turn_history"].append(event)
            result["actions"] += 1
            output = generate(ids, budget, allowed_names)
            count = output["token_count"]
            if not isinstance(count, int) or not 0 <= count <= budget:
                raise ValueError("backend returned invalid token count")
            result["generated_tokens"] += count
            event.update(generated_tokens=count, finish_reason=output["finish_reason"])
            if config.save_completions:
                event["raw"] = output["text"]
            if output["finish_reason"] == "length":
                result["status"] = "truncated"
                break
            if output["finish_reason"] != "stop":
                result["status"] = "generation_error"
                break
            try:
                action, action_format = executable_action(output["text"])
            except ActionError as exc:
                result["status"] = "invalid_action"
                event["parse_error"] = str(exc)
                break
            if action["name"] not in allowed_names:
                result["status"] = "invalid_action"
                event["parse_error"] = "unavailable_tool"
                break
            result["executable_actions"] += 1
            event["action_format"] = action_format
            event["protocol_valid"] = action_format == "strict"
            if event["protocol_valid"]:
                result["valid_actions"] += 1
            event["tool"] = action["name"]
            if config.save_completions:
                event["action"] = action
            if action["name"] == "submit":
                if result["submissions"] >= config.max_submits:
                    result["status"] = "submit_budget"
                    break
                result["submissions"] += 1
                code = action["arguments"]["code"]
                if code in submitted_feedback:
                    result["duplicate_submissions"] += 1
                    observation = dict(submitted_feedback[code], duplicate=True,
                        message="Identical code was already submitted; change the candidate before submitting again.")
                else:
                    observation = env.submit(code)
                    submitted_feedback[code] = observation
                accepted = (observation["status"] == "accepted" and
                            observation["passed"] == observation["total"] and
                            observation["total"] > 0 and observation["pass_rate"] == 1.0)
                if result["submissions"] == 1:
                    result["first_submit_success"] = accepted
                if not accepted:
                    result["repair_eligible"] = True
                result["solved"] = accepted
            else:
                if result["runs"] >= config.max_runs:
                    result["status"] = "run_budget"
                    break
                result["runs"] += 1
                observation = env.run(action["arguments"]["input"])
                if (observation.get("error") or {}).get("type") == "NoCandidate":
                    result["no_candidate_calls"] += 1
            event["observation"] = observation
            if observation["status"] == "timeout":
                result["tool_timeouts"] += 1
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": action}]})
            messages.append({"role": "tool", "name": action["name"],
                             "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))})
            if result["solved"]:
                result["status"] = "accepted"
                break
    except Exception as exc:
        result["status"] = "infrastructure_error"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        result["wall_seconds"] = round(time.monotonic() - started, 3)
    return result


def summarize(results, expected_total):
    n = len(results)
    solved = sum(r["solved"] for r in results)
    eligible = [r for r in results if r["repair_eligible"]]
    actions = sum(r["actions"] for r in results)
    rate = lambda num, den: num / den if den else None
    return {"expected_total": expected_total, "completed": n,
            "complete": n == expected_total, "solved": solved,
            "budgeted_agent_success_rate": rate(solved, n),
            "first_submit_success_rate": rate(sum(r["first_submit_success"] is True for r in results), n),
            "repair_eligible": len(eligible),
            "repair_success_rate": rate(sum(r["solved"] for r in eligible), len(eligible)),
            "tool_format_valid_rate": rate(sum(r["valid_actions"] for r in results), actions),
            "tool_action_parse_rate": rate(sum(r.get("executable_actions", r["valid_actions"])
                                                for r in results), actions),
            "mean_submissions": rate(sum(r["submissions"] for r in results), n),
            "mean_runs": rate(sum(r["runs"] for r in results), n),
            "no_candidate_calls": sum(r["no_candidate_calls"] for r in results),
            "tool_timeouts": sum(r["tool_timeouts"] for r in results),
            "duplicate_submissions": sum(r.get("duplicate_submissions", 0) for r in results),
            "terminal_status": dict(Counter(r["status"] for r in results)),
            "parse_errors": dict(Counter(t["parse_error"] for r in results
                for t in r["turn_history"] if "parse_error" in t)),
            "by_difficulty": {d: {"total": len(items), "solved": sum(r["solved"] for r in items),
                "success_rate": rate(sum(r["solved"] for r in items), len(items))}
                for d in ("introductory", "interview", "competition")
                for items in [[r for r in results if r["difficulty"] == d]]}}


class ProgressStore:
    """Single-writer, config-bound progress; fsync every completed problem."""
    def __init__(self, root, manifest, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "progress.jsonl"
        self.hash = digest(manifest)
        mp = self.root / "run_manifest.json"
        if mp.exists():
            if not resume:
                raise FileExistsError("run exists; use --resume or a new --output-dir")
            if json.loads(mp.read_text()) != manifest:
                raise ValueError("resume model/data/config/source fingerprint mismatch")
        else:
            if self.path.exists() or (self.root / "results.json").exists():
                raise ValueError("unbound existing results; use a new output directory")
            atomic_write_json(mp, manifest)
        self.records = {}
        if self.path.exists():
            data = self.path.read_bytes().splitlines(keepends=True)
            for i, line in enumerate(data):
                try:
                    row = json.loads(line)
                except ValueError:
                    if i == len(data)-1 and not line.endswith(b"\n"):
                        # Recover only an interrupted final append, never corruption.
                        atomic_write_jsonl(self.path, self.records.values())
                        break
                    raise ValueError("corrupt progress JSONL")
                if row.get("run_hash") != self.hash or row["id"] in self.records:
                    raise ValueError("progress fingerprint mismatch or duplicate id")
                self.records[row["id"]] = row
            # Normalize a valid but newline-free final record before append.
            if data and not data[-1].endswith(b"\n"):
                atomic_write_jsonl(self.path, self.records.values())

    def append(self, result):
        if result["id"] in self.records:
            raise ValueError("duplicate result")
        row = dict(result, run_hash=self.hash)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records[row["id"]] = row
