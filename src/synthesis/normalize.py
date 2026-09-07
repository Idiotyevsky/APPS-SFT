from __future__ import annotations

import ast
from hashlib import sha256
import io
import json
import tokenize
from typing import Any

from .schemas import IOMode


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_call_input(value: Any) -> str:
    args = value if isinstance(value, list) else [value]
    canonical_json(args)  # prove serializability
    return canonical_json(args)


def normalize_stdin_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) + "\n"
    return str(value)


def normalize_input(value: Any, mode: str) -> str:
    return normalize_call_input(value) if mode == IOMode.CALL.value else normalize_stdin_input(value)


def normalize_stdout(value: str) -> list[str]:
    return [line.rstrip() for line in value.strip().splitlines()]


def _candidate_expected_forms(expected: Any, mode: str) -> list[Any]:
    if mode == IOMode.STDIN.value:
        if (
            isinstance(expected, list)
            and expected
            and all(isinstance(x, str) for x in expected)
        ):
            # APPS sometimes stores several accepted textual representations.
            return list(expected)
        return [expected]
    # APPS call-based records commonly encode one expected sequence as
    # [value]. The official evaluator accepts both the wrapper and value.
    forms = [expected]
    if isinstance(expected, list) and expected:
        forms.append(expected[0])
    return forms


def compare_output(actual: Any, expected: Any, mode: str) -> bool:
    if mode == IOMode.STDIN.value:
        actual_lines = normalize_stdout(str(actual))
        for form in _candidate_expected_forms(expected, mode):
            if actual_lines == normalize_stdout(str(form)):
                return True
            if " ".join(actual_lines).split() == str(form).split():
                return True
        return False
    actual_forms = [actual]
    if isinstance(actual, tuple):
        actual_forms.append(list(actual))
    if isinstance(actual, list) and actual and all(
        isinstance(value, tuple) for value in actual
    ):
        actual_forms.append([list(value) for value in actual])
    for actual_form in actual_forms:
        for form in _candidate_expected_forms(expected, mode):
            if actual_form == form:
                return True
            try:
                if canonical_json(actual_form) == canonical_json(form):
                    return True
            except (TypeError, ValueError):
                pass
    return False


def parses_and_compiles(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def ast_node_count(code: str) -> int:
    return sum(1 for _ in ast.walk(ast.parse(code)))


def normalize_code(code: str) -> str:
    """Normalize Python code without changing string literals or semantics."""
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except SyntaxError:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        return " ".join(tok.string for tok in tokens if tok.type not in {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING})


def content_hash(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + sha256(raw).hexdigest()


def normalized_code_hash(code: str) -> str:
    return content_hash(normalize_code(code))


def normalized_input_equal(left: str, right: str, mode: str) -> bool:
    if mode == IOMode.CALL.value:
        try:
            return canonical_json(json.loads(left)) == canonical_json(json.loads(right))
        except json.JSONDecodeError:
            return False
    return left.encode("utf-8") == right.encode("utf-8")
