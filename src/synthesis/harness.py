from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .normalize import canonical_json


CALL_RESULT_MARKER = "__SYNTHESIS_CALL_RESULT_7d39__"


def build_call_harness(candidate: str, fn_name: str) -> str:
    if not fn_name or not fn_name.isidentifier():
        raise ValueError("invalid fn_name")
    encoded_name = json.dumps(fn_name)
    return candidate + f'''

import json as _synthesis_json
import sys as _synthesis_sys
_synthesis_args = _synthesis_json.loads(_synthesis_sys.stdin.read())
if not isinstance(_synthesis_args, list):
    raise TypeError("call input must be a JSON argument list")
_synthesis_name = {encoded_name}
_synthesis_fn = globals().get(_synthesis_name)
if _synthesis_fn is None:
    _synthesis_solution = globals().get("Solution")
    if _synthesis_solution is None:
        raise AttributeError("callable not found: " + _synthesis_name)
    _synthesis_fn = getattr(_synthesis_solution(), _synthesis_name)
_synthesis_value = _synthesis_fn(*_synthesis_args)
print({CALL_RESULT_MARKER!r} + _synthesis_json.dumps(_synthesis_value, ensure_ascii=False, separators=(",", ":")))
'''


def parse_call_stdout(stdout: str) -> tuple[Any, str]:
    lines = stdout.splitlines()
    marked = [line for line in lines if line.startswith(CALL_RESULT_MARKER)]
    if len(marked) != 1:
        raise ValueError("call harness did not emit exactly one result")
    payload = marked[0][len(CALL_RESULT_MARKER):]
    value = json.loads(payload)
    visible = "\n".join(line for line in lines if not line.startswith(CALL_RESULT_MARKER))
    if visible:
        visible += "\n"
    return value, visible


def call_input_string(args: Any) -> str:
    if not isinstance(args, list):
        args = [args]
    return canonical_json(args)

