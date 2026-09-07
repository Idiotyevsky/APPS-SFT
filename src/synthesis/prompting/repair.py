from __future__ import annotations

import json
from typing import Any

from ..tools import TOOL_SCHEMAS


def build_action_prompt(messages: list[dict[str, Any]]) -> str:
    public = [{key: value for key, value in message.items() if key != "trainable"} for message in messages]
    return (
        "Choose the next tool action. Return only JSON with shape "
        '{"name":"run_candidate","arguments":{"input":"..."}} or '
        '{"name":"submit","arguments":{"code":"complete Python program"}}.\n'
        + json.dumps({"tools": TOOL_SCHEMAS, "messages": public}, ensure_ascii=False)
    )

