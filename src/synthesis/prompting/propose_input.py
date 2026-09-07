from __future__ import annotations

import json
from typing import Any

from ..schemas import PublicProblem


PROPOSAL_OBSERVABLE_FIELDS = ("question", "starter_code", "io_mode", "input_format_note", "fn_name", "candidate")


def build_proposal_prompt(problem: PublicProblem, candidate: str) -> tuple[str, dict[str, Any]]:
    """Build only from the explicit observable whitelist; signature rejects oracle fields."""
    values = {
        "question": problem.question,
        "starter_code": problem.starter_code,
        "io_mode": problem.io_mode,
        "input_format_note": problem.input_format_note,
        "fn_name": problem.fn_name,
        "candidate": candidate,
    }
    prompt = (
        "Propose exactly one diagnostic input using only the public problem and current candidate below. "
        "Return one JSON object with exactly the key input and a string value. Do not solve or submit.\n"
        + json.dumps(values, ensure_ascii=False, sort_keys=True)
    )
    return prompt, values


def validate_proposal_provenance(fields: dict[str, Any]) -> None:
    if tuple(fields.keys()) != PROPOSAL_OBSERVABLE_FIELDS:
        raise ValueError("proposal provenance contains non-whitelisted fields")


def parse_proposed_input(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"input"} or not isinstance(value["input"], str):
        raise ValueError("proposal must be a JSON object containing only string key input")
    return value["input"]

