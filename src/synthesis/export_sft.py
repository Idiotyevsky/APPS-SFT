from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

from .io_utils import atomic_write_jsonl
from .loss_mask import decode_supervised, tokenize_with_loss_mask


SECRET_METADATA_KEYS = {"private_evaluation", "reference_code", "expected", "cases", "mutation_edits_internal"}


def public_episode(episode: dict[str, Any]) -> dict[str, Any]:
    return {"id": episode["id"], "tools": episode["tools"], "messages": episode["messages"], "metadata_ref": episode["id"]}


def export_messages(episodes: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    return atomic_write_jsonl(output_path, (public_episode(episode) for episode in episodes))


_CANDIDATE_PATTERN = re.compile(
    r"Current candidate:\n```python\n(.*?)\n```",
    flags=re.DOTALL,
)


def validate_decoded_supervision(
    episode: dict[str, Any], supervised: str,
) -> None:
    for message in episode["messages"]:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):
                if (
                    message.get("trainable")
                    and call["name"] not in supervised
                ):
                    raise ValueError(
                        "decoded labels omit a trainable tool call"
                    )
                if (
                    not message.get("trainable")
                    and call["name"] == "submit"
                    and call["arguments"]["code"] in supervised
                ):
                    raise ValueError(
                        "failed submit code appears in supervised labels"
                    )
        if message.get("role") == "tool":
            content = message.get("content") or ""
            if len(content) >= 8 and content in supervised:
                raise ValueError(
                    "tool response appears in supervised labels"
                )
        if message.get("role") == "user":
            match = _CANDIDATE_PATTERN.search(
                message.get("content") or ""
            )
            if match and match.group(1).strip() in supervised:
                raise ValueError(
                    "seed buggy candidate appears as a full target"
                )


def export_tokenized(episodes: Iterable[dict[str, Any]], tokenizer, output_path: str | Path) -> int:
    def records():
        for episode in episodes:
            tokenized = tokenize_with_loss_mask(tokenizer, episode["messages"], episode["tools"])
            supervised = decode_supervised(tokenizer, tokenized["input_ids"], tokenized["labels"])
            validate_decoded_supervision(episode, supervised)
            tokenized["supervised_text"] = supervised
            yield {"id": episode["id"], **tokenized}
    return atomic_write_jsonl(output_path, records())

