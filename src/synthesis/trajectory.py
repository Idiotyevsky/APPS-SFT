from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any

from .schemas import Behavior, CleanProblem, validate_tool_call
from .tools import tool_schemas


@dataclass(slots=True)
class Episode:
    id: str
    problem_id: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]

    def validate(self) -> None:
        if not self.messages or self.messages[-1].get("role") != "tool" or self.messages[-1].get("name") != "submit":
            raise ValueError("episode must end with a submit tool response")
        final = json.loads(self.messages[-1]["content"])
        if final.get("status") != "accepted" or final.get("pass_rate") != 1.0:
            raise ValueError("episode does not end accepted")
        for message in self.messages:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls", []):
                    validate_tool_call(call)
            if message.get("role") != "assistant" and message.get("trainable"):
                raise ValueError("only assistant messages may be trainable")

    def to_dict(self, include_metadata: bool = True) -> dict[str, Any]:
        self.validate()
        result = {"id": self.id, "tools": self.tools, "messages": self.messages, "metadata_ref": self.id}
        if include_metadata:
            result["metadata"] = self.metadata
        return result


def stable_episode_id(problem_id: str, behavior: str, candidate_hash: str) -> str:
    suffix = sha256(f"{problem_id}:{behavior}:{candidate_hash}".encode()).hexdigest()[:12]
    return f"apps-{problem_id}-{behavior}-{suffix}"


def build_episode(problem: CleanProblem, behavior: str, messages: list[dict[str, Any]], metadata: dict[str, Any]) -> Episode:
    if behavior not in {item.value for item in Behavior}:
        raise ValueError("unknown behavior")
    candidate_hash = metadata.get("candidate", {}).get("code_hash") or metadata.get("final_code_hash", "")
    episode_id = stable_episode_id(problem.problem_id, behavior, candidate_hash)
    metadata = {**metadata, "schema_version": "1.0", "episode_id": episode_id, "id": problem.problem_id}
    episode = Episode(episode_id, problem.problem_id, tool_schemas(), messages, metadata)
    episode.validate()
    return episode

