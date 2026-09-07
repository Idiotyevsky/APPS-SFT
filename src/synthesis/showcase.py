from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .apps_loader import stream_apps_jsonl
from .io_utils import read_jsonl


def _message_view(index: int, message: dict[str, Any]) -> str:
    role = message["role"].upper()
    marker = (
        "[SUPERVISED]" if message.get("trainable")
        else "[MODEL VISIBLE]"
    )
    loss = "assistant span contributes loss" if message.get("trainable") else "labels=-100"
    body = message.get("content")
    if body is None:
        body = json.dumps(
            message.get("tool_calls"), ensure_ascii=False, indent=2,
        )
    fence = chr(96) * 3
    return (
        f"#### {marker} {role} (message {index})\n\n"
        f"loss_mask: {loss}\n\n"
        f"{fence}text\n{body}\n{fence}"
    )


def _choose_examples(
    episodes: list[dict[str, Any]],
    examples_per_origin: int,
    examples_per_behavior: int,
) -> list[dict[str, Any]]:
    if examples_per_origin < 1 or examples_per_behavior < 1:
        raise ValueError("example counts must be positive")
    chosen: dict[str, dict[str, Any]] = {}
    origin_counts: dict[str, int] = {}
    behavior_counts: dict[str, int] = {}
    bug_counts: dict[int, int] = {}
    io_counts: dict[str, int] = {}
    for episode in sorted(episodes, key=lambda item: item["id"]):
        metadata = episode["metadata"]
        origin = metadata["candidate"]["origin"]
        behavior = metadata["behavior_sequence"][0]
        mutation = metadata.get("mutation") or {}
        bug_count = mutation.get("bug_count")
        io_mode = metadata.get("io_mode")
        has_failed_submit = any(
            message.get("role") == "assistant"
            and not message.get("trainable")
            and any(
                call.get("name") == "submit"
                for call in message.get("tool_calls", [])
            )
            for message in episode.get("messages", [])
        )

        def add() -> None:
            chosen.setdefault(episode["id"], episode)

        if origin_counts.get(origin, 0) < examples_per_origin:
            add()
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
        if behavior_counts.get(behavior, 0) < examples_per_behavior:
            add()
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        if origin == "synthetic_multi" and bug_count in {2, 3, 4}:
            if bug_counts.get(int(bug_count), 0) < 1:
                add()
                bug_counts[int(bug_count)] = 1
        if io_mode in {"stdin", "call"} and io_counts.get(io_mode, 0) < 1:
            add()
            io_counts[io_mode] = 1
        if has_failed_submit:
            add()
    return list(chosen.values())


def _source_sample(
    root: Path, episodes: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    manifest_path = root / "source_manifest.json"
    cleaned_path = root / "cleaned" / "problems.jsonl"
    if not manifest_path.exists() or not cleaned_path.exists():
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("path")
    if not isinstance(source, str) or not Path(source).exists():
        return None, None
    cleaned = {
        str(row.get("id", row.get("problem_id"))): row
        for row in read_jsonl(cleaned_path)
    }
    preferred_id = (
        str(episodes[0]["metadata"].get("id", episodes[0]["metadata"].get("problem_id"))) if episodes else None
    )
    raw_sample: dict[str, Any] | None = None
    for _, raw in stream_apps_jsonl(source):
        if preferred_id is None or str(raw.get("id", raw.get("problem_id"))) == preferred_id:
            raw_sample = raw
            break
    if raw_sample is None:
        return None, None
    cleaned_row = cleaned.get(str(raw_sample.get("id", raw_sample.get("problem_id"))))
    public = None
    if cleaned_row is not None:
        public = {
            "id": cleaned_row.get("id", cleaned_row.get("problem_id")),
            "public_problem": cleaned_row["public_problem"],
            "source": cleaned_row.get("source", {}),
        }
    return raw_sample, public


def generate_showcase(
    output_dir: str | Path, examples_per_origin: int = 1,
    examples_per_behavior: int = 1,
    show_raw_apps_record: bool = False,
) -> Path:
    root = Path(output_dir)
    episodes = list(read_jsonl(root / "episodes.jsonl"))
    qa_path = root / "qa_report.json"
    qa = (
        json.loads(qa_path.read_text(encoding="utf-8"))
        if qa_path.exists() else {"passed": False}
    )
    if not qa.get("passed"):
        raise ValueError(
            "showcase can only be generated from QA-passing artifacts"
        )
    chosen = _choose_examples(
        episodes, examples_per_origin, examples_per_behavior,
    )
    tokenized = {}
    tokenized_path = root / "sft_tokenized.jsonl"
    if tokenized_path.exists():
        tokenized = {
            row["id"]: row for row in read_jsonl(tokenized_path)
        }
    lines = [
        "# Synthesis Showcase", "",
        "All examples below were loaded from QA-passing artifacts; "
        "none are hand-written.", "",
    ]
    fence = chr(96) * 3

    if show_raw_apps_record:
        raw, public = _source_sample(root, episodes)
        lines += ["## APPS source sample", ""]
        if raw is None:
            lines += [
                "[INTERNAL ONLY] Raw APPS record unavailable at the recorded "
                "source path.", "",
            ]
        else:
            lines += [
                "### [INTERNAL ONLY] Raw APPS record", "",
                fence + "json",
                json.dumps(raw, ensure_ascii=False, indent=2),
                fence, "",
                "### [MODEL VISIBLE] Cleaned public record", "",
                fence + "json",
                json.dumps(public, ensure_ascii=False, indent=2),
                fence, "",
            ]

    for episode in chosen:
        metadata = episode["metadata"]
        lines += [
            f"## {episode['id']}", "",
            f"Origin: {metadata['candidate']['origin']}",
            f"Behavior: {metadata['behavior_sequence'][0]}",
            f"Difficulty: {metadata['difficulty']}",
            f"I/O mode: {metadata['io_mode']}", "",
        ]
        lines.extend(
            _message_view(index, message)
            for index, message in enumerate(episode["messages"])
        )
        row = tokenized.get(episode["id"])
        if row is not None:
            lines += [
                "", "### [INTERNAL ONLY] Token-level loss mask spans", "",
                fence + "json",
                json.dumps(row.get("spans", []), ensure_ascii=False, indent=2),
                fence, "",
            ]
        lines += [
            "", "### [INTERNAL ONLY] Message-level loss mask", "",
            fence + "json",
            json.dumps([
                {
                    "index": index,
                    "role": message.get("role"),
                    "name": message.get("name"),
                    "trainable": bool(message.get("trainable")),
                    "labels": (
                        "assistant_tokens" if message.get("trainable")
                        else "-100"
                    ),
                }
                for index, message in enumerate(episode["messages"])
            ], ensure_ascii=False, indent=2),
            fence, "",
            "### [INTERNAL ONLY] Metadata", "",
            fence + "json",
            json.dumps(metadata, ensure_ascii=False, indent=2),
            fence, "",
        ]
    path = root / "SYNTHESIS_SHOWCASE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
