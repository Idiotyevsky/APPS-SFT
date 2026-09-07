from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    key: str
    payload: dict[str, Any]


class JSONLCheckpoint:
    """Append-only, config-bound checkpoint for idempotent stage resumption.

    Each key is committed at most once. A crash can leave a truncated final
    line; readers ignore only that final incomplete record and retain all
    previous commits. A different config digest is rejected to prevent mixing
    artifacts from incompatible runs.
    """

    VERSION = 1

    def __init__(self, path: str | Path, config_hash: str) -> None:
        if not isinstance(config_hash, str) or not config_hash:
            raise ValueError("checkpoint requires a non-empty config hash")
        self.path = Path(path)
        self.config_hash = config_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, CheckpointRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # A process may have been killed during the final append.
                    # A malformed complete line indicates corruption; a
                    # newline-free final line is the only tolerated truncation.
                    if line.endswith("\n"):
                        raise ValueError(
                            f"invalid checkpoint JSON at line {line_number}"
                        )
                    break
                if not isinstance(value, dict):
                    raise ValueError(
                        f"checkpoint line {line_number} is not an object"
                    )
                if value.get("version") != self.VERSION:
                    raise ValueError("unsupported checkpoint version")
                if value.get("config_hash") != self.config_hash:
                    raise ValueError(
                        "checkpoint config hash differs from current config"
                    )
                key = value.get("key")
                payload = value.get("payload")
                if not isinstance(key, str) or not key:
                    raise ValueError("checkpoint key must be a non-empty string")
                if not isinstance(payload, dict):
                    raise ValueError("checkpoint payload must be an object")
                previous = self._records.get(key)
                record = CheckpointRecord(key, payload)
                if previous is not None and previous.payload != payload:
                    raise ValueError(
                        f"checkpoint key {key!r} was committed with conflicting payloads"
                    )
                self._records[key] = record

    def keys(self) -> set[str]:
        return set(self._records)

    def get(self, key: str) -> dict[str, Any] | None:
        record = self._records.get(key)
        return None if record is None else dict(record.payload)

    def records(self) -> Iterator[CheckpointRecord]:
        yield from self._records.values()

    def append(self, key: str, payload: dict[str, Any]) -> bool:
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint key must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be an object")
        previous = self._records.get(key)
        if previous is not None:
            if previous.payload != payload:
                raise ValueError(
                    f"checkpoint key {key!r} was already committed differently"
                )
            return False
        value = {
            "version": self.VERSION,
            "config_hash": self.config_hash,
            "key": key,
            "payload": payload,
        }
        encoded = (
            json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._records[key] = CheckpointRecord(key, dict(payload))
        return True
