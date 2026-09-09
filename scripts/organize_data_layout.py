#!/usr/bin/env python3
"""Safely organize ToolAPPS datasets without deleting payloads.

Default mode is read-only. Use --apply to checksum, move, verify, and create
compatibility symlinks. Use --rollback only while the manifest still matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "archive" / "archive_manifest.json"

MOVES = (
    ("coding_sft_v5", "work/pools/coding_sft_v5", "active_pool", False),
    ("tiny_protocol_t2", "diagnostics/tiny_protocol_t2", "diagnostic", False),
    ("coding_sft", "archive/datasets/coding_sft", "archived", True),
    ("coding_sft_v3", "archive/datasets/coding_sft_v3", "archived", True),
    ("coding_sft_v3_episodes", "archive/datasets/coding_sft_v3_episodes", "archived", True),
    ("coding_sft_v4", "archive/datasets/coding_sft_v4", "archived", True),
    ("sft_v4_final", "archive/datasets/sft_v4_final", "rejected_qa", True),
    ("sft_v5_final", "archive/datasets/sft_v5_final", "rejected_qa", True),
    ("sft_debug_rule_handoff", "archive/datasets/sft_debug_rule_handoff", "archived", True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict:
    records = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        rel = path.relative_to(root).as_posix()
        info = path.lstat()
        record = {"path": rel, "mode": stat.S_IMODE(info.st_mode)}
        if path.is_symlink():
            record.update(type="symlink", target=os.readlink(path))
        elif path.is_dir():
            record.update(type="directory")
        elif path.is_file():
            size = info.st_size
            total_bytes += size
            record.update(type="file", size=size, sha256=sha256(path))
        else:
            raise RuntimeError(f"unsupported filesystem entry: {path}")
        records.append(record)
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    return {
        "file_count": sum(row["type"] == "file" for row in records),
        "directory_count": sum(row["type"] == "directory" for row in records),
        "total_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": records,
    }


def layout_state(src: Path, dst: Path) -> str:
    if src.is_symlink() and dst.is_dir():
        expected = os.path.relpath(dst, src.parent)
        return "organized" if os.readlink(src) == expected else "wrong_symlink"
    if src.is_dir() and not src.is_symlink() and not dst.exists():
        return "legacy"
    if not src.exists() and not src.is_symlink() and dst.is_dir():
        return "missing_symlink"
    return "conflict"


def readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def restore_modes(root: Path, entries: list[dict], root_mode: int) -> None:
    root.chmod(root_mode | stat.S_IWUSR)
    for row in entries:
        path = root / row["path"]
        if row["type"] != "symlink":
            path.chmod(row["mode"])
    root.chmod(root_mode)


def check() -> int:
    problems = []
    for source, target, role, readonly in MOVES:
        src, dst = DATA / source, DATA / target
        state = layout_state(src, dst)
        print(f"{state:16} {source:28} -> {target} [{role}]")
        if state not in {"legacy", "organized"}:
            problems.append(f"{source}: {state}")
        if state == "organized" and readonly:
            writable = [p for p in (dst, *dst.rglob("*")) if not p.is_symlink() and p.stat().st_mode & 0o222]
            if writable:
                problems.append(f"{target}: {len(writable)} writable archived entries")
    if problems:
        print("problems:", *problems, sep="\n  - ", file=sys.stderr)
        return 1
    return 0


def apply() -> int:
    states = [(entry, layout_state(DATA / entry[0], DATA / entry[1])) for entry in MOVES]
    if all(state == "organized" for _, state in states):
        print("layout already organized")
        return check()
    invalid = [(entry[0], state) for entry, state in states if state != "legacy"]
    if invalid:
        raise RuntimeError(f"refusing mixed/conflicting layout: {invalid}")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "moves": [],
    }
    print("computing pre-move checksums...")
    for source, target, role, readonly in MOVES:
        src = DATA / source
        snap = inventory(src)
        payload["moves"].append({
            "source": source,
            "target": target,
            "role": role,
            "readonly": readonly,
            "root_mode": stat.S_IMODE(src.stat().st_mode),
            "inventory": snap,
        })
        print(f"  {source}: {snap['file_count']} files, {snap['total_bytes']} bytes, {snap['tree_sha256'][:12]}")

    moved = []
    try:
        for row in payload["moves"]:
            src, dst = DATA / row["source"], DATA / row["target"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            moved.append((src, dst))
            after = inventory(dst)
            if after["tree_sha256"] != row["inventory"]["tree_sha256"]:
                raise RuntimeError(f"post-move checksum mismatch: {dst}")
            src.symlink_to(os.path.relpath(dst, src.parent), target_is_directory=True)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for row in payload["moves"]:
            if row["readonly"]:
                readonly_tree(DATA / row["target"])
    except Exception:
        for src, dst in reversed(moved):
            if src.is_symlink():
                src.unlink()
            if dst.exists() and not src.exists():
                dst.rename(src)
        raise
    print(f"wrote {MANIFEST}")
    return check()


def rollback() -> int:
    if not MANIFEST.is_file():
        raise RuntimeError(f"missing manifest: {MANIFEST}")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in payload["moves"]:
        src, dst = DATA / row["source"], DATA / row["target"]
        if layout_state(src, dst) != "organized":
            raise RuntimeError(f"cannot rollback inconsistent entry: {src}")
        current = inventory(dst)
        if current["tree_sha256"] != row["inventory"]["tree_sha256"]:
            raise RuntimeError(f"refusing rollback; content changed: {dst}")
    for row in reversed(payload["moves"]):
        src, dst = DATA / row["source"], DATA / row["target"]
        if row["readonly"]:
            restore_modes(dst, row["inventory"]["entries"], row["root_mode"])
        src.unlink()
        dst.rename(src)
    print("rollback complete; manifest retained for audit")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.apply:
        return apply()
    if args.rollback:
        return rollback()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
