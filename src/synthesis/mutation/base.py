from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable


PathPart = str | int


@dataclass(slots=True)
class MutationRecord:
    operator: str
    family: str
    node_type: str
    source_span: dict[str, int | None]
    before: str
    after: str
    semantic_edit_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator, "family": self.family, "node_type": self.node_type,
            "source_span": self.source_span, "before": self.before, "after": self.after,
            "semantic_edit_count": self.semantic_edit_count,
        }


@dataclass(slots=True)
class MutationEdit:
    path: tuple[PathPart, ...]
    replacement: ast.AST
    record: MutationRecord
    statement_path: tuple[PathPart, ...]

    @property
    def stable_key(self) -> str:
        payload = json.dumps({"path": self.path, "record": self.record.to_dict()}, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


def source_span(node: ast.AST) -> dict[str, int | None]:
    return {
        "line": getattr(node, "lineno", None), "col": getattr(node, "col_offset", None),
        "end_line": getattr(node, "end_lineno", None), "end_col": getattr(node, "end_col_offset", None),
    }


def render_node(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (ValueError, AttributeError):
        return ast.dump(node, include_attributes=False)


def walk_with_paths(root: ast.AST):
    def walk(node: ast.AST, path: tuple[PathPart, ...], statement: tuple[PathPart, ...]):
        current_statement = path if isinstance(node, ast.stmt) else statement
        yield node, path, current_statement
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                yield from walk(value, path + (field,), current_statement)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        yield from walk(item, path + (field, index), current_statement)
    yield from walk(root, (), ())


def get_at_path(root: ast.AST, path: tuple[PathPart, ...]) -> ast.AST:
    value: Any = root
    for part in path:
        value = value[part] if isinstance(part, int) else getattr(value, part)
    if not isinstance(value, ast.AST):
        raise TypeError("mutation path does not identify an AST node")
    return value


def replace_at_path(root: ast.AST, path: tuple[PathPart, ...], replacement: ast.AST) -> ast.AST:
    if not path:
        return deepcopy(replacement)
    parent: Any = root
    for part in path[:-1]:
        parent = parent[part] if isinstance(part, int) else getattr(parent, part)
    final = path[-1]
    if isinstance(final, int):
        parent[final] = deepcopy(replacement)
    else:
        setattr(parent, final, deepcopy(replacement))
    return root


def compatible(left: MutationEdit, right: MutationEdit, allow_same_statement: bool = False) -> bool:
    if left.path == right.path:
        return False
    shorter, longer = sorted((left.path, right.path), key=len)
    if longer[:len(shorter)] == shorter:
        return False
    if left.record.before == right.record.after and left.record.after == right.record.before:
        return False
    if left.statement_path == right.statement_path and not allow_same_statement:
        return False
    return True


def apply_edits(source: str, edits: Iterable[MutationEdit]) -> tuple[str, list[MutationRecord]]:
    selected = list(edits)
    for index, left in enumerate(selected):
        for right in selected[index + 1:]:
            if not compatible(left, right):
                raise ValueError("incompatible mutation edits")
    tree = ast.parse(source)
    # Paths point into the original tree. Non-overlap means replacement order is immaterial.
    for edit in sorted(selected, key=lambda value: (len(value.path), value.path), reverse=True):
        replace_at_path(tree, edit.path, edit.replacement)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", [edit.record for edit in selected]


def mutation_set_hash(edits: Iterable[MutationEdit]) -> str:
    payload = ":".join(edit.stable_key for edit in edits)
    return "sha256:" + sha256(payload.encode()).hexdigest()

