from __future__ import annotations

import ast
from copy import deepcopy
from typing import Iterable

from .base import MutationEdit, MutationRecord, render_node, source_span, walk_with_paths


COMPARE_SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
BINOP_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Add,
}


def _edit(path, statement, old: ast.AST, new: ast.AST, operator: str, family: str) -> MutationEdit:
    ast.copy_location(new, old)
    return MutationEdit(
        path=path, replacement=new, statement_path=statement,
        record=MutationRecord(operator, family, type(old).__name__, source_span(old), render_node(old), render_node(new)),
    )


def enumerate_supported_single_edits(source: str, families: Iterable[str] | None = None) -> list[MutationEdit]:
    tree = ast.parse(source)
    allowed = set(families or ())
    edits: list[MutationEdit] = []
    seen: set[tuple[object, ...]] = set()

    def add(edit: MutationEdit) -> None:
        key = (edit.path, edit.record.family, edit.record.after)
        if (not allowed or edit.record.family in allowed) and key not in seen:
            edits.append(edit)
            seen.add(key)

    for node, path, statement in walk_with_paths(tree):
        if isinstance(node, ast.Compare):
            for index, op in enumerate(node.ops):
                replacement = COMPARE_SWAPS.get(type(op))
                if replacement:
                    add(_edit(path + ("ops", index), statement, op, replacement(), "compare_boundary", "comparator"))
        if isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAPS:
            replacement = BINOP_SWAPS[type(node.op)]()
            add(_edit(path + ("op",), statement, node.op, replacement, "binary_operator_swap", "arithmetic"))
        if isinstance(node, ast.BoolOp):
            replacement = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            add(_edit(path + ("op",), statement, node.op, replacement, "boolean_operator_swap", "boolean"))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            add(_edit(path, statement, node, deepcopy(node.operand), "remove_condition_negation", "boolean"))
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            candidates = [1 if node.value == 0 else 0] if node.value in (0, 1) else [node.value - 1, node.value + 1]
            for value in candidates:
                add(_edit(path, statement, node, ast.Constant(value=value), "boundary_constant_offset", "boundary_constant"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            stop_index = 0 if len(node.args) == 1 else 1
            old = node.args[stop_index]
            for delta in (-1, 1):
                op = ast.Sub() if delta < 0 else ast.Add()
                replacement = ast.BinOp(left=deepcopy(old), op=op, right=ast.Constant(value=1))
                add(_edit(path + ("args", stop_index), statement, old, replacement, "range_stop_offset", "range_bound"))
            if len(node.args) == 1:
                reverse = ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.BinOp(deepcopy(old), ast.Sub(), ast.Constant(1)), ast.Constant(0), ast.Constant(-1)], keywords=[],
                )
                add(_edit(path, statement, node, reverse, "reverse_loop_iterator", "loop_direction"))
        if isinstance(node, ast.Subscript):
            old = node.slice
            for delta in (-1, 1):
                replacement = ast.BinOp(deepcopy(old), ast.Sub() if delta < 0 else ast.Add(), ast.Constant(1))
                add(_edit(path + ("slice",), statement, old, replacement, "index_offset", "index_offset"))
            if isinstance(node.ctx, ast.Store):
                replacement = deepcopy(node)
                replacement.slice = ast.BinOp(deepcopy(node.slice), ast.Add(), ast.Constant(1))
                add(_edit(path, statement, node, replacement, "adjacent_update_target", "update_target"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"min", "max", "sum", "len"}:
            swaps = {"min": "max", "max": "min", "sum": "len", "len": "sum"}
            old = node.func
            add(_edit(path + ("func",), statement, old, ast.Name(id=swaps[old.id], ctx=ast.Load()), "aggregation_swap", "aggregation"))
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
                replacement = ast.Constant(value=value.value + 1)
                add(_edit(path + ("value",), statement, value, replacement, "initialization_offset", "initialization"))
        if isinstance(node, ast.Return) and node.value is not None:
            replacement = ast.BinOp(deepcopy(node.value), ast.Add(), ast.Constant(1))
            add(_edit(path + ("value",), statement, node.value, replacement, "return_expression_offset", "return_expression"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join" and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
            old = node.func.value
            separator = " " if old.value != " " else ""
            add(_edit(path + ("func", "value"), statement, old, ast.Constant(separator), "join_separator_change", "input_output_handling"))
    return edits

