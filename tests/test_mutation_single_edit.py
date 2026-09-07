from synthesis.mutation import (
    apply_edits, enumerate_supported_single_edits,
    sample_compatible_sets, validate_multi_ast_edits,
    validate_single_ast_edit,
)


SOURCE = """
def solve(n):
    total = n + 1
    if total < 10:
        total = max(total, 2)
    return total
"""


def test_supported_families_are_enumerated():
    edits = enumerate_supported_single_edits(SOURCE)
    families = {edit.record.family for edit in edits}
    assert {
        "arithmetic", "comparator", "boundary_constant",
        "aggregation", "return_expression",
    } <= families


def test_single_edit_reapplies_from_original_ast():
    edit = next(
        item for item in enumerate_supported_single_edits(SOURCE)
        if item.record.family == "comparator"
    )
    mutant, records = apply_edits(SOURCE, [edit])
    assert validate_single_ast_edit(SOURCE, mutant, edit)
    assert len(records) == 1
    assert records[0].semantic_edit_count == 1


def test_multi_edits_are_pairwise_compatible_and_exact():
    edits = enumerate_supported_single_edits(SOURCE)
    selected = next(sample_compatible_sets(edits, 2, seed=7))
    mutant, records = apply_edits(SOURCE, selected)
    assert validate_multi_ast_edits(SOURCE, mutant, selected)
    assert len(records) == 2
    assert selected[0].statement_path != selected[1].statement_path
