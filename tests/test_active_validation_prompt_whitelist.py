import inspect
import json

import pytest

from synthesis.prompting.propose_input import (
    PROPOSAL_OBSERVABLE_FIELDS, build_proposal_prompt,
    parse_proposed_input, validate_proposal_provenance,
)
from synthesis.schemas import PublicProblem


def problem():
    return PublicProblem(
        question="q", starter_code="", difficulty="interview",
        io_mode="stdin", input_format_note="stdin", fn_name=None,
    )


def test_builder_signature_cannot_accept_oracle_fields():
    parameters = set(inspect.signature(build_proposal_prompt).parameters)
    assert parameters == {"problem", "candidate"}
    assert not {
        "grader_result", "reference_solution", "mutation", "hidden_tests",
    } & parameters


def test_prompt_provenance_is_exact_whitelist():
    prompt, fields = build_proposal_prompt(problem(), "print(1)")
    validate_proposal_provenance(fields)
    assert tuple(fields) == PROPOSAL_OBSERVABLE_FIELDS
    assert "hidden_tests" not in prompt


def test_proposal_parser_is_strict():
    assert parse_proposed_input('{"input":"1\\n"}') == "1\n"
    with pytest.raises(ValueError):
        parse_proposed_input('{"input":"1","expected":"2"}')
