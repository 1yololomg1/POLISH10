"""Canonical state schema. OD-04 Appendix A; D21, D44, D79, D86, D61, UXC2, UXC9."""

import ast
import inspect
import sys
from pathlib import Path

import pytest

from polish.model import states
from polish.model.states import (
    USER_MAPPED_LABEL,
    BlockType,
    DependencyRecord,
    DependencyState,
    Disposition,
    ExportPermission,
    Finding,
    GapFillOutcome,
    GapFillRecord,
    IssueDisposition,
    UserAction,
    export_permission,
    permitted_actions,
)

_STATES_PATH = Path(__file__).resolve().parents[2] / "polish" / "model" / "states.py"

_VIEW_ONLY = (
    BlockType.STRUCTURAL_DATA_FAILURE,
    BlockType.SCIENTIFIC_PRECONDITION_FAILURE,
    BlockType.EXPORT_CONFORMANCE_FAILURE,
)


def _enums():
    found = []
    for name, obj in inspect.getmembers(states, inspect.isclass):
        if obj.__module__ == states.__name__ and issubclass(obj, states.Enum):
            found.append((name, obj))
    return found


def _finding(finding_id, block_type, disposition, reason=""):
    return Finding(
        finding_id,
        block_type,
        Disposition(disposition, reason),
    )


def test_user_mapped_label_is_the_approved_sentence():
    assert USER_MAPPED_LABEL == "User-mapped — not identified by POLISH"


def test_every_enum_value_round_trips_through_its_string():
    for _name, enum_cls in _enums():
        assert list(enum_cls), _name
        for member in enum_cls:
            assert isinstance(member.value, str)
            assert enum_cls(member.value) is member


def test_enum_string_values_are_unique_within_each_enum():
    for name, enum_cls in _enums():
        values = [member.value for member in enum_cls]
        assert len(values) == len(set(values)), name


def test_states_module_imports_only_the_standard_library():
    tree = ast.parse(_STATES_PATH.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    assert modules
    assert modules <= sys.stdlib_module_names


def test_permitted_actions_cover_every_block_type():
    assert set(permitted_actions.__defaults__ or ()) == set()
    for block_type in BlockType:
        actions = permitted_actions(block_type)
        assert isinstance(actions, frozenset)


def test_permitted_actions_match_the_approved_matrix():
    assert permitted_actions(BlockType.INFORMATIONAL) == frozenset(
        {UserAction.ACKNOWLEDGE, UserAction.DEFER}
    )
    assert permitted_actions(BlockType.AMBIGUOUS_INTERPRETATION) == frozenset(
        {UserAction.CHOOSE_OPTION, UserAction.REJECT_PROPOSAL, UserAction.DEFER}
    )
    assert permitted_actions(BlockType.RULE_DEPENDENT_WARNING) == frozenset(
        {UserAction.MAP_EXPLICITLY, UserAction.ACKNOWLEDGE, UserAction.DEFER}
    )


@pytest.mark.parametrize("block_type", _VIEW_ONLY)
def test_view_only_block_types_permit_no_disposition_clearing_action(block_type):
    assert permitted_actions(block_type).isdisjoint(set(UserAction))


def test_export_permission_has_no_override_parameter():
    assert list(inspect.signature(export_permission).parameters) == [
        "findings",
        "dependency_states",
    ]


def _clear_trio():
    return (
        _finding("inf-1", BlockType.INFORMATIONAL, IssueDisposition.ACCEPTED),
        _finding("amb-1", BlockType.AMBIGUOUS_INTERPRETATION, IssueDisposition.RESOLVED),
        _finding(
            "amb-2",
            BlockType.AMBIGUOUS_INTERPRETATION,
            IssueDisposition.DEFERRED,
            reason="Waiting on the client copy of the repeat pass.",
        ),
    )


def test_structural_or_conformance_blocks_whatever_else_is_true():
    current = (
        DependencyRecord("stage-4", DependencyState.CURRENT, affects_export=True),
    )
    structural = _finding(
        "struct-1",
        BlockType.STRUCTURAL_DATA_FAILURE,
        IssueDisposition.ACCEPTED,
    )
    conformance = _finding(
        "conf-1",
        BlockType.EXPORT_CONFORMANCE_FAILURE,
        IssueDisposition.RESOLVED,
    )
    structural_result = export_permission(_clear_trio() + (structural,), current)
    conformance_result = export_permission(_clear_trio() + (conformance,), current)
    both = export_permission(_clear_trio() + (structural, conformance), current)
    assert structural_result.allowed is False
    assert conformance_result.allowed is False
    assert both.allowed is False
    assert any("struct-1" in reason for reason in structural_result.reasons)
    assert any("conf-1" in reason for reason in conformance_result.reasons)
    assert any("struct-1" in reason for reason in both.reasons)
    assert any("conf-1" in reason for reason in both.reasons)


@pytest.mark.parametrize(
    "block_type",
    (
        BlockType.INFORMATIONAL,
        BlockType.AMBIGUOUS_INTERPRETATION,
        BlockType.RULE_DEPENDENT_WARNING,
    ),
)
def test_dispositionable_finding_without_a_terminal_disposition_blocks(block_type):
    finding = _finding("open-1", block_type, IssueDisposition.UNDISPOSITIONED)
    result = export_permission((finding,), ())
    assert result.allowed is False
    assert len(result.reasons) == 1
    assert "open-1" in result.reasons[0]
    assert block_type.value in result.reasons[0]


def test_stale_dependency_that_affects_export_blocks():
    stale = DependencyRecord(
        "stage-4",
        DependencyState.STALE,
        affects_export=True,
        cause="Stage 3 unit decision changed.",
    )
    result = export_permission(_clear_trio(), (stale,))
    assert result.allowed is False
    assert len(result.reasons) == 1
    assert "stage-4" in result.reasons[0]
    assert "Stage 3 unit decision changed." in result.reasons[0]


def test_stale_dependency_that_does_not_affect_export_does_not_block():
    stale = DependencyRecord(
        "presentation",
        DependencyState.STALE,
        affects_export=False,
        cause="Track order changed.",
    )
    result = export_permission((), (stale,))
    assert result.allowed is True
    assert result.reasons == ()


def test_export_allowed_for_accepted_resolved_and_deferred_with_nothing_stale():
    current = (
        DependencyRecord("stage-4", DependencyState.CURRENT, affects_export=True),
    )
    result = export_permission(_clear_trio(), current)
    assert result.allowed is True
    assert result.reasons == ()
    assert isinstance(result, ExportPermission)


def test_scientific_precondition_failure_does_not_block_export():
    finding = _finding(
        "sci-1",
        BlockType.SCIENTIFIC_PRECONDITION_FAILURE,
        IssueDisposition.DECLINED_BY_POLISH,
        reason="Left endpoint is 0.0 ohm·m.",
    )
    result = export_permission(_clear_trio() + (finding,), ())
    assert result.allowed is True
    assert result.reasons == ()


@pytest.mark.parametrize("reason", ("", "   "))
def test_deferred_disposition_without_a_non_empty_reason_cannot_be_constructed(reason):
    with pytest.raises(ValueError, match="DEFERRED"):
        Disposition(IssueDisposition.DEFERRED, reason)


def test_deferred_disposition_with_a_reason_can_be_constructed():
    disposition = Disposition(IssueDisposition.DEFERRED, "Awaiting the splice note.")
    assert disposition.issue_disposition is IssueDisposition.DEFERRED
    assert disposition.reason == "Awaiting the splice note."


def test_refused_gap_fill_without_a_reason_cannot_be_constructed():
    with pytest.raises(ValueError, match="REFUSED"):
        GapFillRecord(GapFillOutcome.REFUSED, reason="")


def test_stale_dependency_without_a_cause_cannot_be_constructed():
    with pytest.raises(ValueError, match="STALE"):
        DependencyRecord("stage-4", DependencyState.STALE, affects_export=True, cause="")
