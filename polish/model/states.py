"""Canonical state schema for POLISH.

Implements D21 (as extended), D44, D79, D86, D33, D34, D32, D40, UXC1, UXC2,
UXC9. Approved as OD-04, Appendix A, on 15 Sept, plus the OD-05 rule that unit
conversion never produces a CHANGED sample.

Structure
---------
Enums, one per operational field in Appendix A.1, plus ``UserAction``.
Frozen records that carry the extra facts the tables require: a deferral
reason, a stale cause, gap evidence and confidence, a gap-fill reason.
Two pure functions: ``permitted_actions`` and ``export_permission``.

No UI label is derived by collapsing these fields into "validated", "flagged",
or "unknown" (UXC1).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

# Approved 15 Sept addition to D34. Every USER_DEFINED curve carries this label,
# including in the export.
USER_MAPPED_LABEL = "User-mapped — not identified by POLISH"


# ---------------------------------------------------------------------------
# Enums — Appendix A.1. String values are the serialisation tokens.
# ---------------------------------------------------------------------------


class CurveRole(Enum):
    """Role of one curve. D32, D40, F2.

    ``GEOMETRIC_REFERENCE`` is the Appendix A name proposed for TVD (OD-17).
    """

    INDEX = "INDEX"
    MEASURED_LOG = "MEASURED_LOG"
    DRILLING = "DRILLING"
    ANCILLARY = "ANCILLARY"
    SYNTHETIC_SOURCE = "SYNTHETIC_SOURCE"
    GEOMETRIC_REFERENCE = "GEOMETRIC_REFERENCE"
    UNKNOWN = "UNKNOWN"


class EvidenceTier(Enum):
    """How a curve identification was obtained. D33.

    Appendix A adds ``NO_MATCH`` to D33's four tiers. CITED, OBSERVED and
    USER_DEFINED are eligible for processing; PROPOSED and NO_MATCH are not
    (D34, approved 15 Sept). A USER_DEFINED curve always carries
    ``USER_MAPPED_LABEL``.
    """

    CITED = "CITED"
    OBSERVED = "OBSERVED"
    USER_DEFINED = "USER_DEFINED"
    PROPOSED = "PROPOSED"
    NO_MATCH = "NO_MATCH"


class ProcessingEligibility(Enum):
    """Whether a curve may enter automatic processing. D34, D11, F15.

    Low tier, missing family, unresolved unit, or a non-processing role each
    stay ineligible. They are not marked invalid.
    """

    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE_NO_FAMILY = "NOT_ELIGIBLE_NO_FAMILY"
    NOT_ELIGIBLE_LOW_TIER = "NOT_ELIGIBLE_LOW_TIER"
    NOT_ELIGIBLE_UNIT_UNRESOLVED = "NOT_ELIGIBLE_UNIT_UNRESOLVED"
    NOT_ELIGIBLE_ROLE = "NOT_ELIGIBLE_ROLE"


class UnitInterpretation(Enum):
    """How the curve's declared unit was read. C1, D87.

    The unit travels with the values. Conversion re-expresses them and does
    not change stored samples (OD-05).
    """

    RECOGNISED = "RECOGNISED"
    ABSENT = "ABSENT"
    EMPTY = "EMPTY"
    UNRECOGNISED = "UNRECOGNISED"
    INCOMPATIBLE_WITH_FAMILY = "INCOMPATIBLE_WITH_FAMILY"


class ValidationOutcome(Enum):
    """Result of automatic validation for one curve. UXC9.

    Distinct from processing eligibility. A curve that was not assessed has
    not passed.
    """

    PASSED = "PASSED"
    PLAUSIBILITY_WARNING = "PLAUSIBILITY_WARNING"
    HARD_VIOLATION_EXCLUDED = "HARD_VIOLATION_EXCLUDED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    PENDING = "PENDING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SampleStatus(Enum):
    """Status of one sample. D44.

    There is no REFUSED sample status. A gap POLISH declined to fill leaves
    samples ABSENT; the gap record carries the refusal. INVALID means the
    value is outside a confirmed hard limit. Whether that sample is left out
    of the processed file is ``OutputInclusion``, decided separately.
    Unit conversion never produces CHANGED (OD-05).
    """

    MEASURED = "MEASURED"
    INVALID = "INVALID"
    ABSENT = "ABSENT"
    CREATED = "CREATED"
    CHANGED = "CHANGED"
    RESAMPLED = "RESAMPLED"


class OutputInclusion(Enum):
    """Whether an invalid sample, or a measured sample outside a plausibility limit (D92), is written to the processed file.

    EXCLUDED also means a suspicious sample is not used by later stages.
    The original file is never altered. How later stages treat a suspicious
    sample that is still PENDING_DECISION is not decided here (P23 onward).
    """

    PENDING_DECISION = "PENDING_DECISION"
    EXCLUDED = "EXCLUDED"
    KEPT = "KEPT"


class GapClass(Enum):
    """Classification of one gap. D49, as revised by F6.

    A classification also carries evidence and a confidence. See
    ``GapClassification``. No numeric confidence scale is approved.
    """

    OUTSIDE_INTERVAL = "OUTSIDE_INTERVAL"
    TOOL_ABSENCE_EVIDENCED = "TOOL_ABSENCE_EVIDENCED"
    DROPOUT = "DROPOUT"
    AMBIGUOUS = "AMBIGUOUS"


class GapFillOutcome(Enum):
    """What happened when a gap was considered for filling. D50, D52, D56.

    REFUSED carries a reason on ``GapFillRecord``. The samples stay ABSENT.
    """

    NOT_REQUESTED = "NOT_REQUESTED"
    FILLED = "FILLED"
    REFUSED = "REFUSED"
    NOT_FILLABLE = "NOT_FILLABLE"


class MethodOutcome(Enum):
    """Outcome of one method attempt in Stages 5–8. D86, UXC7."""

    RUN = "RUN"
    NOT_RUN_PRECONDITION = "NOT_RUN_PRECONDITION"
    NOT_RUN_NOT_REQUESTED = "NOT_RUN_NOT_REQUESTED"


class BlockType(Enum):
    """What kind of finding this is, and therefore what the user may do. D86, UXC2.

    Structural and export-conformance findings have no disposition that clears
    them. There is no force-export action.
    """

    INFORMATIONAL = "INFORMATIONAL"
    AMBIGUOUS_INTERPRETATION = "AMBIGUOUS_INTERPRETATION"
    RULE_DEPENDENT_WARNING = "RULE_DEPENDENT_WARNING"
    SCIENTIFIC_PRECONDITION_FAILURE = "SCIENTIFIC_PRECONDITION_FAILURE"
    STRUCTURAL_DATA_FAILURE = "STRUCTURAL_DATA_FAILURE"
    EXPORT_CONFORMANCE_FAILURE = "EXPORT_CONFORMANCE_FAILURE"


class IssueDisposition(Enum):
    """Disposition of one finding. D79, as revised by D86.

    UNDISPOSITIONED is non-terminal. Silence is not a completed disposition.
    DECLINED_BY_POLISH replaces D79's bare "Refused". DEFERRED is terminal,
    does not block export, and requires a non-empty reason.
    """

    UNDISPOSITIONED = "UNDISPOSITIONED"
    RESOLVED = "RESOLVED"
    ACCEPTED = "ACCEPTED"
    DEFERRED = "DEFERRED"
    DECLINED_BY_POLISH = "DECLINED_BY_POLISH"


class DependencyState(Enum):
    """Whether a result or decision is still current. D80.

    STALE carries a cause on ``DependencyRecord``.
    """

    CURRENT = "CURRENT"
    STALE = "STALE"


class StageStatus(Enum):
    """Status of one stage. D80, D86."""

    NOT_RUN = "NOT_RUN"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    STALE = "STALE"
    BLOCKED = "BLOCKED"
    REFUSED_STRUCTURAL = "REFUSED_STRUCTURAL"


class UserAction(Enum):
    """An action a person may take on a finding. D79, D86, UXC2.

    ``permitted_actions`` is the matrix. View-only block types permit none.
    REJECT_PROPOSAL records the rejection as the choice and yields RESOLVED
    (Appendix A.2). It is not a sample status and not a POLISH refusal.
    """

    ACKNOWLEDGE = "ACKNOWLEDGE"
    DEFER = "DEFER"
    CHOOSE_OPTION = "CHOOSE_OPTION"
    REJECT_PROPOSAL = "REJECT_PROPOSAL"
    MAP_EXPLICITLY = "MAP_EXPLICITLY"


# ---------------------------------------------------------------------------
# Records that carry the facts the enum value alone does not.
# ---------------------------------------------------------------------------


def _nonempty(text: str) -> bool:
    return isinstance(text, str) and text.strip() != ""


@dataclass(frozen=True)
class Disposition:
    """Disposition of one finding, with the reason where the schema requires one.

    Implements D79 and D86. A DEFERRED disposition cannot be constructed
    without a non-empty reason. That reason is listed in the export. DEFERRED
    is terminal and does not block export.
    """

    issue_disposition: IssueDisposition
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.issue_disposition, IssueDisposition):
            raise TypeError("issue_disposition must be an IssueDisposition")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a str")
        if self.issue_disposition is IssueDisposition.DEFERRED and not _nonempty(self.reason):
            raise ValueError("A DEFERRED disposition requires a non-empty reason (D79).")


@dataclass(frozen=True)
class Finding:
    """One classified finding, as the export gate reads it. D79, D86, UXC2.

    Parameters
    ----------
    finding_id:
        Stable id named in any export-permission reason.
    block_type:
        Classification that selects the permitted actions and the export effect.
    disposition:
        Current disposition. Structural and conformance findings stay open;
        a disposition on them does not clear the block.
    """

    finding_id: str
    block_type: BlockType
    disposition: Disposition

    def __post_init__(self) -> None:
        if not _nonempty(self.finding_id):
            raise ValueError("finding_id must be a non-empty string")
        if not isinstance(self.block_type, BlockType):
            raise TypeError("block_type must be a BlockType")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("disposition must be a Disposition")


@dataclass(frozen=True)
class DependencyRecord:
    """Freshness of one result or decision. D80.

    Parameters
    ----------
    dependency_id:
        Id named when this record blocks export.
    state:
        CURRENT or STALE.
    affects_export:
        Caller-supplied. This module does not decide which nodes affect
        exported claims. A STALE record blocks export only when this is true.
    cause:
        Required when ``state`` is STALE. Empty when the record is current
        is acceptable; a cause on a current record is ignored by the gate.
    """

    dependency_id: str
    state: DependencyState
    affects_export: bool
    cause: str = ""

    def __post_init__(self) -> None:
        if not _nonempty(self.dependency_id):
            raise ValueError("dependency_id must be a non-empty string")
        if not isinstance(self.state, DependencyState):
            raise TypeError("state must be a DependencyState")
        if type(self.affects_export) is not bool:
            raise TypeError("affects_export must be a bool")
        if not isinstance(self.cause, str):
            raise TypeError("cause must be a str")
        if self.state is DependencyState.STALE and not _nonempty(self.cause):
            raise ValueError("A STALE dependency requires a cause (D80).")


@dataclass(frozen=True)
class GapClassification:
    """A gap class plus the evidence and confidence it carries. D49.

    No numeric confidence scale is approved, so both fields are text the
    caller supplies. Neither may be empty: a classification that omits them
    is not the D49 record.
    """

    gap_class: GapClass
    evidence: str
    confidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.gap_class, GapClass):
            raise TypeError("gap_class must be a GapClass")
        if not _nonempty(self.evidence) or not _nonempty(self.confidence):
            raise ValueError("A gap classification carries evidence and a confidence (D49).")


@dataclass(frozen=True)
class GapFillRecord:
    """Outcome of considering one gap for filling. D50, D52, D56.

    REFUSED cannot be constructed without a non-empty reason. Samples in a
    refused gap stay ABSENT.
    """

    outcome: GapFillOutcome
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, GapFillOutcome):
            raise TypeError("outcome must be a GapFillOutcome")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a str")
        if self.outcome is GapFillOutcome.REFUSED and not _nonempty(self.reason):
            raise ValueError("A REFUSED gap fill requires a non-empty reason (D52, D56).")


@dataclass(frozen=True)
class ExportPermission:
    """Whether export may proceed, and the human-readable reasons if it may not.

    Implements D79, D86, D80. ``reasons`` names each blocking finding id, or
    each blocking dependency id, and why. An allowed export has an empty
    reason tuple. Deferral text stays on the Disposition for the export to
    list; it is not restated here.

    There is no field that overrides a structural or conformance block.
    """

    allowed: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be a bool")
        if type(self.reasons) is not tuple or not all(isinstance(item, str) for item in self.reasons):
            raise TypeError("reasons must be a tuple of str")


# ---------------------------------------------------------------------------
# Block type → permitted actions. Appendix A.3. View-only types map to empty.
# ---------------------------------------------------------------------------

_PERMITTED_ACTIONS: dict[BlockType, frozenset[UserAction]] = {
    BlockType.INFORMATIONAL: frozenset({UserAction.ACKNOWLEDGE, UserAction.DEFER}),
    BlockType.AMBIGUOUS_INTERPRETATION: frozenset(
        {UserAction.CHOOSE_OPTION, UserAction.REJECT_PROPOSAL, UserAction.DEFER}
    ),
    BlockType.RULE_DEPENDENT_WARNING: frozenset(
        {UserAction.MAP_EXPLICITLY, UserAction.ACKNOWLEDGE, UserAction.DEFER}
    ),
    BlockType.SCIENTIFIC_PRECONDITION_FAILURE: frozenset(),
    BlockType.STRUCTURAL_DATA_FAILURE: frozenset(),
    BlockType.EXPORT_CONFORMANCE_FAILURE: frozenset(),
}

# These three are the block types whose export effect is "until dispositioned".
# The other three are not cleared by a user disposition (A.3).
_DISPOSITIONABLE: frozenset[BlockType] = frozenset(
    {
        BlockType.INFORMATIONAL,
        BlockType.AMBIGUOUS_INTERPRETATION,
        BlockType.RULE_DEPENDENT_WARNING,
    }
)

_TERMINAL: frozenset[IssueDisposition] = frozenset(
    {
        IssueDisposition.RESOLVED,
        IssueDisposition.ACCEPTED,
        IssueDisposition.DEFERRED,
        IssueDisposition.DECLINED_BY_POLISH,
    }
)


def permitted_actions(block_type: BlockType) -> frozenset[UserAction]:
    """Return the user actions Appendix A.3 permits for this block type.

    Implements D86 and UXC2. Structural data failure, scientific precondition
    failure, and export conformance failure permit no disposition-clearing
    action. The returned set is frozen and empty for those three.

    Raises
    ------
    TypeError
        If ``block_type`` is not a ``BlockType``.
    """

    if not isinstance(block_type, BlockType):
        raise TypeError("block_type must be a BlockType")
    try:
        return _PERMITTED_ACTIONS[block_type]
    except KeyError as exc:
        raise TypeError(f"No action set recorded for {block_type!r}") from exc


def _blocking_reason(finding: Finding) -> str | None:
    """One denial sentence for this finding, or None if it does not block."""

    if finding.block_type is BlockType.STRUCTURAL_DATA_FAILURE:
        return (
            f"{finding.finding_id}: structural data failure blocks export; "
            "the workflow stops at that stage."
        )
    if finding.block_type is BlockType.EXPORT_CONFORMANCE_FAILURE:
        return (
            f"{finding.finding_id}: export conformance failure blocks export "
            "until this condition no longer holds."
        )
    if finding.block_type in _DISPOSITIONABLE and finding.disposition.issue_disposition not in _TERMINAL:
        return f"{finding.finding_id}: {finding.block_type.value} has no terminal disposition."
    return None


def export_permission(
    findings: Iterable[Finding],
    dependency_states: Iterable[DependencyRecord],
) -> ExportPermission:
    """Decide export from findings and dependency freshness. D79, D86, D80, UXC2.

    Rules, from Appendix A.3. Nothing in the signature overrides them.

    - A structural data failure blocks export whatever its disposition and
      whatever else is true.
    - An export conformance failure blocks until that finding is gone.
    - A dispositionable finding (informational, ambiguous interpretation,
      rule-dependent warning) blocks until it has a terminal disposition.
      DEFERRED is terminal and does not block.
    - A scientific precondition failure does not block export.
    - A STALE dependency blocks only when ``affects_export`` is true.

    Each denial reason names the finding id, or the dependency id, and why.
    Reasons follow the order of ``findings``, then the order of
    ``dependency_states``.

    Which dependencies affect exported claims is not decided here. The caller
    sets ``affects_export`` on each record.
    """

    reasons: list[str] = []
    for finding in findings:
        if not isinstance(finding, Finding):
            raise TypeError("findings must contain Finding instances")
        reason = _blocking_reason(finding)
        if reason is not None:
            reasons.append(reason)
    for record in dependency_states:
        if not isinstance(record, DependencyRecord):
            raise TypeError("dependency_states must contain DependencyRecord instances")
        if record.affects_export and record.state is DependencyState.STALE:
            reasons.append(
                f"{record.dependency_id}: dependency is STALE and affects export; cause: {record.cause}."
            )
    return ExportPermission(allowed=not reasons, reasons=tuple(reasons))
