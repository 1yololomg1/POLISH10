"""Stage 4: validity, the rule library, and the gap inventory. It does not rewrite samples.

Implements D11 (as revised by F15), D87, D44, D49 (as revised by F6), D50, D66,
D26, D92, D96, C2, C3, UXC4, UXC6, and UXC9. This is the keystone stage (D26).

Structure
---------
CONTRACT
    Stage 4 contract. alters_data is False. depends_on_stages is (0, 1, 2, 3).
legacy_depth_gap_label
    Feet-only label copied from petrophysics/constants.py. assess() does not
    call it. OD-15 is unanswered, and the labels are not D49 classes.
is_structurally_non_fillable
    D50. True for TOOL_ABSENCE_EVIDENCED and OUTSIDE_INTERVAL. No record
    field changes it.
assess
    Sample-status mask, validation outcome, measured interval, gap inventory,
    interval summaries, and findings. Source arrays stay read-only.
registered_stage
    The same assessment behind the stage runner.

OD-13 is unset. Shipped families have no tool group, no definition of
"shared", and no vertical resolution. Case 2 applies only when a caller
injects both a tool group and a definition of shared (tests). Case 3 cannot
apply. Every other interior gap is AMBIGUOUS. A missing answer is recorded
as cannot-apply. It is not replaced with a number.

The non-finite rule applies to every curve. It does not, by itself, make a
curve PASSED. A family rule applies only when the curve is eligible and its
unit is the rule's unit or converts to it exactly (OD-05, read_values and
require_unit). A GAPI rule is never applied to a CPS curve.

A sample outside a suspicious limit stays MEASURED, with output inclusion
PENDING_DECISION until the user answers keep or exclude (D92). How a later
stage treats a suspicious sample that is still PENDING_DECISION is not
decided here (P23 onward). The SP rule stays a log-level review flag. It
does not mark a sample.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy

from polish.model.dataset import Curve, Dataset
from polish.model.findings import DecisionScope, Evidence, FindingRecord, finding_key
from polish.model.ids import CurveRef
from polish.model.states import (
    BlockType,
    GapClass,
    GapClassification,
    GapFillOutcome,
    OutputInclusion,
    ProcessingEligibility,
    SampleStatus,
    StageStatus,
    UnitInterpretation,
    ValidationOutcome,
)
from polish.rules.loader import ValidityRule, validity_rules
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, RunContext, StageOutcome
from polish.stages.s00_ingest import IngestResult
from polish.stages.s01_depth import DepthAxisResult
from polish.stages.s02_identify import CurveIdentification, IdentificationResult
from polish.stages.s03_units import CurveUnits, UnitMismatch, UnitsResult, read_values, require_unit
from polish.tables.loader import ShippedTables, load_shipped_tables, load_unit_spellings

# Source: petrophysics/constants.py PetrophysicalConstants.GAP_SIZE_DEPTH_NORMALIZED.
# D51, feet only. The metric table is OD-15 and is unanswered. These numbers
# are not a gap class and are not a dropout cutoff.
_LEGACY_GAP_SIZE_DEPTH_FT = {
    "SMALL": 2.0,
    "MEDIUM": 5.0,
    "LARGE": 15.0,
    "VERY_LARGE": 50.0,
}

_NON_FINITE_ID = "non_finite_value"
_SCOPE_OPTIONS = (DecisionScope.THIS_ITEM.value, DecisionScope.LOCAL_RULE.value)
_KEEP_OR_EXCLUDE = ("KEEP", "EXCLUDE")
_QUESTION = "Exclude these samples from the processed output, or keep them?"

# OD-13 sentences. They are the record that the case cannot apply.
_NO_CO_TOOL = "no co-tool evidence available"
_SHARED_UNSET = "The meaning of shared is an OD-13 parameter with no shipped value, so case 2 cannot apply."
_NOT_SHARED = "The supplied definition of shared did not treat this gap as shared."
_DROPOUT_UNSET = "Vertical resolution is OD-13 and unset, so DROPOUT cannot apply."
_AMBIGUOUS_UNSET = "The ambiguous threshold is OD-13 and unset, so this interior gap is AMBIGUOUS."
_OUTSIDE_EVIDENCE = "The absent run is outside the measured interval, from the first non-absent sample to the last."
_OUTSIDE_CONFIDENCE = "Determined by the measured interval. No separate confidence scale applies."
_AMBIGUOUS_CONFIDENCE = "not assessed"

_ELIGIBILITY_TEXT = {
    ProcessingEligibility.NOT_ELIGIBLE_NO_FAMILY: (
        "no family",
        "A CITED, OBSERVED, or USER_DEFINED identification that assigns a shipped family.",
    ),
    ProcessingEligibility.NOT_ELIGIBLE_LOW_TIER: (
        "low tier",
        "An evidence tier of CITED, OBSERVED, or USER_DEFINED.",
    ),
    ProcessingEligibility.NOT_ELIGIBLE_ROLE: (
        "role",
        "A curve role that Stage 2 accepts for processing.",
    ),
    ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED: (
        "unit unresolved",
        "A recognised unit in a dimension the family accepts.",
    ),
}


def legacy_depth_gap_label(gap_points: int, depth_spacing_ft: float) -> str:
    """Feet-only gap label copied from ``PetrophysicalConstants.classify_gap_by_depth``.

    Source: petrophysics/constants.py ``PetrophysicalConstants.classify_gap_by_depth``.
    The thresholds are ``GAP_SIZE_DEPTH_NORMALIZED`` in that class. The copy
    keeps the original comparisons, including that a gap of ``LARGE`` feet or
    more returns ``VERY_LARGE`` and the 50-foot entry is not consulted.

    assess() does not call this. The labels are not D49 classes. The table is
    feet-only, and OD-15 (a metric table) is unanswered. Wiring it in would
    be a guessed cutoff.

    Parameters
    ----------
    gap_points:
        Number of samples in the gap, as the legacy function counted them.
    depth_spacing_ft:
        Spacing in feet per sample, as the legacy function required.
    """

    if type(gap_points) is not int or isinstance(gap_points, bool):
        raise TypeError("gap_points must be an int.")
    if isinstance(depth_spacing_ft, bool) or not isinstance(depth_spacing_ft, (int, float)):
        raise TypeError("depth_spacing_ft must be a real number of feet.")
    gap_depth_ft = gap_points * float(depth_spacing_ft)
    if gap_depth_ft < _LEGACY_GAP_SIZE_DEPTH_FT["SMALL"]:
        return "SMALL"
    if gap_depth_ft < _LEGACY_GAP_SIZE_DEPTH_FT["MEDIUM"]:
        return "MEDIUM"
    if gap_depth_ft < _LEGACY_GAP_SIZE_DEPTH_FT["LARGE"]:
        return "LARGE"
    return "VERY_LARGE"


def is_structurally_non_fillable(gap_class: GapClass) -> bool:
    """Whether this class is non-fillable as a matter of structure. D50.

    TOOL_ABSENCE_EVIDENCED and OUTSIDE_INTERVAL are non-fillable. Nothing on
    a gap record changes that. Stage 4 still stores fill_outcome
    NOT_REQUESTED; NOT_FILLABLE belongs to a later fill request.
    """

    if not isinstance(gap_class, GapClass):
        raise TypeError("gap_class must be a GapClass.")
    return gap_class in (GapClass.TOOL_ABSENCE_EVIDENCED, GapClass.OUTSIDE_INTERVAL)


@dataclass(frozen=True)
class MeasuredInterval:
    """First and last non-absent sample of one curve. D49.

    Indexes are inclusive. Depths are the grid values at those indexes, in
    index order.
    """

    sample_start: int
    sample_last: int
    depth_at_start: float
    depth_at_last: float


@dataclass(frozen=True)
class SampleMark:
    """Status of one sample, stored beside the array. D44, D66.

    The source value is not in this record. INVALID carries
    PENDING_DECISION and the rule ids that marked it. A measured sample
    outside a plausibility limit also carries PENDING_DECISION and no rule
    id (D92). ABSENT carries neither. EXCLUDED and KEPT are not set in this
    stage. How a later stage treats PENDING_DECISION on a suspicious sample
    is not decided here (P23 onward).
    """

    status: SampleStatus
    output_inclusion: OutputInclusion | None
    invalid_rules: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, SampleStatus):
            raise TypeError("status must be a SampleStatus.")
        if self.status is SampleStatus.INVALID:
            if self.output_inclusion is not OutputInclusion.PENDING_DECISION:
                raise ValueError("An INVALID sample stays PENDING_DECISION until the user chooses.")
            if not self.invalid_rules or not all(isinstance(rule_id, str) and rule_id for rule_id in self.invalid_rules):
                raise ValueError("An INVALID sample names the rule that marked it.")
            return
        if self.status is SampleStatus.MEASURED and self.output_inclusion is OutputInclusion.PENDING_DECISION:
            if self.invalid_rules:
                raise ValueError("A suspicious sample stays MEASURED and does not carry an invalid-rule id.")
            return
        if self.status not in (SampleStatus.MEASURED, SampleStatus.ABSENT):
            raise ValueError("Stage 4 sets MEASURED, INVALID, or ABSENT.")
        if self.output_inclusion is not None or self.invalid_rules:
            raise ValueError("Only an invalid or suspicious sample carries output inclusion.")


@dataclass(frozen=True)
class GapRecord:
    """One contiguous absent run. D49, D50.

    ``sample_end`` is exclusive. ``fill_outcome`` is NOT_REQUESTED. Whether
    the class is structurally non-fillable is ``is_structurally_non_fillable``,
    not a field of this record.
    """

    curve: CurveRef
    mnemonic: str
    sample_start: int
    sample_end: int
    depth_at_start: float
    depth_at_end: float
    classification: GapClassification
    fill_outcome: GapFillOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.curve, CurveRef):
            raise TypeError("curve must be a CurveRef.")
        if not isinstance(self.mnemonic, str) or self.mnemonic.strip() == "":
            raise ValueError("A gap names its curve.")
        if type(self.sample_start) is not int or type(self.sample_end) is not int:
            raise TypeError("sample span indexes must be ints.")
        if self.sample_end <= self.sample_start:
            raise ValueError("A gap needs a non-empty sample span.")
        if not isinstance(self.classification, GapClassification):
            raise TypeError("classification must be a GapClassification.")
        if self.fill_outcome is not GapFillOutcome.NOT_REQUESTED:
            raise ValueError("Stage 4 sets fill_outcome to NOT_REQUESTED. NOT_FILLABLE is not set here.")


@dataclass(frozen=True)
class IntervalSpan:
    """One merged span for rendering. UXC6. No pixel width is stored.

    ``status`` is MEASURED, INVALID, or ABSENT. ``gap_class`` is set only
    for an absent span.
    """

    status: SampleStatus
    gap_class: GapClass | None
    sample_start: int
    sample_end: int
    depth_at_start: float
    depth_at_end: float

    def __post_init__(self) -> None:
        if self.status not in (SampleStatus.MEASURED, SampleStatus.INVALID, SampleStatus.ABSENT):
            raise ValueError("An interval span is measured, invalid, or absent.")
        if self.status is SampleStatus.ABSENT:
            if not isinstance(self.gap_class, GapClass):
                raise ValueError("An absent span names its gap class.")
        elif self.gap_class is not None:
            raise ValueError("Only an absent span names a gap class.")
        if type(self.sample_start) is not int or type(self.sample_end) is not int or self.sample_end <= self.sample_start:
            raise ValueError("A span needs a non-empty sample range.")


@dataclass(frozen=True)
class CurveValidity:
    """Stage 4 record for one curve. The sample arrays are not in here. C5.

    ``rule_set_version`` is set only for PASSED. The non-finite rule alone
    does not set it. ``ineligibility_reason`` is set only for NOT_ELIGIBLE.
    """

    curve: CurveRef
    mnemonic: str
    family_id: str | None
    tool_group: str | None
    outcome: ValidationOutcome
    rule_set_version: str | None
    ineligibility_reason: str | None
    eligibility_evidence: str | None
    family_rules_applied: tuple[str, ...]
    samples: tuple[SampleMark, ...]
    measured_interval: MeasuredInterval | None
    gaps: tuple[GapRecord, ...]
    summaries: tuple[IntervalSpan, ...]

    @property
    def non_finite_count(self) -> int:
        """Samples marked INVALID by the non-finite rule."""

        return sum(1 for sample in self.samples if _NON_FINITE_ID in sample.invalid_rules)


@dataclass(frozen=True)
class ValidityResult:
    """What Stage 4 records. Source arrays are not in this result. D24, C5."""

    status: StageStatus
    findings: tuple[FindingRecord, ...]
    curves: tuple[CurveValidity, ...]
    library_version: str


@dataclass(frozen=True)
class GroupMember:
    """One curve in a tool group, for a caller-supplied definition of shared.

    OD-13 has no shipped definition. assess() builds these only when a tool
    group is known and a definition was injected.
    """

    mnemonic: str
    family_id: str
    curve: CurveRef
    absent_indexes: tuple[int, ...]


@dataclass(frozen=True)
class InteriorGap:
    """An interior absent run handed to a definition of shared. OD-13."""

    mnemonic: str
    family_id: str
    curve: CurveRef
    tool_group: str
    sample_start: int
    sample_end: int


@dataclass(frozen=True)
class SharedMatch:
    """The injected definition says this gap is shared.

    ``overlap_measure`` and ``confidence`` are text the definition supplies.
    No numeric confidence scale is approved, so assess() does not invent one.
    """

    mnemonics: tuple[str, ...]
    overlap_measure: str
    confidence: str

    def __post_init__(self) -> None:
        if type(self.mnemonics) is not tuple or not self.mnemonics:
            raise ValueError("A shared match names the curves that share the gap.")
        if not all(isinstance(name, str) and name.strip() for name in self.mnemonics):
            raise ValueError("A shared match names the curves that share the gap.")
        if not isinstance(self.overlap_measure, str) or self.overlap_measure.strip() == "":
            raise ValueError("A shared match states an overlap measure.")
        if not isinstance(self.confidence, str) or self.confidence.strip() == "":
            raise ValueError("A shared match states a confidence.")


SharedDefinition = Callable[[InteriorGap, tuple[GroupMember, ...]], SharedMatch | None]


@dataclass
class _Hit:
    """One rule violation gathered before findings are written."""

    rule: ValidityRule
    indexes: tuple[int, ...]
    outside_count: int
    non_absent_count: int


@dataclass
class _Prepared:
    """Per-curve facts needed before gaps can be classified together."""

    curve: Curve
    identification: CurveIdentification
    units: CurveUnits
    samples: tuple[SampleMark, ...]
    absent_indexes: tuple[int, ...]
    interval: MeasuredInterval | None
    runs: tuple[tuple[int, int], ...]
    family_rules_applied: tuple[str, ...]
    hits: tuple[_Hit, ...]
    tool_group: str | None
    had_family_candidate: bool
    unit_blocked: bool


_STAGE4_CHECKS: tuple[tuple[str, BlockType], ...] = (
    ("invalid_samples", BlockType.AMBIGUOUS_INTERPRETATION),
    ("log_level_flag", BlockType.INFORMATIONAL),
    ("suspicious_samples", BlockType.AMBIGUOUS_INTERPRETATION),
    ("not_validated", BlockType.INFORMATIONAL),
    ("tool_absence_gap", BlockType.INFORMATIONAL),
)
CHECKS = frozenset(name for name, _block in _STAGE4_CHECKS)
_CHECK_BLOCK = dict(_STAGE4_CHECKS)


class _Ledger:
    """Findings for one assessment. An id is the check and what it is about. D96."""

    def __init__(self, ingest: IngestResult) -> None:
        self._read_at = ingest.source_file.read_at
        self.items: list[FindingRecord] = []

    def add(
        self,
        check: str,
        identity: tuple[tuple[str, str], ...],
        message: str,
        ref: CurveRef,
        fields: dict[str, str | int | float | bool],
    ) -> None:
        """Record one finding. The id is not this finding's position in the run. D96."""

        if check not in CHECKS:
            raise ValueError(f"Check {check!r} is not declared for this stage.")
        block = _CHECK_BLOCK[check]
        finding_id = finding_key(4, check, identity)
        if any(item.id == finding_id for item in self.items):
            raise ValueError(
                f"Finding {finding_id} for check {check!r} and identity {identity!r} is already in this run."
            )
        self.items.append(
            FindingRecord(
                id=finding_id,
                stage=4,
                check=check,
                identity=identity,
                block_type=block,
                subject_refs=(ref,),
                evidence=Evidence(block.value, tuple(sorted(fields.items()))),
                message=message,
                created_at=self._read_at,
            )
        )


def _absent_runs(absent: numpy.ndarray) -> tuple[tuple[int, int], ...]:
    """Half-open index spans of contiguous absent samples.

    Adapted from advanced_preprocessing_system10.py
    ``AdvancedPreprocessingApplication._count_consecutive_missing``. That
    method returned only the lengths. ``update_data_display`` in the same
    class then split those lengths into a geological label and a data-error
    label and used the split only to adjust a quality string. The interval
    was not kept, and the label did not decide whether the gap stayed empty.
    This copy keeps the index span.
    """

    runs: list[tuple[int, int]] = []
    start: int | None = None
    count = int(absent.size)
    for index in range(count):
        if bool(absent[index]):
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, count))
    return tuple(runs)


def _measured_interval(absent: numpy.ndarray, grid: numpy.ndarray) -> MeasuredInterval | None:
    present = numpy.flatnonzero(~absent)
    if int(present.size) == 0:
        return None
    first = int(present[0])
    last = int(present[-1])
    return MeasuredInterval(first, last, float(grid[first]), float(grid[last]))


def _is_outside(start: int, end: int, interval: MeasuredInterval | None) -> bool:
    if interval is None:
        return True
    if end <= interval.sample_start:
        return True
    if start > interval.sample_last:
        return True
    return False


def _check_library(rules: tuple[ValidityRule, ...], shipped: ShippedTables) -> str:
    spellings = load_unit_spellings()
    canonical = {row.canonical_unit for row in spellings.spellings}
    version = ""
    for rule in rules:
        if version == "":
            version = rule.library_version
        elif rule.library_version != version:
            raise ValueError("Validity rules do not share one library version.")
        if rule.family is not None and shipped.family(rule.family) is None:
            raise ValueError(f"Validity rule {rule.id} names unknown family {rule.family!r}.")
        if rule.canonical_unit is not None and rule.canonical_unit not in canonical:
            raise ValueError(f"Validity rule {rule.id} unit {rule.canonical_unit!r} is not a canonical spelling.")
        for unit in rule.accepted_units:
            if unit not in canonical:
                raise ValueError(f"Validity rule {rule.id} unit {unit!r} is not a canonical spelling.")
    if version == "":
        raise ValueError("The validity library has no version.")
    return version


def _tool_group(
    family_id: str | None,
    shipped: ShippedTables,
    injected: Mapping[str, str] | None,
) -> str | None:
    if family_id is None:
        return None
    if injected is not None and family_id in injected:
        return injected[family_id]
    family = shipped.family(family_id)
    if family is None:
        raise ValueError(f"Family {family_id!r} is not in the shipped tables.")
    return family.tool_group


def _gapi_on_cps(rule: ValidityRule, curve_units: CurveUnits) -> bool:
    """A GAPI rule does not apply to a CPS curve. OD-05 case 4, D87."""

    if rule.canonical_unit != "GAPI":
        return False
    return curve_units.canonical_unit == "CPS" or curve_units.dimension == "count_rate"


def _view_for_rule(dataset: Dataset, curve_units: CurveUnits, rule: ValidityRule):
    """Values in the rule's unit, or None when the rule does not apply.

    None is "no applicable rule", not a failed sample. UnitMismatch is the
    recorded refusal from Stage 3; it is not swallowed as a pass.
    """

    if rule.canonical_unit is None:
        return None
    if curve_units.processing_eligibility is not ProcessingEligibility.ELIGIBLE:
        return None
    if curve_units.outcome is not UnitInterpretation.RECOGNISED:
        return None
    if _gapi_on_cps(rule, curve_units):
        return None
    try:
        return require_unit(read_values(dataset, curve_units, rule.canonical_unit), rule.canonical_unit)
    except UnitMismatch:
        return None


def _outside_finite(values: numpy.ndarray, minimum: float, maximum: float) -> numpy.ndarray:
    """Finite samples strictly outside inclusive bounds. Non-finite values are excluded."""

    finite = numpy.isfinite(values)
    return finite & ((values < minimum) | (values > maximum))


def _marks(
    stored: numpy.ndarray,
    hard_indexes: dict[int, list[str]],
    warning_indexes: frozenset[int],
) -> tuple[SampleMark, ...]:
    """INVALID for a hard limit. MEASURED and pending for a suspicious limit. D92."""

    marks: list[SampleMark] = []
    for index, value in enumerate(stored):
        number = float(value)
        if numpy.isnan(number):
            marks.append(SampleMark(SampleStatus.ABSENT, None, ()))
            continue
        rules = tuple(hard_indexes.get(index, ()))
        if numpy.isinf(number):
            rules = (_NON_FINITE_ID,) + tuple(rule_id for rule_id in rules if rule_id != _NON_FINITE_ID)
        if rules:
            marks.append(SampleMark(SampleStatus.INVALID, OutputInclusion.PENDING_DECISION, rules))
        elif index in warning_indexes:
            marks.append(SampleMark(SampleStatus.MEASURED, OutputInclusion.PENDING_DECISION, ()))
        else:
            marks.append(SampleMark(SampleStatus.MEASURED, None, ()))
    return tuple(marks)


def _prepare_curve(
    dataset: Dataset,
    curve: Curve,
    identification: CurveIdentification,
    curve_units: CurveUnits,
    rules: tuple[ValidityRule, ...],
    shipped: ShippedTables,
    injected: Mapping[str, str] | None,
) -> _Prepared:
    stored = curve.values
    absent = numpy.isnan(stored)
    grid = dataset.grid.values
    family_rules = tuple(rule for rule in rules if rule.family is not None and rule.family == identification.family_id)
    applied: list[str] = []
    hits: list[_Hit] = []
    hard_indexes: dict[int, list[str]] = {}
    warning_indexes: set[int] = set()
    unit_blocked = False
    for rule in family_rules:
        if curve_units.processing_eligibility is not ProcessingEligibility.ELIGIBLE:
            continue
        view = _view_for_rule(dataset, curve_units, rule)
        if view is None:
            unit_blocked = True
            continue
        applied.append(rule.id)
        bounds = rule.bound_values
        if bounds.minimum is None or bounds.maximum is None:
            raise ValueError(f"{rule.id} has no bounds.")
        if rule.log_level:
            if rule.violation_effect != "WARNING":
                raise ValueError(f"{rule.id} is log-level and its effect is not WARNING.")
            non_absent = ~numpy.isnan(view.values)
            outside = non_absent & ((view.values < bounds.minimum) | (view.values > bounds.maximum))
            outside_count = int(numpy.count_nonzero(outside))
            non_absent_count = int(numpy.count_nonzero(non_absent))
            fraction = bounds.outside_fraction_more_than
            if fraction is None:
                raise ValueError(f"{rule.id} is log-level and has no fraction.")
            # More than the rational. Exactly 1/5 is not more than 1/5.
            flagged = (
                non_absent_count > 0
                and outside_count * fraction.denominator > fraction.numerator * non_absent_count
            )
            if flagged:
                hits.append(_Hit(rule, (), outside_count, non_absent_count))
            continue
        outside = _outside_finite(view.values, bounds.minimum, bounds.maximum)
        indexes = tuple(int(index) for index in numpy.flatnonzero(outside))
        if rule.violation_effect == "INVALID":
            for index in indexes:
                hard_indexes.setdefault(index, []).append(rule.id)
        elif rule.violation_effect == "WARNING":
            if indexes:
                hits.append(_Hit(rule, indexes, len(indexes), 0))
                warning_indexes.update(indexes)
        else:
            raise ValueError(f"{rule.id} effect {rule.violation_effect} is not applied to samples.")
    samples = _marks(stored, hard_indexes, frozenset(warning_indexes))
    return _Prepared(
        curve=curve,
        identification=identification,
        units=curve_units,
        samples=samples,
        absent_indexes=tuple(int(index) for index in numpy.flatnonzero(absent)),
        interval=_measured_interval(absent, grid),
        runs=_absent_runs(absent),
        family_rules_applied=tuple(applied),
        hits=tuple(hits),
        tool_group=_tool_group(identification.family_id, shipped, injected),
        had_family_candidate=len(family_rules) > 0,
        unit_blocked=unit_blocked,
    )


def _members(prepared: tuple[_Prepared, ...], tool_group: str) -> tuple[GroupMember, ...]:
    members: list[GroupMember] = []
    for item in prepared:
        if item.tool_group != tool_group or item.identification.family_id is None:
            continue
        members.append(
            GroupMember(
                mnemonic=item.identification.mnemonic,
                family_id=item.identification.family_id,
                curve=item.curve.ref,
                absent_indexes=item.absent_indexes,
            )
        )
    return tuple(members)


def _classify_run(
    item: _Prepared,
    start: int,
    end: int,
    prepared: tuple[_Prepared, ...],
    shared: SharedDefinition | None,
) -> tuple[GapClassification, SharedMatch | None]:
    if _is_outside(start, end, item.interval):
        return GapClassification(GapClass.OUTSIDE_INTERVAL, _OUTSIDE_EVIDENCE, _OUTSIDE_CONFIDENCE), None
    if item.interval is None or not (start > item.interval.sample_start and (end - 1) < item.interval.sample_last):
        raise ValueError("Absent run is neither outside the measured interval nor strictly interior.")
    if item.tool_group is None:
        note = _NO_CO_TOOL
    elif shared is None:
        note = _SHARED_UNSET
    else:
        if item.identification.family_id is None:
            raise ValueError("A tool group is set on a curve with no family.")
        gap = InteriorGap(
            item.identification.mnemonic,
            item.identification.family_id,
            item.curve.ref,
            item.tool_group,
            start,
            end,
        )
        match = shared(gap, _members(prepared, item.tool_group))
        if match is None:
            note = _NOT_SHARED
        else:
            if not isinstance(match, SharedMatch):
                raise TypeError("A definition of shared returns a SharedMatch or None.")
            listed = ", ".join(match.mnemonics)
            evidence = (
                f"Curves {listed} share this interior gap ({match.overlap_measure}). "
                "This is evidence consistent with tool absence, not proof."
            )
            return GapClassification(GapClass.TOOL_ABSENCE_EVIDENCED, evidence, match.confidence), match
    evidence = f"{note} {_DROPOUT_UNSET} {_AMBIGUOUS_UNSET}"
    return GapClassification(GapClass.AMBIGUOUS, evidence, _AMBIGUOUS_CONFIDENCE), None


def _gaps_for(
    item: _Prepared,
    prepared: tuple[_Prepared, ...],
    shared: SharedDefinition | None,
    grid: numpy.ndarray,
) -> tuple[tuple[GapRecord, SharedMatch | None], ...]:
    built: list[tuple[GapRecord, SharedMatch | None]] = []
    for start, end in item.runs:
        classification, match = _classify_run(item, start, end, prepared, shared)
        record = GapRecord(
            curve=item.curve.ref,
            mnemonic=item.identification.mnemonic,
            sample_start=start,
            sample_end=end,
            depth_at_start=float(grid[start]),
            depth_at_end=float(grid[end - 1]),
            classification=classification,
            fill_outcome=GapFillOutcome.NOT_REQUESTED,
        )
        built.append((record, match))
    return tuple(built)


def _summaries(
    samples: tuple[SampleMark, ...],
    gaps: tuple[GapRecord, ...],
    grid: numpy.ndarray,
) -> tuple[IntervalSpan, ...]:
    classes: list[GapClass | None] = [None] * len(samples)
    for gap in gaps:
        for index in range(gap.sample_start, gap.sample_end):
            if classes[index] is not None:
                raise ValueError("An absent sample is in two gaps.")
            classes[index] = gap.classification.gap_class
    for index, sample in enumerate(samples):
        if sample.status is SampleStatus.ABSENT and classes[index] is None:
            raise ValueError("An absent sample has no gap class.")
        if sample.status is not SampleStatus.ABSENT and classes[index] is not None:
            raise ValueError("A present sample was given a gap class.")
    spans: list[IntervalSpan] = []
    start = 0
    for index in range(1, len(samples) + 1):
        boundary = index == len(samples)
        if not boundary and samples[index].status is samples[start].status and classes[index] is classes[start]:
            continue
        spans.append(
            IntervalSpan(
                status=samples[start].status,
                gap_class=classes[start],
                sample_start=start,
                sample_end=index,
                depth_at_start=float(grid[start]),
                depth_at_end=float(grid[index - 1]),
            )
        )
        start = index
    return tuple(spans)


def _why_ineligible(item: _Prepared) -> tuple[str, str]:
    eligibility = item.units.processing_eligibility
    stated = _ELIGIBILITY_TEXT.get(eligibility)
    if stated is not None:
        return stated
    if eligibility is not ProcessingEligibility.ELIGIBLE:
        raise ValueError(f"No ineligibility text for {eligibility.value}.")
    if item.had_family_candidate and item.unit_blocked:
        return (
            "no exact conversion",
            "A unit equal to the rule's unit, or an exact conversion to it. A GAPI rule is not applied to a CPS curve.",
        )
    if not item.had_family_candidate:
        return (
            "no family rule",
            "A versioned family rule for this family and unit in the rule library.",
        )
    raise ValueError("A family rule applied; the curve is not ineligible.")


def _outcome(
    item: _Prepared,
    library_version: str,
) -> tuple[ValidationOutcome, str | None, str | None, str | None]:
    """Outcome ladder. A hard sample wins. The non-finite rule alone is not a pass."""

    invalid = any(sample.status is SampleStatus.INVALID for sample in item.samples)
    if invalid:
        return ValidationOutcome.HARD_VIOLATION_EXCLUDED, None, None, None
    if item.hits:
        return ValidationOutcome.PLAUSIBILITY_WARNING, None, None, None
    if item.family_rules_applied:
        return ValidationOutcome.PASSED, library_version, None, None
    reason, evidence = _why_ineligible(item)
    return ValidationOutcome.NOT_ELIGIBLE, None, reason, evidence


def _number_text(value: float) -> str:
    """A bound as written, without a trailing decimal on a whole number."""

    if value.is_integer():
        return str(int(value))
    return format(value, "g")


def _fraction_text(rule: ValidityRule) -> str:
    fraction = rule.bound_values.outside_fraction_more_than
    if fraction is None:
        raise ValueError(f"{rule.id} has no log-level fraction.")
    return f"{fraction.numerator}/{fraction.denominator}"


def _required_bound(rule: ValidityRule, which: str) -> float:
    value = rule.bound_values.minimum if which == "minimum" else rule.bound_values.maximum
    if which not in ("minimum", "maximum"):
        raise ValueError(f"Unknown bound {which!r}.")
    if value is None:
        raise ValueError(f"{rule.id} has no {which}.")
    return value


def _depths(grid: numpy.ndarray, indexes: tuple[int, ...]) -> str:
    return json.dumps([float(grid[index]) for index in indexes])


def _add_findings(
    ledger: _Ledger,
    item: _Prepared,
    outcome: ValidationOutcome,
    reason: str | None,
    evidence_text: str | None,
    gaps: tuple[tuple[GapRecord, SharedMatch | None], ...],
    grid: numpy.ndarray,
    non_finite: ValidityRule,
) -> None:
    mnemonic = item.identification.mnemonic
    family_id = item.identification.family_id if item.identification.family_id is not None else ""
    ref = item.curve.ref
    by_rule: dict[str, list[int]] = {}
    for index, sample in enumerate(item.samples):
        for rule_id in sample.invalid_rules:
            by_rule.setdefault(rule_id, []).append(index)
    for rule_id, indexes in by_rule.items():
        rule = non_finite if rule_id == non_finite.id else next(
            (hit.rule for hit in item.hits if hit.rule.id == rule_id),
            None,
        )
        if rule is None:
            rule = _rule_by_id(rule_id)
        fields: dict[str, str | int | float | bool] = {
            "rule_id": rule.id,
            "library_version": rule.library_version,
            "depths": _depths(grid, tuple(indexes)),
            "sample_indexes": json.dumps(indexes),
            "sample_count": len(indexes),
            "question": _QUESTION,
            "output_inclusion": OutputInclusion.PENDING_DECISION.value,
            "violation_effect": rule.violation_effect,
            "scope_options": json.dumps(list(_SCOPE_OPTIONS)),
            "options": json.dumps(list(_KEEP_OR_EXCLUDE)),
            "mnemonic": mnemonic,
            "family_id": family_id,
        }
        if rule.bound_values.minimum is not None and rule.bound_values.maximum is not None:
            fields["minimum"] = rule.bound_values.minimum
            fields["maximum"] = rule.bound_values.maximum
        name = "non-finite value" if rule.id == _NON_FINITE_ID else rule.id
        message = (
            f"Curve {mnemonic} has {len(indexes)} samples marked INVALID by {name}. "
            "They stay visible. Exclude these samples from the processed output, or keep them? "
            "Scope is this well or all wells. Nothing is excluded until that choice is recorded. "
            "Nothing is clipped. The original file is not altered. Every affected depth is listed."
        )
        ledger.add(
            "invalid_samples",
            (("column", str(ref.column_index)), ("mnemonic", mnemonic), ("rule_id", rule.id)),
            message,
            ref,
            fields,
        )
    for hit in item.hits:
        low = _required_bound(hit.rule, "minimum")
        high = _required_bound(hit.rule, "maximum")
        if hit.rule.log_level:
            # The library token is MV. The review sentence uses mV (D92, 18 Sept).
            unit = "mV" if hit.rule.canonical_unit == "MV" else hit.rule.canonical_unit
            fields = {
                "rule_id": hit.rule.id,
                "library_version": hit.rule.library_version,
                "sample_count": hit.outside_count,
                "outside_count": hit.outside_count,
                "non_absent_count": hit.non_absent_count,
                "outside_fraction_more_than": _fraction_text(hit.rule),
                "log_level": True,
                "minimum": low,
                "maximum": high,
                "mnemonic": mnemonic,
                "family_id": family_id,
            }
            message = (
                f"Curve {mnemonic}: more than {_fraction_text(hit.rule)} of its samples are outside "
                f"{_number_text(low)} to {_number_text(high)} {unit}. The log is flagged for review."
            )
            ledger.add(
                "log_level_flag",
                (("column", str(ref.column_index)), ("mnemonic", mnemonic), ("rule_id", hit.rule.id)),
                message,
                ref,
                fields,
            )
            continue
        unit = hit.rule.canonical_unit if hit.rule.canonical_unit is not None else ""
        fields = {
            "rule_id": hit.rule.id,
            "library_version": hit.rule.library_version,
            "depths": _depths(grid, hit.indexes),
            "sample_indexes": json.dumps(list(hit.indexes)),
            "sample_count": len(hit.indexes),
            "output_inclusion": OutputInclusion.PENDING_DECISION.value,
            "violation_effect": hit.rule.violation_effect,
            "scope_options": json.dumps(list(_SCOPE_OPTIONS)),
            "options": json.dumps(list(_KEEP_OR_EXCLUDE)),
            "minimum": low,
            "maximum": high,
            "mnemonic": mnemonic,
            "family_id": family_id,
        }
        message = (
            f"Curve {mnemonic} has {len(hit.indexes)} samples outside "
            f"{_number_text(low)} to {_number_text(high)} {unit} (suspicious). "
            "Keep them (used by later stages, stays flagged) or exclude them "
            "(not used by later stages, still shown, listed in the export)? "
            "A reason is required. Scope: this well or all wells."
        )
        ledger.add(
            "suspicious_samples",
            (("column", str(ref.column_index)), ("mnemonic", mnemonic), ("rule_id", hit.rule.id)),
            message,
            ref,
            fields,
        )
    if outcome is ValidationOutcome.NOT_ELIGIBLE:
        if reason is None or evidence_text is None:
            raise ValueError("A NOT_ELIGIBLE curve needs a reason and the evidence that would make it eligible.")
        ledger.add(
            "not_validated",
            (("column", str(ref.column_index)), ("mnemonic", mnemonic)),
            (
                f"Curve {mnemonic} was not automatically validated: {reason}. "
                f"Evidence that would make it eligible: {evidence_text}"
            ),
            ref,
            {
                "mnemonic": mnemonic,
                "family_id": family_id,
                "reason": reason,
                "eligibility_evidence": evidence_text,
            },
        )
    for gap, match in gaps:
        if gap.classification.gap_class is not GapClass.TOOL_ABSENCE_EVIDENCED or match is None:
            continue
        ledger.add(
            "tool_absence_gap",
            (
                ("column", str(ref.column_index)),
                ("mnemonic", mnemonic),
                ("sample_end", str(gap.sample_end)),
                ("sample_start", str(gap.sample_start)),
            ),
            (
                f"Curve {mnemonic} has an interior gap that is evidence consistent with tool absence, not proof. "
                f"Not fillable. {gap.classification.evidence} Confidence: {gap.classification.confidence}."
            ),
            ref,
            {
                "mnemonic": mnemonic,
                "family_id": family_id,
                "gap_class": gap.classification.gap_class.value,
                "gap_evidence": gap.classification.evidence,
                "confidence": gap.classification.confidence,
                "not_fillable": True,
                "sharing_curves": json.dumps(list(match.mnemonics)),
                "overlap_measure": match.overlap_measure,
                "depth_at_start": gap.depth_at_start,
                "depth_at_end": gap.depth_at_end,
                "sample_start": gap.sample_start,
                "sample_end": gap.sample_end,
            },
        )


def _rule_by_id(rule_id: str) -> ValidityRule:
    for rule in validity_rules():
        if rule.id == rule_id:
            return rule
    raise ValueError(f"INVALID sample names unknown rule {rule_id!r}.")


def _check_injection(tool_groups: Mapping[str, str] | None, shared: SharedDefinition | None) -> None:
    if tool_groups is not None:
        if not isinstance(tool_groups, Mapping):
            raise TypeError("tool_groups must be a mapping of family id to tool group, or None.")
        for family_id, name in tool_groups.items():
            if not isinstance(family_id, str) or family_id.strip() == "":
                raise ValueError("An injected tool group names a family id.")
            if not isinstance(name, str) or name.strip() == "":
                raise ValueError("An injected tool group needs a non-empty name.")
    if shared is not None and not callable(shared):
        raise TypeError("shared must be a callable or None.")


def assess(
    ingest: IngestResult,
    depth: DepthAxisResult,
    identification: IdentificationResult,
    units: UnitsResult,
    *,
    tool_groups: Mapping[str, str] | None = None,
    shared: SharedDefinition | None = None,
) -> ValidityResult:
    """Assess every curve. D11, D87, D44, D49, D50, D66, D96, C2, C3, UXC6, UXC9.

    Does not write the dataset. Declared nulls (NaN) are ABSENT. A finite
    value equal to an undeclared sentinel stays a number. ±inf is INVALID
    under the non-finite rule. A plausibility breach leaves the sample
    MEASURED. Nothing is clipped and nothing is dropped from a file.

    ``tool_groups`` and ``shared`` are unset on the shipped path (None).
    None means OD-13 is unanswered: case 2 cannot apply unless both are
    injected together, and case 3 cannot apply at all. They are not default
    cutoffs.

    Parameters
    ----------
    ingest:
        The Stage 0 result whose dataset is read.
    depth:
        A completed Stage 1 result. The grid already on the dataset is the
        depth used for spans.
    identification:
        A completed Stage 2 result, one row per curve.
    units:
        A completed Stage 3 result, one row per curve.
    tool_groups:
        Optional family-id to tool-group names for a test. Shipped families
        stay null when this is None or omits the family.
    shared:
        Optional definition of a shared interior gap. Unset means case 2
        cannot apply.

    Returns
    -------
    ValidityResult
        One CurveValidity per curve, in column order, and the findings.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If an earlier stage did not complete for every curve.
    """

    if not isinstance(ingest, IngestResult):
        raise TypeError("ingest must be an IngestResult.")
    if not isinstance(depth, DepthAxisResult):
        raise TypeError("depth must be a DepthAxisResult.")
    if not isinstance(identification, IdentificationResult):
        raise TypeError("identification must be an IdentificationResult.")
    if not isinstance(units, UnitsResult):
        raise TypeError("units must be a UnitsResult.")
    _check_injection(tool_groups, shared)
    if ingest.dataset is None:
        raise ValueError("Stage 4 requires the Stage 0 dataset.")
    if depth.status is not StageStatus.COMPLETE:
        raise ValueError("Stage 4 requires a completed Stage 1 result.")
    if identification.status is not StageStatus.COMPLETE:
        raise ValueError("Stage 4 requires a completed Stage 2 result.")
    if units.status is not StageStatus.COMPLETE:
        raise ValueError("Stage 4 requires a completed Stage 3 result.")
    dataset = ingest.dataset
    if len(identification.curves) != len(dataset.curves) or len(units.curves) != len(dataset.curves):
        raise ValueError("Stages 2 and 3 did not record every curve.")
    shipped = load_shipped_tables()
    rules = validity_rules()
    version = _check_library(rules, shipped)
    non_finite = next((rule for rule in rules if rule.id == _NON_FINITE_ID), None)
    if non_finite is None:
        raise ValueError("The non-finite rule is not in the library.")
    prepared_list: list[_Prepared] = []
    for curve, identified, interpreted in zip(dataset.curves, identification.curves, units.curves, strict=True):
        if identified.curve != curve.ref or interpreted.curve != curve.ref:
            raise ValueError("Stage 2 or Stage 3 curve order does not match the dataset.")
        prepared_list.append(_prepare_curve(dataset, curve, identified, interpreted, rules, shipped, tool_groups))
    prepared = tuple(prepared_list)
    grid = dataset.grid.values
    ledger = _Ledger(ingest)
    recorded: list[CurveValidity] = []
    for item in prepared:
        classified = _gaps_for(item, prepared, shared, grid)
        gap_records = tuple(gap for gap, _match in classified)
        summaries = _summaries(item.samples, gap_records, grid)
        outcome, rule_version, reason, evidence_text = _outcome(item, version)
        _add_findings(ledger, item, outcome, reason, evidence_text, classified, grid, non_finite)
        recorded.append(
            CurveValidity(
                curve=item.curve.ref,
                mnemonic=item.identification.mnemonic,
                family_id=item.identification.family_id,
                tool_group=item.tool_group,
                outcome=outcome,
                rule_set_version=rule_version,
                ineligibility_reason=reason,
                eligibility_evidence=evidence_text,
                family_rules_applied=item.family_rules_applied,
                samples=item.samples,
                measured_interval=item.interval,
                gaps=gap_records,
                summaries=summaries,
            )
        )
    return ValidityResult(
        status=StageStatus.COMPLETE,
        findings=tuple(ledger.items),
        curves=tuple(recorded),
        library_version=version,
    )


CONTRACT = StageContract(
    stage_number=4,
    name="Validity",
    requires=StageRequires(
        units="Stage 3 has recorded a unit interpretation for every curve. This stage does not assume a unit.",
        completeness="Stages 0, 1, 2 and 3 have completed. Sample status is not yet recorded.",
        preconditions=(
            "Stage 0 returned the dataset.",
            "Stage 1 completed.",
            "Stage 2 identified every curve.",
            "Stage 3 interpreted every curve.",
        ),
    ),
    guarantees=StageGuarantees(
        row_count="Unchanged. Stage 4 does not add, drop, or reorder a sample.",
        units="Unchanged. Values are not converted in the stored arrays and are not clipped.",
        validity=(
            "Each curve has a validation outcome, a sample-status mask, a measured interval, "
            "a gap inventory, and interval summaries. INVALID samples stay visible with "
            "output inclusion PENDING_DECISION. Nothing is excluded from a file."
        ),
    ),
    declines_when=(
        "A curve that is not eligible is not given a family range, and is not marked PASSED.",
        "A unit with no exact conversion to the rule's unit is not compared.",
        "A GAPI rule is not applied to a CPS curve.",
        "OD-13 is unset, so dropout and an uninjected shared gap cannot be classified as such.",
        "NOT_APPLICABLE is not used.",
    ),
    depends_on_stages=(0, 1, 2, 3),
    alters_data=False,
)


def registered_stage(
    ingest: IngestResult,
    depth: DepthAxisResult,
    identification: IdentificationResult,
    units: UnitsResult,
) -> RegisteredStage:
    """A runner stage that assesses validity and does not replace the dataset. D24, D26, C5.

    The shipped call does not inject a tool group or a definition of shared.
    """

    def unmet(context: RunContext) -> tuple[str, ...]:
        if ingest.dataset is None:
            return ("Stage 0 did not return a dataset.",)
        if context.dataset is not ingest.dataset:
            return ("Stage 4 requires the Stage 0 dataset.",)
        if depth.status is not StageStatus.COMPLETE:
            return ("Stage 1 did not complete.",)
        if identification.status is not StageStatus.COMPLETE:
            return ("Stage 2 did not complete.",)
        if units.status is not StageStatus.COMPLETE:
            return ("Stage 3 did not complete.",)
        if len(identification.curves) != len(ingest.dataset.curves) or len(units.curves) != len(ingest.dataset.curves):
            return ("Stages 2 and 3 did not record every curve.",)
        return ()

    def run(_context: RunContext) -> StageOutcome:
        result = assess(ingest, depth, identification, units)
        return StageOutcome(result.status, result.findings)

    return RegisteredStage(CONTRACT, run, unmet)
