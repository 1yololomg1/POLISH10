"""Stage 3: interpret units and convert on read. It does not rewrite samples.

Implements C1, C4, C6, contracts section 3 and section 4 items 2 and 4,
D87, D90, and OD-05. Unit conversion is a recorded re-expression. It is
never a CHANGED sample.

Structure
---------
CONTRACT
    Stage 3 contract. alters_data is False. depends_on_stages is (0, 1, 2).
    Stage 4 depends on this stage's unit interpretation and target system.
TargetUnitSystem
    Imperial, metric, or as-recorded. None of the three is a default.
    Until the resolution point is decided, the recorded target is AS_RECORDED.
declared_unit_state
    ABSENT, EMPTY, or None when the text still has to be looked up.
units_eligibility
    Stage 3 eligibility from the Stage 2 value and the unit outcome. D34, D90.
interpret
    One unit outcome per curve, plus findings. Nothing is converted here.
read_values
    Copy a curve's samples and re-express them in a requested unit. OD-05.
require_unit
    Raise UnitMismatch unless the view is already in the reference unit. C1.
declare_depth_conversion
    A depth-derived parameter is invalid after the depth unit changes. C4.
    P14 marks it STALE. This stage only declares that.
registered_stage
    The same interpretation behind the stage runner.

Conversions are loaded from polish/tables/unit_spellings.json. The factors
were harvested from core/unit_standardization.py
UnitStandardizer.unit_conversions. Three of those rows were not copied:
conductivity to resistivity (a reciprocal), API gravity to kg/m3, and SG
to kg/m3. PU has no conversion; the legacy factor 1.0 onto V/V was not copied.
USEC/FT, US/FT and LB/FT3 use exact factors from 1 ft = 0.3048 m and
1 lb = 0.45359237 kg, not the rounded legacy numbers.

UXC9: this stage does not set a validation outcome. A compatible unit does
not mean a validation rule exists for that unit. That is Stage 4.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import numpy

from polish.model.dataset import Curve, Dataset
from polish.model.findings import Evidence, FindingRecord
from polish.model.ids import CurveRef
from polish.model.states import (
    BlockType,
    ProcessingEligibility,
    StageStatus,
    UnitInterpretation,
    UserAction,
    permitted_actions,
)
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, RunContext, StageOutcome
from polish.stages.s00_ingest import IngestResult
from polish.stages.s01_depth import DerivedQuantity
from polish.stages.s02_identify import CurveIdentification, IdentificationResult
from polish.tables.loader import (
    ConversionRecord,
    FamilyRecord,
    SpellingRecord,
    UnitSpellings,
    load_shipped_tables,
    load_unit_spellings,
)

# Stage 4 (validity) reads these. Declared here; Stage 4 is not built in P11.
STAGE_4_DEPENDS_ON = ("unit_interpretation", "target_system")

_UNRESOLVED = frozenset(
    {
        UnitInterpretation.ABSENT,
        UnitInterpretation.EMPTY,
        UnitInterpretation.UNRECOGNISED,
    }
)
_CROSS_DIMENSION = frozenset({"gamma_api", "count_rate"})


class TargetUnitSystem(Enum):
    """The section 4 item 4 resolution point. C6.

    Imperial and metric are not chosen. AS_RECORDED is the recorded target
    until that choice exists, and it converts nothing by itself.
    """

    IMPERIAL = "IMPERIAL"
    METRIC = "METRIC"
    AS_RECORDED = "AS_RECORDED"


class UnitMismatch(Exception):
    """C1. The units do not match. Values are not coerced.

    ``reason`` is the record of the refusal.
    """

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or reason.strip() == "":
            raise ValueError("UnitMismatch needs a recorded reason.")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class CurveUnits:
    """Unit interpretation of one curve. C1, D87, D90.

    ``canonical_unit`` and ``dimension`` are set when a single meaning was
    resolved. ``candidates`` is the list on an ambiguous spelling. A curve
    with no family is not checked for compatibility.
    """

    curve: CurveRef
    mnemonic: str
    outcome: UnitInterpretation
    processing_eligibility: ProcessingEligibility
    declared_unit: str | None
    canonical_unit: str | None
    dimension: str | None
    family_id: str | None
    candidates: tuple[SpellingRecord, ...]


@dataclass(frozen=True)
class UnitsResult:
    """What Stage 3 records. Source arrays are not in this result. C5, OD-05.

    ``target_system`` is AS_RECORDED until imperial or metric is decided.
    No sample is converted because of that target.
    """

    status: StageStatus
    findings: tuple[FindingRecord, ...]
    curves: tuple[CurveUnits, ...]
    target_system: TargetUnitSystem
    spellings_version: str


@dataclass(frozen=True)
class ValuesView:
    """Samples re-expressed in one unit. The dataset array is not this array. OD-05.

    ``conversion`` is None when the requested unit is already the curve's
    canonical unit. The array is read-only.
    """

    curve: CurveRef
    unit: str
    dimension: str
    values: numpy.ndarray
    conversion: ConversionRecord | None

    def __post_init__(self) -> None:
        if not isinstance(self.curve, CurveRef):
            raise TypeError("curve must be a CurveRef.")
        if not isinstance(self.unit, str) or self.unit.strip() == "":
            raise ValueError("A values view needs a unit.")
        if not isinstance(self.dimension, str) or self.dimension.strip() == "":
            raise ValueError("A values view needs a dimension.")
        if not isinstance(self.values, numpy.ndarray) or self.values.dtype != numpy.float64:
            raise TypeError("values must be a float64 array.")
        if self.values.flags.writeable:
            raise ValueError("A values view is read-only.")
        if self.conversion is not None and not isinstance(self.conversion, ConversionRecord):
            raise TypeError("conversion must be a ConversionRecord or None.")


@dataclass(frozen=True)
class DepthDerivedDeclaration:
    """C4. Converting the depth unit invalidates a parameter derived under the old unit.

    P14 marks the parameter STALE. This record is only the declaration.
    """

    name: str
    derived_under: str
    depth_unit: str
    invalidated: bool
    reason: str


def declared_unit_state(unit: str | None) -> UnitInterpretation | None:
    """ABSENT, EMPTY, or None when ``unit`` is text to look up. C1, C3.

    None means the file declared no unit. A blank string means the unit
    field is present and empty. Stage 0 stores a blank LAS unit token as "".
    """

    if unit is None:
        return UnitInterpretation.ABSENT
    if not isinstance(unit, str):
        raise TypeError("A declared unit must be a str or None.")
    if unit.strip() == "":
        return UnitInterpretation.EMPTY
    return None


def units_eligibility(
    prior: ProcessingEligibility,
    outcome: UnitInterpretation,
) -> ProcessingEligibility:
    """Eligibility after the unit outcome. D34, D90, owner 12 Oct 2026.

    A Stage 2 value other than ELIGIBLE is kept. An unresolved unit on an
    otherwise eligible curve is NOT_ELIGIBLE_UNIT_UNRESOLVED. A recognised
    unit stays eligible, including a unit outside the family's accepted
    dimensions: that curve carries a RULE_DEPENDENT_WARNING instead.
    """

    if not isinstance(prior, ProcessingEligibility):
        raise TypeError("prior must be a ProcessingEligibility.")
    if not isinstance(outcome, UnitInterpretation):
        raise TypeError("outcome must be a UnitInterpretation.")
    if prior is not ProcessingEligibility.ELIGIBLE:
        return prior
    if outcome in _UNRESOLVED:
        return ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED
    if outcome in (UnitInterpretation.RECOGNISED, UnitInterpretation.INCOMPATIBLE_WITH_FAMILY):
        return ProcessingEligibility.ELIGIBLE
    raise ValueError(f"No eligibility rule for {outcome.value}.")


def _resolve(
    declared: str | None,
    family: FamilyRecord | None,
    spellings: UnitSpellings,
) -> tuple[UnitInterpretation, SpellingRecord | None, tuple[SpellingRecord, ...]]:
    state = declared_unit_state(declared)
    if state is not None:
        return state, None, ()
    if declared is None:
        return UnitInterpretation.ABSENT, None, ()
    matches = spellings.meanings(declared)
    if len(matches) == 0:
        return UnitInterpretation.UNRECOGNISED, None, ()
    if len(matches) == 1:
        chosen = matches[0]
        if family is None or chosen.dimension in family.accepted_dimensions:
            return UnitInterpretation.RECOGNISED, chosen, ()
        return UnitInterpretation.INCOMPATIBLE_WITH_FAMILY, chosen, ()
    if family is None:
        return UnitInterpretation.UNRECOGNISED, None, matches
    accepted = tuple(match for match in matches if match.dimension in family.accepted_dimensions)
    if len(accepted) == 1:
        return UnitInterpretation.RECOGNISED, accepted[0], ()
    return UnitInterpretation.UNRECOGNISED, None, matches


class _Ledger:
    """Findings for one interpretation. Identifiers are stable inside that call."""

    def __init__(self, ingest: IngestResult) -> None:
        self._read_at = ingest.source_file.read_at
        self.items: list[FindingRecord] = []

    def add(
        self,
        block: BlockType,
        message: str,
        ref: CurveRef,
        fields: dict[str, str | int | float | bool],
    ) -> None:
        number = len(self.items) + 1
        self.items.append(
            FindingRecord(
                id=f"s03-{number:04d}",
                stage=3,
                block_type=block,
                subject_refs=(ref,),
                evidence=Evidence(block.value, tuple(sorted(fields.items()))),
                message=message,
                created_at=self._read_at,
            )
        )


def _candidates_json(matches: tuple[SpellingRecord, ...]) -> str:
    payload = [
        {"canonical_unit": match.canonical_unit, "dimension": match.dimension, "spelling": match.spelling}
        for match in matches
    ]
    return json.dumps(payload, sort_keys=True)


def _record_finding(
    ledger: _Ledger,
    item: CurveIdentification,
    outcome: UnitInterpretation,
    chosen: SpellingRecord | None,
    candidates: tuple[SpellingRecord, ...],
    family: FamilyRecord | None,
    declared: str | None,
) -> None:
    if outcome is UnitInterpretation.RECOGNISED:
        return
    mnemonic = item.mnemonic
    family_id = item.family_id if item.family_id is not None else ""
    common: dict[str, str | int | float | bool] = {
        "outcome": outcome.value,
        "family_id": family_id,
        "mnemonic": mnemonic,
    }
    if declared is not None:
        common["declared_unit"] = declared
    if outcome in (
        UnitInterpretation.ABSENT,
        UnitInterpretation.EMPTY,
        UnitInterpretation.UNRECOGNISED,
    ):
        if candidates:
            common["candidates"] = _candidates_json(candidates)
            listed = ", ".join(f"{match.dimension} ({match.canonical_unit})" for match in candidates)
            message = (
                f"Curve {mnemonic} unit {declared!r} has more than one meaning ({listed}). "
                "The family does not accept exactly one of them, so the unit stays unrecognised."
            )
        elif outcome is UnitInterpretation.ABSENT:
            message = f"Curve {mnemonic} declares no unit. The unit is not assumed."
        elif outcome is UnitInterpretation.EMPTY:
            message = f"Curve {mnemonic} has an empty unit field. The unit is not assumed."
        else:
            shown = declared if declared is not None else ""
            message = f"Curve {mnemonic} unit {shown!r} is not a known unit. The unit is not assumed."
        ledger.add(BlockType.AMBIGUOUS_INTERPRETATION, message, item.curve, common)
        return
    if chosen is None or family is None:
        raise ValueError("An incompatible unit needs a resolved spelling and a family.")
    accepted = json.dumps(list(family.accepted_dimensions))
    common["canonical_unit"] = chosen.canonical_unit
    common["dimension"] = chosen.dimension
    common["accepted_dimensions"] = accepted
    message = (
        f"Curve {mnemonic} unit {chosen.canonical_unit} is {chosen.dimension}, "
        f"which is not an accepted dimension of {family.id}."
    )
    ledger.add(BlockType.RULE_DEPENDENT_WARNING, message, item.curve, common)


def interpret(ingest: IngestResult, identification: IdentificationResult) -> UnitsResult:
    """Interpret every curve's declared unit. C1, C6, D87, D90, OD-05.

    Does not write the dataset. The target system is AS_RECORDED, so this
    call converts nothing. Imperial and metric are not selected.

    Parameters
    ----------
    ingest:
        The Stage 0 result whose dataset is read.
    identification:
        A completed Stage 2 result, one row per curve, in column order.

    Returns
    -------
    UnitsResult
        One CurveUnits per curve, the findings, and the spellings version.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If Stage 0 has no dataset or Stage 2 did not complete for every curve.
    """

    if not isinstance(ingest, IngestResult):
        raise TypeError("ingest must be an IngestResult.")
    if not isinstance(identification, IdentificationResult):
        raise TypeError("identification must be an IdentificationResult.")
    if ingest.dataset is None:
        raise ValueError("Stage 3 requires the Stage 0 dataset.")
    if identification.status is not StageStatus.COMPLETE:
        raise ValueError("Stage 3 requires a completed Stage 2 result.")
    dataset = ingest.dataset
    if len(identification.curves) != len(dataset.curves):
        raise ValueError("Stage 2 did not identify every curve.")
    shipped = load_shipped_tables()
    spellings = load_unit_spellings()
    ledger = _Ledger(ingest)
    interpreted: list[CurveUnits] = []
    for curve, item in zip(dataset.curves, identification.curves, strict=True):
        if item.curve != curve.ref:
            raise ValueError("Stage 2 curve order does not match the dataset.")
        family = shipped.family(item.family_id) if item.family_id is not None else None
        if item.family_id is not None and family is None:
            raise ValueError(f"Stage 2 family {item.family_id!r} is not in the shipped tables.")
        outcome, chosen, candidates = _resolve(curve.original_unit, family, spellings)
        eligibility = units_eligibility(item.processing_eligibility, outcome)
        _record_finding(ledger, item, outcome, chosen, candidates, family, curve.original_unit)
        interpreted.append(
            CurveUnits(
                curve=curve.ref,
                mnemonic=item.mnemonic,
                outcome=outcome,
                processing_eligibility=eligibility,
                declared_unit=curve.original_unit,
                canonical_unit=None if chosen is None else chosen.canonical_unit,
                dimension=None if chosen is None else chosen.dimension,
                family_id=item.family_id,
                candidates=candidates,
            )
        )
    return UnitsResult(
        status=StageStatus.COMPLETE,
        findings=tuple(ledger.items),
        curves=tuple(interpreted),
        target_system=TargetUnitSystem.AS_RECORDED,
        spellings_version=spellings.version,
    )


def _dataset_curve(dataset: Dataset, ref: CurveRef) -> Curve:
    for curve in dataset.curves:
        if curve.ref == ref:
            return curve
    raise ValueError("curve_ref is not a curve of this dataset.")


def _apply(values: numpy.ndarray, record: ConversionRecord) -> numpy.ndarray:
    converted = numpy.empty(values.shape, dtype=numpy.float64)
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            converted[index] = value
            continue
        exact = Fraction(float(value)) * record.factor + record.offset
        converted[index] = float(exact)
    converted.flags.writeable = False
    return converted


def _refusal(source: SpellingRecord, requested: SpellingRecord) -> str:
    pair = {source.dimension, requested.dimension}
    if pair == _CROSS_DIMENSION or source.dimension != requested.dimension:
        if pair == _CROSS_DIMENSION:
            return (
                "No fixed factor converts gamma_api and count_rate (OD-05 case 4). "
                f"{source.canonical_unit} is not converted to {requested.canonical_unit}."
            )
        return (
            f"No conversion from {source.canonical_unit} ({source.dimension}) "
            f"to {requested.canonical_unit} ({requested.dimension}). "
            "Conversion stays inside one dimension."
        )
    if source.canonical_unit == "PU" or requested.canonical_unit == "PU":
        return (
            f"No conversion is recorded from {source.canonical_unit} to {requested.canonical_unit}. "
            "PU is a recognised porosity spelling and is not converted to V/V."
        )
    return (
        f"No conversion is recorded from {source.canonical_unit} to {requested.canonical_unit} "
        f"in {source.dimension}."
    )


def _requested_meaning(spellings: UnitSpellings, unit: str, dimension: str) -> SpellingRecord:
    matches = spellings.meanings(unit)
    if len(matches) == 0:
        raise UnitMismatch(f"Unit {unit!r} is not a known unit. It is not assumed.")
    in_dimension = tuple(match for match in matches if match.dimension == dimension)
    if len(in_dimension) == 1:
        return in_dimension[0]
    if len(matches) == 1 and matches[0].dimension != dimension:
        raise UnitMismatch(_refusal(_stand_in(dimension), matches[0]))
    listed = ", ".join(match.dimension for match in matches)
    raise UnitMismatch(f"Unit {unit!r} has more than one meaning ({listed}) in this read.")


def _stand_in(dimension: str) -> SpellingRecord:
    return SpellingRecord(
        spelling="",
        canonical_unit=dimension,
        dimension=dimension,
        source="",
        source_tier="",
    )


def read_values(dataset: Dataset, curve_units: CurveUnits, unit: str) -> ValuesView:
    """Copy ``curve_units`` samples and re-express them in ``unit``. OD-05, C1.

    The dataset array is not written. A different dimension is refused, and
    gamma_api is never converted to count_rate. The refusal's reason is on
    the UnitMismatch.

    Parameters
    ----------
    dataset:
        The source dataset. Its arrays stay read-only.
    curve_units:
        A recognised interpretation. Other outcomes cannot be read in a unit.
    unit:
        The unit the caller wants the values in.

    Returns
    -------
    ValuesView
        A read-only copy, the unit it is in, and the conversion when one ran.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a Dataset.")
    if not isinstance(curve_units, CurveUnits):
        raise TypeError("curve_units must be a CurveUnits.")
    if not isinstance(unit, str):
        raise TypeError("unit must be a str.")
    if curve_units.outcome is not UnitInterpretation.RECOGNISED:
        raise UnitMismatch(
            f"Curve {curve_units.mnemonic} is {curve_units.outcome.value}. "
            "Its values are not read in another unit, and no unit is assumed."
        )
    if curve_units.canonical_unit is None or curve_units.dimension is None:
        raise UnitMismatch(f"Curve {curve_units.mnemonic} has no resolved unit.")
    spellings = load_unit_spellings()
    requested = _requested_meaning(spellings, unit, curve_units.dimension)
    source = _dataset_curve(dataset, curve_units.curve)
    copied = numpy.array(source.values, dtype=numpy.float64, copy=True)
    if requested.dimension != curve_units.dimension:
        copied.flags.writeable = False
        raise UnitMismatch(
            _refusal(
                SpellingRecord(
                    curve_units.declared_unit or curve_units.canonical_unit,
                    curve_units.canonical_unit,
                    curve_units.dimension,
                    "",
                    "",
                ),
                requested,
            )
        )
    if requested.canonical_unit == curve_units.canonical_unit:
        copied.flags.writeable = False
        return ValuesView(curve_units.curve, curve_units.canonical_unit, curve_units.dimension, copied, None)
    record = spellings.conversion(curve_units.canonical_unit, requested.canonical_unit, curve_units.dimension)
    if record is None:
        copied.flags.writeable = False
        raise UnitMismatch(
            _refusal(
                SpellingRecord(
                    curve_units.canonical_unit,
                    curve_units.canonical_unit,
                    curve_units.dimension,
                    "",
                    "",
                ),
                requested,
            )
        )
    converted = _apply(copied, record)
    return ValuesView(curve_units.curve, requested.canonical_unit, requested.dimension, converted, record)


def require_unit(view: ValuesView, reference_unit: str) -> ValuesView:
    """Return ``view`` when it is already in ``reference_unit``. C1.

    A mismatch raises UnitMismatch. The values are not coerced into the
    reference unit. A rule written in g/cc therefore cannot see numbers
    that were re-expressed in kg/m3.

    Parameters
    ----------
    view:
        Samples already expressed in one canonical unit.
    reference_unit:
        The unit a rule or formula is written in.
    """

    if not isinstance(view, ValuesView):
        raise TypeError("view must be a ValuesView.")
    if not isinstance(reference_unit, str):
        raise TypeError("reference_unit must be a str.")
    matches = load_unit_spellings().meanings(reference_unit)
    same = [
        match
        for match in matches
        if match.dimension == view.dimension and match.canonical_unit == view.unit
    ]
    if len(same) == 1:
        return view
    raise UnitMismatch(
        f"View is in {view.unit} ({view.dimension}); reference {reference_unit!r} does not match. "
        "Values are not coerced (C1)."
    )


def _canonical_length(unit: str, spellings: UnitSpellings) -> str:
    matches = tuple(match for match in spellings.meanings(unit) if match.dimension == "length")
    if len(matches) == 1:
        return matches[0].canonical_unit
    return unit


def declare_depth_conversion(quantity: DerivedQuantity, depth_unit: str) -> DepthDerivedDeclaration:
    """Declare whether a depth-derived parameter survives a depth-unit change. C4.

    The parameter stays usable only while the depth unit is still the unit
    it was derived under. Feet and FEET are the same length unit. Feet and
    metres are not. P14 marks a invalidated parameter STALE; this function
    does not.

    Parameters
    ----------
    quantity:
        A number stored with the unit it was derived under.
    depth_unit:
        The depth unit after a conversion, or the current depth unit when
        nothing was converted.
    """

    if not isinstance(quantity, DerivedQuantity):
        raise TypeError("quantity must be a DerivedQuantity.")
    if not isinstance(depth_unit, str):
        raise TypeError("depth_unit must be a str.")
    spellings = load_unit_spellings()
    before = _canonical_length(quantity.unit, spellings)
    after = _canonical_length(depth_unit, spellings)
    invalidated = before != after
    if invalidated:
        reason = (
            f"{quantity.name} was derived under {quantity.unit} and is invalid after the depth unit "
            f"is converted to {depth_unit} (C4). P14 marks it STALE. "
            "A step derived in feet is not applied to depths in metres."
        )
    else:
        reason = f"{quantity.name} was derived under {quantity.unit}, which is still the depth unit."
    return DepthDerivedDeclaration(
        name=quantity.name,
        derived_under=quantity.unit,
        depth_unit=depth_unit,
        invalidated=invalidated,
        reason=reason,
    )


CONTRACT = StageContract(
    stage_number=3,
    name="Units",
    requires=StageRequires(
        units="Each curve still carries the unit text Stage 0 stored. No unit has been assumed.",
        completeness="Stages 0, 1 and 2 have completed for this dataset.",
        preconditions=(
            "Stage 0 returned the dataset.",
            "Stage 1 completed.",
            "Stage 2 identified every curve.",
        ),
    ),
    guarantees=StageGuarantees(
        row_count="Unchanged. Stage 3 does not add, drop, or reorder a sample.",
        units=(
            "Each curve has a unit interpretation. The target system is AS_RECORDED "
            "until imperial or metric is decided, and no sample is converted. "
            "Stage 4 results depend on this unit interpretation and target system."
        ),
        validity=(
            "Processing eligibility is updated from the unit outcome. "
            "A validation outcome is not set. Sample status is not set."
        ),
    ),
    declines_when=(
        "An absent, empty, or unrecognised unit is not assumed.",
        "A unit outside the family's accepted dimensions is not converted into one.",
        "gamma_api and count_rate are not converted between.",
        "Imperial and metric are not selected.",
    ),
    depends_on_stages=(0, 1, 2),
    alters_data=False,
)


def registered_stage(ingest: IngestResult, identification: IdentificationResult) -> RegisteredStage:
    """A runner stage that interprets units and does not replace the dataset. D24, C5, C6.

    Stage 4 depends on the interpretation and the target system this stage
    records. Those names are STAGE_4_DEPENDS_ON.
    """

    def unmet(context: RunContext) -> tuple[str, ...]:
        if ingest.dataset is None:
            return ("Stage 0 did not return a dataset.",)
        if context.dataset is not ingest.dataset:
            return ("Stage 3 requires the Stage 0 dataset.",)
        if identification.status is not StageStatus.COMPLETE:
            return ("Stage 2 did not complete.",)
        if ingest.dataset is not None and len(identification.curves) != len(ingest.dataset.curves):
            return ("Stage 2 did not identify every curve.",)
        return ()

    def run(_context: RunContext) -> StageOutcome:
        result = interpret(ingest, identification)
        return StageOutcome(result.status, result.findings)

    return RegisteredStage(CONTRACT, run, unmet)


def incompatible_unit_actions() -> frozenset[UserAction]:
    """Actions on an incompatible-unit warning. D90.

    MAP_EXPLICITLY is permitted. Eligibility does not bypass require_unit.
    """

    return permitted_actions(BlockType.RULE_DEPENDENT_WARNING)
