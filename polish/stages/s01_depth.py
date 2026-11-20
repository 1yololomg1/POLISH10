"""Stage 1: classify the depth axis. It does not rewrite samples.

Implements D3 (ascending only; a bottom-up log is refused with the reason),
D8, D9 (record the alias, do not copy the curve), D12 (implied interval from
the first and last depths in the data; header STRT and STOP are judged by the
endpoint check; grid measured from the first depth), D28, F10, D67 (record
STEP 0, do not invent a step for an uneven grid), D86, D99 (a recorded
depth-unit choice), D88 as revised to ascending only, D89, C4, and contracts
section 4 item 5.

Structure
---------
CONTRACT
    Stage 1 contract. alters_data is False. depends_on_stages is (0,).
depth_index_mnemonics
    The small index-alias set. It stays in this module (D28, F10).
DerivedQuantity
    A number stored with the unit it was derived under. C4.
classify
    Read an ingest result and return a DepthAxisResult. A depth-unit
    choice is applied only when the caller passes it. D86.
registered_stage
    The same classification behind the stage runner, with no choice.

Direction is stamped ASCENDING on an accepted result. DESCENDING is never
stamped. Depths are not reversed, re-sorted, or replaced. A corrected step
is a recorded quantity for a later export. This stage does not write it
onto the samples. Stages 0–4 cannot alter data (D24).

Irregular-but-ascending grids are accepted (D67, the revised D88). A declared
STEP that the samples do not meet is INFORMATIONAL, not a structural failure.
The allowance is one unit of each depth's own last written decimal, and never
more than half the step (D12). The grid is measured from the first depth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from polish.model.dataset import DepthDirection, DepthRegularity
from polish.model.depth_unit_choice import DepthUnitChoice
from polish.model.depth_units import depth_unit_spellings
from polish.model.findings import Evidence, FindingRecord, finding_key
from polish.model.ids import CurveRef, DatasetId
from polish.model.states import BlockType, StageStatus
from polish.rules.loader import load_rule
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, RunContext, StageOutcome
from polish.stages.s00_ingest import IngestResult

_CANONICAL_DEPTH_NAME = "DEPT"
_GRID_RULE_ID = "depth.grid_rounding"
_STEP_RULE_ID = "depth.standard_steps"


def depth_index_mnemonics() -> frozenset[str]:
    """Mnemonics Stage 1 accepts as the first-curve depth index.

    D28, F10. Stage 2 owns the mnemonic table. This stage needs the subset
    before that table exists, so the set stays in this module. Do not move it.

    Harvested from ``DepthValidationManager.required_depth_keywords`` at
    commit dc0bd5d: DEPT, DEPTH, MD, TVD, TVDSS. That method searched every
    column by substring and could insert a column. This stage matches the
    whole first-curve mnemonic, without regard to case, and does not search
    the other curves. DEPTH_MD is not in the set. A later depth-like curve
    is kept as it is and is not an ambiguity.
    """

    return frozenset({"DEPT", "DEPTH", "MD", "TVD", "TVDSS"})


@dataclass(frozen=True)
class DerivedQuantity:
    """A number and the unit it was derived under. C4.

    ``text`` is the decimal spelling. ``value`` is that spelling as a float
    and is only meaningful while ``unit`` is still the derivation unit.
    """

    name: str
    text: str
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name.strip() == "":
            raise ValueError("DerivedQuantity.name must be a non-empty string.")
        if not isinstance(self.text, str) or self.text.strip() == "":
            raise ValueError("DerivedQuantity.text must be a non-empty string.")
        if not isinstance(self.unit, str):
            raise TypeError("DerivedQuantity.unit must be a str.")
        try:
            number = Decimal(self.text)
        except InvalidOperation as exc:
            raise ValueError(f"DerivedQuantity text {self.text!r} is not a decimal.") from exc
        if not number.is_finite():
            raise ValueError("DerivedQuantity text must be finite.")

    def value(self) -> float:
        """The spelling as a float. Call ``assert_derived_unit`` first. C4."""

        return float(Decimal(self.text))


def assert_derived_unit(quantity: DerivedQuantity, unit: str) -> Decimal:
    """Return the decimal value when ``quantity`` was derived under ``unit``.

    C4. A depth spacing derived in one unit is not usable in another. This
    raises rather than converting.
    """

    if not isinstance(quantity, DerivedQuantity):
        raise TypeError("quantity must be a DerivedQuantity.")
    if quantity.unit != unit:
        raise AssertionError(
            f"{quantity.name} was derived under {quantity.unit!r} and cannot be used under {unit!r} (C4)."
        )
    return Decimal(quantity.text)


@dataclass(frozen=True)
class DepthAlias:
    """The D9 decision. Export materialises the copy. This stage does not."""

    source: CurveRef
    canonical_name: str


@dataclass(frozen=True)
class DepthAxisResult:
    """What Stage 1 records. Source arrays are not in this result."""

    status: StageStatus
    findings: tuple[FindingRecord, ...]
    direction: DepthDirection | None
    regularity: DepthRegularity | None
    declared_step: DerivedQuantity | None
    observed_step: DerivedQuantity | None
    corrected_step: DerivedQuantity | None
    recorded_strt: DerivedQuantity | None
    recorded_stop: DerivedQuantity | None
    displacement: DerivedQuantity | None
    alias: DepthAlias | None
    primary_standard_step: DerivedQuantity | None
    depth_unit: str | None
    depth_unit_decision_id: str | None


CONTRACT = StageContract(
    stage_number=1,
    name="Depth axis",
    requires=StageRequires(
        units="The depth unit is the first curve's unit as written. An absent unit is not filled in (C1, C3).",
        completeness="Stage 0 has read the file. Sample status is still unknown.",
        preconditions=("Stage 0 returned the dataset this stage classifies.",),
    ),
    guarantees=StageGuarantees(
        row_count="Unchanged. Stage 1 does not add, drop, or reorder a sample.",
        units="Unchanged. A corrected step is recorded with the depth unit and is not written onto the samples.",
        validity="Regularity, direction, and the D9 alias are recorded. Sample status is not set.",
    ),
    declines_when=(
        "The first curve is not a plausible depth index.",
        "Depths decrease somewhere, repeat, or the header STEP is negative while the depths increase.",
        "The log is recorded bottom-up. Valid LAS; POLISH supports increasing depth only.",
        "STRT or STOP disagrees with the first or last depth by more than one sample interval.",
    ),
    depends_on_stages=(0,),
    alters_data=False,
)


def registered_stage(ingest: IngestResult) -> RegisteredStage:
    """A runner stage that classifies ``ingest`` and does not replace it. D24, C5."""

    def unmet(context: RunContext) -> tuple[str, ...]:
        if ingest.dataset is None:
            return ("Stage 0 did not return a dataset.",)
        if context.dataset is not ingest.dataset:
            return ("Stage 1 requires the Stage 0 dataset.",)
        return ()

    def run(_context: RunContext) -> StageOutcome:
        result = classify(ingest)
        return StageOutcome(result.status, result.findings)

    return RegisteredStage(CONTRACT, run, unmet)


_STAGE1_CHECKS: tuple[tuple[str, BlockType], ...] = (
    ("depth_no_curve", BlockType.STRUCTURAL_DATA_FAILURE),
    ("depth_index_not_recognised", BlockType.STRUCTURAL_DATA_FAILURE),
    ("depth_too_few_samples", BlockType.STRUCTURAL_DATA_FAILURE),
    ("depth_token_not_decimal", BlockType.STRUCTURAL_DATA_FAILURE),
    ("depth_duplicate", BlockType.STRUCTURAL_DATA_FAILURE),
    ("depth_decreases", BlockType.STRUCTURAL_DATA_FAILURE),
    ("depth_bottom_up", BlockType.STRUCTURAL_DATA_FAILURE),
    ("step_negative", BlockType.STRUCTURAL_DATA_FAILURE),
    ("depth_unit_unrecognised", BlockType.AMBIGUOUS_INTERPRETATION),
    ("step_zero", BlockType.INFORMATIONAL),
    ("step_blank_regular", BlockType.INFORMATIONAL),
    ("step_blank_irregular", BlockType.INFORMATIONAL),
    ("strt_blank", BlockType.INFORMATIONAL),
    ("strt_not_a_number", BlockType.INFORMATIONAL),
    ("depth_exponential", BlockType.INFORMATIONAL),
    ("grid_off_declared_step", BlockType.INFORMATIONAL),
    ("step_corrected", BlockType.INFORMATIONAL),
    ("step_not_standard", BlockType.INFORMATIONAL),
    ("endpoint_unit_mismatch", BlockType.AMBIGUOUS_INTERPRETATION),
    ("endpoint_not_compared", BlockType.INFORMATIONAL),
    ("endpoint_within_step", BlockType.INFORMATIONAL),
    ("endpoint_beyond_step", BlockType.STRUCTURAL_DATA_FAILURE),
)
CHECKS = frozenset(name for name, _block in _STAGE1_CHECKS)
_CHECK_BLOCK = dict(_STAGE1_CHECKS)


class _Ledger:
    """Findings for one classification. An id is the check and what it is about. D96."""

    def __init__(self, ingest: IngestResult) -> None:
        self._ingest = ingest
        self.items: list[FindingRecord] = []

    def add(
        self,
        check: str,
        identity: tuple[tuple[str, str], ...],
        message: str,
        fields: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Record one finding. The id is not this finding's position in the run. D96."""

        if check not in CHECKS:
            raise ValueError(f"Check {check!r} is not declared for this stage.")
        block = _CHECK_BLOCK[check]
        finding_id = finding_key(1, check, identity)
        if any(item.id == finding_id for item in self.items):
            raise ValueError(
                f"Finding {finding_id} for check {check!r} and identity {identity!r} is already in this run."
            )
        dataset_id = DatasetId.for_source(self._ingest.source_file.file_hash)
        refs = (self._ingest.source_file.file_hash, CurveRef(dataset_id, 0))
        evidence = Evidence(block.value, tuple(sorted((fields or {}).items())))
        self.items.append(
            FindingRecord(
                id=finding_id,
                stage=1,
                check=check,
                identity=identity,
                block_type=block,
                subject_refs=refs,
                evidence=evidence,
                message=message,
                created_at=self._ingest.source_file.read_at,
            )
        )


def classify(
    ingest: IngestResult,
    depth_unit_choices: tuple[DepthUnitChoice, ...] = (),
) -> DepthAxisResult:
    """Classify the depth axis of one Stage 0 result.

    Implements D3, D8, D9, D12, D28, F10, D67, D86, D88 (ascending only), D89, D96, D99, C4.
    ``depth_unit_choices`` are applied only because the caller passed them (D99).
    One choice, of a spelling the depth-unit question lists, supplies the unit
    for the standard steps and the endpoint checks. The question is not raised
    again. A held session passes nothing, and a spelling the question does not
    list is refused. The choice is used only when the file's own depth unit is
    absent or not recognised. Does not reverse, re-sort, or copy a curve.
    Does not open the network.

    Parameters
    ----------
    ingest:
        The Stage 0 result whose depth column is classified.
    depth_unit_choices:
        THIS_ITEM depth-unit choices. Empty applies none.

    Returns
    -------
    DepthAxisResult
        Direction, regularity, and the recorded quantities. Source arrays
        are not included.

    Raises
    ------
    TypeError
        ``ingest`` is not an IngestResult, or a choice is not a DepthUnitChoice.
    ValueError
        More than one choice was passed, or the chosen unit is not one the
        depth-unit question lists.
    """

    if not isinstance(ingest, IngestResult):
        raise TypeError("ingest must be an IngestResult.")
    load_rule(_GRID_RULE_ID)
    standards = _standard_steps()
    spellings = depth_unit_spellings()
    choice = _one_depth_choice(depth_unit_choices, spellings.options())
    ledger = _Ledger(ingest)
    curves = ingest.audit.curves
    tokens = ingest.audit.depth_tokens
    if not curves:
        ledger.add("depth_no_curve", (), "The file has no curve to use as a depth index.")
        return _refused(ledger)

    mnemonic = curves[0].mnemonic
    file_unit = curves[0].unit
    dataset_id = DatasetId.for_source(ingest.source_file.file_hash)
    alias = DepthAlias(CurveRef(dataset_id, 0), _CANONICAL_DEPTH_NAME)
    if mnemonic.strip().upper() not in depth_index_mnemonics():
        ledger.add(
            "depth_index_not_recognised",
            (),
            f"The first curve {mnemonic!r} is not a depth index. Stage 1 halted.",
            {"observed": mnemonic},
        )
        return _refused(ledger)

    if len(tokens) < 2:
        ledger.add(
            "depth_too_few_samples",
            (),
            "The depth column has fewer than 2 samples, so the axis cannot be classified.",
        )
        return _refused(ledger, alias=alias, depth_unit=file_unit)

    parsed = _decimals(tokens)
    if parsed is None:
        ledger.add(
            "depth_token_not_decimal",
            (),
            "A depth token is not a finite decimal, so the axis was not classified.",
        )
        return _refused(ledger, alias=alias, depth_unit=file_unit)
    depths = parsed

    duplicate = _first_duplicate(depths)
    if duplicate is not None:
        index, text = duplicate
        ledger.add(
            "depth_duplicate",
            (),
            f"Duplicate depth {text} at indexes {index} and {index + 1}.",
            {"observed": text},
        )
        return _refused(ledger, alias=alias, depth_unit=file_unit)

    if _bottom_up(depths):
        unit_suffix = f" {file_unit.strip()}" if file_unit.strip() != "" else ""
        ledger.add(
            "depth_bottom_up",
            (),
            (
                f"This log is recorded bottom-up: depth decreases from {_dec_text(depths[0])} "
                f"to {_dec_text(depths[-1])}{unit_suffix}. That is valid LAS, but POLISH supports "
                "increasing depth only, so the file was not processed. The file is unchanged."
            ),
            {"declared": _dec_text(depths[0]), "observed": _dec_text(depths[-1])},
        )
        return _refused(ledger, alias=alias, depth_unit=file_unit)

    decrease = _first_decrease(depths)
    if decrease is not None:
        index, earlier, later = decrease
        ledger.add(
            "depth_decreases",
            (),
            f"Depth decreases from {earlier} to {later} at indexes {index} and {index + 1}.",
            {"declared": earlier, "observed": later, "difference": _dec_text(Decimal(later) - Decimal(earlier))},
        )

    step = _decimal_or_none(ingest.audit.step_text)
    if step is not None and step < 0:
        ledger.add(
            "step_negative",
            (),
            (
                f"STEP {ingest.audit.step_text} is negative, but the depths increase. "
                "The header contradicts the data, so the file was not processed."
            ),
            {"declared": ingest.audit.step_text or ""},
        )
    if ledger.items and any(item.block_type is BlockType.STRUCTURAL_DATA_FAILURE for item in ledger.items):
        return _refused(ledger, alias=alias, depth_unit=file_unit)

    decision_id: str | None = None
    unit = file_unit
    if choice is not None and spellings.family_name(file_unit) is None:
        unit = choice.unit
        decision_id = choice.decision_id
    family = spellings.family_name(unit)
    if family is None:
        shown = file_unit if file_unit.strip() != "" else ""
        ledger.add(
            "depth_unit_unrecognised",
            (),
            f"Depth unit {shown!r} is absent or not recognised. Feet or metres were not assumed.",
            {"observed": shown, "options": json.dumps(list(spellings.options()))},
        )

    declared = _declared_step(ingest, step, unit, standards, family)
    observed: DerivedQuantity | None = None
    corrected: DerivedQuantity | None = None
    displacement: DerivedQuantity | None = None
    regularity: DepthRegularity | None = None
    primary = _primary(family, unit, standards) if family is not None else None
    _note_strt(ledger, ingest.audit.strt_text)

    if _any_exponent(tokens):
        ledger.add(
            "depth_exponential",
            (),
            (
                "The depths are written in exponential form, which LAS 2.0 does not support, "
                "so the spacing was not assessed."
            ),
        )
    elif step is None or step == 0:
        regularity, observed = _median_grid(ledger, depths, tokens, unit, step == 0)
        if step == 0:
            declared = DerivedQuantity("declared_step", "0", _step_unit(ingest, unit))
    else:
        regularity, observed = _declared_grid(ledger, depths, tokens, step, unit, ingest)

    if family is not None and not _any_exponent(tokens):
        compared_name, compared_value = _compared_spacing(depths)
        corrected, displacement = _standard_step(
            ledger,
            ingest,
            depths,
            tokens,
            compared_name,
            compared_value,
            unit,
            family,
            _finest_places(tokens),
            standards,
        )

    recorded_strt, recorded_stop = _endpoints(ledger, ingest, depths, step, observed, unit, standards)
    if primary is not None:
        assert_derived_unit(primary, unit)
    return _finish_structural(
        DepthAxisResult(
            StageStatus.COMPLETE,
            tuple(ledger.items),
            DepthDirection.ASCENDING,
            regularity,
            declared,
            observed,
            corrected,
            recorded_strt,
            recorded_stop,
            displacement,
            alias,
            primary,
            unit,
            decision_id,
        )
    )


def _refused(
    ledger: _Ledger,
    alias: DepthAlias | None = None,
    depth_unit: str | None = None,
) -> DepthAxisResult:
    return DepthAxisResult(
        StageStatus.REFUSED_STRUCTURAL,
        tuple(ledger.items),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        alias,
        None,
        depth_unit,
        None,
    )


def _one_depth_choice(choices: tuple[DepthUnitChoice, ...], options: tuple[str, ...]) -> DepthUnitChoice | None:
    """The one passed choice, or None. A unit the question does not list is refused. D86."""

    if type(choices) is not tuple or any(not isinstance(item, DepthUnitChoice) for item in choices):
        raise TypeError("depth_unit_choices must be a tuple of DepthUnitChoice.")
    if len(choices) > 1:
        raise ValueError("Two depth-unit choices are in force.")
    if not choices:
        return None
    choice = choices[0]
    if choice.unit not in options:
        shown = ", ".join(options) if options else "none"
        raise ValueError(f"{choice.unit!r} is not an option this finding lists ({shown}).")
    return choice


def _bottom_up(depths: tuple[Decimal, ...]) -> bool:
    """True when every depth is lower than the one before it. D3."""

    return all(depths[index + 1] < depths[index] for index in range(len(depths) - 1))


def _any_exponent(tokens: tuple[str, ...]) -> bool:
    """True when a depth token is written with an exponent. LAS 2.0 section 3.0."""

    return any(_exponent(token) for token in tokens)


def _exponent(token: str) -> bool:
    text = token.strip()
    if text[:1] in "+-":
        text = text[1:]
    return "e" in text.lower()


def _decimals(tokens: tuple[str, ...]) -> tuple[Decimal, ...] | None:
    parsed: list[Decimal] = []
    for token in tokens:
        number = _decimal_or_none(token)
        if number is None:
            return None
        parsed.append(number)
    return tuple(parsed)


def _decimal_or_none(text: str | None) -> Decimal | None:
    if text is None or text.strip() == "":
        return None
    try:
        number = Decimal(text.strip())
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    return number


def _dec_text(number: Decimal) -> str:
    """The full decimal spelling. Evidence keeps this; messages do not."""

    return format(number, "f")


def _display_decimal(number: Decimal, places: int) -> str:
    """Fixed-point text with exactly ``places`` digits after the point.

    Messages use this. The evidence field keeps ``_dec_text``. Half-even
    rounding is Decimal's own rule for a digit that is exactly 5.
    """

    quantum = Decimal(1).scaleb(-places)
    return format(number.quantize(quantum), "f")


def _places(token: str) -> int | None:
    """Digits after the decimal point, or None when the token is not fixed-point."""

    text = token.strip()
    if text[:1] in "+-":
        text = text[1:]
    if text == "" or "e" in text.lower():
        return None
    if "." not in text:
        return 0
    fraction = text.split(".", 1)[1]
    if fraction != "" and not fraction.isdigit():
        return None
    return len(fraction)


def _finest_places(tokens: tuple[str, ...]) -> int | None:
    found: list[int] = []
    for token in tokens:
        count = _places(token)
        if count is None:
            return None
        found.append(count)
    if not found:
        return None
    return max(found)


def _tolerance(places: int) -> Decimal:
    """One full unit in that decimal place. depth.grid_rounding, 1 Oct 2026."""

    return Decimal(1).scaleb(-places)


def _first_duplicate(depths: tuple[Decimal, ...]) -> tuple[int, str] | None:
    for index in range(len(depths) - 1):
        if depths[index] == depths[index + 1]:
            return index, _dec_text(depths[index])
    return None


def _first_decrease(depths: tuple[Decimal, ...]) -> tuple[int, str, str] | None:
    for index in range(len(depths) - 1):
        if depths[index + 1] < depths[index]:
            return index, _dec_text(depths[index]), _dec_text(depths[index + 1])
    return None


def _on_declared_grid(depths: tuple[Decimal, ...], tokens: tuple[str, ...], step: Decimal) -> bool:
    """True when each depth is within its own allowance of the first depth plus index times STEP. D12.

    The allowance is the smaller of one unit of that depth's last written
    decimal and half the absolute step. The comparison is inclusive.
    """

    start = depths[0]
    half = abs(step) / 2
    for index, (depth, token) in enumerate(zip(depths, tokens)):
        places = _places(token)
        if places is None:
            return False
        allowance = min(_tolerance(places), half)
        grid = start + (Decimal(index) * step)
        if abs(depth - grid) > allowance:
            return False
    return True


def _increments(depths: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
    return tuple(depths[index + 1] - depths[index] for index in range(len(depths) - 1))


def _median(values: tuple[Decimal, ...]) -> Decimal:
    """The median. An even count is the mean of the two middle values. D12."""

    ordered = tuple(sorted(values))
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _gaps_within(tokens: tuple[str, ...], gaps: tuple[Decimal, ...], median: Decimal) -> bool:
    """True when every gap is within its allowance of the median. D12.

    The allowance is the smaller of one unit of the coarser last written
    decimal of the gap's two depths and half the median. Inclusive.
    """

    half = abs(median) / 2
    for index, gap in enumerate(gaps):
        left = _places(tokens[index])
        right = _places(tokens[index + 1])
        if left is None or right is None:
            return False
        allowance = min(_tolerance(min(left, right)), half)
        if abs(gap - median) > allowance:
            return False
    return True


@dataclass(frozen=True)
class _Standards:
    feet_aliases: frozenset[str]
    feet_steps: tuple[Decimal, ...]
    feet_primary: Decimal
    metre_aliases: frozenset[str]
    metre_steps: tuple[Decimal, ...]
    metre_primary: Decimal


def _standard_steps() -> _Standards:
    """The foot and metre lists. Numbers come from the rule file. No fill-in."""

    payload = load_rule(_STEP_RULE_ID)
    feet = _step_group(payload, "feet")
    metres = _step_group(payload, "metres")
    return _Standards(
        feet[0],
        feet[1],
        feet[2],
        metres[0],
        metres[1],
        metres[2],
    )


def _step_group(payload: dict[str, object], key: str) -> tuple[frozenset[str], tuple[Decimal, ...], Decimal]:
    if key not in payload:
        raise ValueError(f"{_STEP_RULE_ID} is missing {key!r}.")
    group = payload[key]
    if not isinstance(group, dict):
        raise ValueError(f"{_STEP_RULE_ID} {key} must be an object.")
    if "spellings" not in group or "steps" not in group or "primary" not in group:
        raise ValueError(f"{_STEP_RULE_ID} {key} is missing spellings, steps, or primary.")
    spellings = group["spellings"]
    steps = group["steps"]
    primary = group["primary"]
    if not isinstance(spellings, list) or not spellings:
        raise ValueError(f"{_STEP_RULE_ID} {key}.spellings must be a non-empty list.")
    aliases: list[str] = []
    for item in spellings:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or item["text"].strip() == "":
            raise ValueError(f"{_STEP_RULE_ID} {key} spellings must name a text.")
        aliases.append(item["text"])
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{_STEP_RULE_ID} {key}.steps must be a non-empty list.")
    parsed: list[Decimal] = []
    for item in steps:
        if not isinstance(item, str):
            raise ValueError(f"{_STEP_RULE_ID} {key} steps must be decimal strings.")
        number = _decimal_or_none(item)
        if number is None or number <= 0:
            raise ValueError(f"{_STEP_RULE_ID} step {item!r} must be a positive decimal.")
        parsed.append(number)
    if not isinstance(primary, str):
        raise ValueError(f"{_STEP_RULE_ID} {key}.primary must be a decimal string.")
    primary_number = _decimal_or_none(primary)
    if primary_number is None or primary_number not in parsed:
        raise ValueError(f"{_STEP_RULE_ID} {key}.primary must be one of its steps.")
    return frozenset(aliases), tuple(parsed), primary_number


def _family(unit: str, standards: _Standards) -> str | None:
    folded = unit.strip().casefold()
    if folded == "":
        return None
    feet = folded in {alias.casefold() for alias in standards.feet_aliases}
    metres = folded in {alias.casefold() for alias in standards.metre_aliases}
    if feet and metres:
        return None
    if feet:
        return "feet"
    if metres:
        return "metres"
    return None


def _same_family(left: str | None, right: str, standards: _Standards) -> bool:
    if left is None:
        return False
    left_family = _family(left, standards)
    right_family = _family(right, standards)
    return left_family is not None and left_family == right_family


def _step_unit(ingest: IngestResult, depth_unit: str) -> str:
    if ingest.audit.step_unit is not None and ingest.audit.step_unit.strip() != "":
        return ingest.audit.step_unit
    return depth_unit


def _primary(family: str, unit: str, standards: _Standards) -> DerivedQuantity:
    if family == "feet":
        text = _dec_text(standards.feet_primary)
    elif family == "metres":
        text = _dec_text(standards.metre_primary)
    else:
        raise ValueError(f"No primary step is defined for family {family!r}.")
    quantity = DerivedQuantity("primary_standard_step", text, unit)
    assert_derived_unit(quantity, unit)
    return quantity


def _declared_step(
    ingest: IngestResult,
    step: Decimal | None,
    unit: str,
    standards: _Standards,
    family: str | None,
) -> DerivedQuantity | None:
    if step is None or ingest.audit.step_text is None:
        return None
    step_unit = _step_unit(ingest, unit)
    if family is not None and not _same_family(ingest.audit.step_unit, unit, standards):
        return DerivedQuantity("declared_step", _dec_text(step), step_unit)
    quantity = DerivedQuantity("declared_step", _dec_text(step), step_unit)
    if _same_family(step_unit, unit, standards) or step_unit == unit:
        assert_derived_unit(quantity, step_unit)
    return quantity


def _note_strt(ledger: _Ledger, strt_text: str | None) -> None:
    """Record a blank or non-numeric STRT. The grid is still checked from the first depth. D12."""

    if strt_text is None or strt_text.strip() == "":
        ledger.add(
            "strt_blank",
            (),
            "STRT is blank. The grid was checked from the first depth.",
        )
        return
    if _decimal_or_none(strt_text) is None:
        shown = strt_text.strip()
        ledger.add(
            "strt_not_a_number",
            (),
            f"STRT is not a number: '{shown}'.",
            {"observed": shown},
        )


def _median_grid(
    ledger: _Ledger,
    depths: tuple[Decimal, ...],
    tokens: tuple[str, ...],
    unit: str,
    step_is_zero: bool,
) -> tuple[DepthRegularity, DerivedQuantity | None]:
    """Judge a blank or zero STEP against the median gap. D12, D67.

    A grid that passes records the median as the data-derived step. A grid
    that fails is irregular. The median is named in the finding and is not
    stored as a validated step.
    """

    gaps = _increments(depths)
    median = _median(gaps)
    median_text = _dec_text(median)
    finest = _finest_places(tokens)
    median_shown = _display_decimal(median, finest + 2) if finest is not None else median_text
    if _gaps_within(tokens, gaps, median):
        quantity = DerivedQuantity("observed_step", median_text, unit)
        assert_derived_unit(quantity, unit)
        if step_is_zero:
            ledger.add(
                "step_zero",
                (),
                f"STEP is 0. Data-derived step {median_shown} {unit} was recorded.",
                {"declared": "0", "observed": median_text},
            )
        else:
            ledger.add(
                "step_blank_regular",
                (),
                f"STEP is blank. Data-derived step {median_shown} {unit} was recorded.",
                {"observed": median_text},
            )
        return DepthRegularity.REGULAR, quantity
    if step_is_zero:
        ledger.add(
            "step_zero",
            (),
            "STEP is 0. The header value was recorded. The grid was not given an invented step.",
            {"declared": "0", "observed": median_text},
        )
    else:
        ledger.add(
            "step_blank_irregular",
            (),
            (
                "STEP is blank. Depth increments are not constant. The grid is irregular. "
                f"The median gap is {median_shown}. No step was validated."
            ),
            {"observed": median_text},
        )
    return DepthRegularity.IRREGULAR, None


def _declared_grid(
    ledger: _Ledger,
    depths: tuple[Decimal, ...],
    tokens: tuple[str, ...],
    step: Decimal,
    unit: str,
    ingest: IngestResult,
) -> tuple[DepthRegularity, DerivedQuantity | None]:
    """Judge a declared non-zero STEP from the first depth. D12, D88.

    A miss is irregular and informational. No step is recorded as validated.
    """

    step_text = ingest.audit.step_text or _dec_text(step)
    if _on_declared_grid(depths, tokens, step):
        quantity = DerivedQuantity("observed_step", _dec_text(step), _step_unit(ingest, unit))
        return DepthRegularity.REGULAR, quantity
    ledger.add(
        "grid_off_declared_step",
        (),
        (
            f"Declared STEP {step_text} does not put every depth on the first depth + i × STEP "
            "within the rounding allowance. The ascending grid is irregular and was not modified."
        ),
        {"declared": step_text},
    )
    return DepthRegularity.IRREGULAR, None


def _compared_spacing(depths: tuple[Decimal, ...]) -> tuple[str, Decimal]:
    """The interval the standard-step test uses. D12.

    ``(last depth − first depth) / (NSAMPLES − 1)``, from the data. Header
    STRT and STOP are not used. The endpoint check judges them. In a
    conforming file the two agree, because LAS 2.0 section 5.4 requires
    STRT to be the first depth and STOP the last.
    """

    return "implied_interval", (depths[-1] - depths[0]) / Decimal(len(depths) - 1)


def _interval_uncertainty(tokens: tuple[str, ...], count: int) -> Decimal | None:
    """How precisely the implied interval is known. D12.

    One unit of the first depth's last written decimal, plus one unit of the
    last depth's last written decimal, divided by the number of gaps. None
    when either endpoint is not fixed-point.
    """

    if count < 2:
        return None
    first = _places(tokens[0])
    last = _places(tokens[-1])
    if first is None or last is None:
        return None
    return (_tolerance(first) + _tolerance(last)) / Decimal(count - 1)


def _quantity_label(name: str) -> str:
    if name == "implied_interval":
        return "Implied interval"
    if name == "median_gap":
        return "Median gap"
    raise ValueError(f"Unknown spacing quantity {name!r}.")


def _standard_step(
    ledger: _Ledger,
    ingest: IngestResult,
    depths: tuple[Decimal, ...],
    tokens: tuple[str, ...],
    compared_name: str,
    compared: Decimal,
    unit: str,
    family: str,
    finest: int | None,
    standards: _Standards,
) -> tuple[DerivedQuantity | None, DerivedQuantity | None]:
    """Compare the implied interval with the standard steps. D12, D89.

    The interval is taken from the first and last depths in the data. It is
    known only to ``_interval_uncertainty``. Within that of one standard step,
    and when the header STEP is already that step, the interval is that step:
    endpoint rounding is not a correction. A wider gap is not rounding, so the
    half-step gate is not offered a correction. A different written step that
    is still within the uncertainty keeps the rounding correction. Samples
    stay put. The finding records the quantity, and the evidence keeps the
    full value.
    """

    if finest is None:
        return None, None
    options = standards.feet_steps if family == "feet" else standards.metre_steps
    uncertainty = _interval_uncertainty(tokens, len(depths))
    nearest, tied = _nearest(compared, options)
    header_step = _decimal_or_none(ingest.audit.step_text)
    if (
        not tied
        and nearest is not None
        and uncertainty is not None
        and abs(compared - nearest) <= uncertainty
        and header_step is not None
        and header_step == nearest
    ):
        compared = nearest
    compared_text = _dec_text(compared)
    shown = _display_decimal(compared, finest + 2)
    label = _quantity_label(compared_name)
    origin = ", from the first and last depths in the data," if compared_name == "implied_interval" else ""
    notice = (
        f"{label} ({shown} {unit}){origin} is not a recognised standard interval; depths not modified."
    )
    common = {"compared_quantity": compared_name, "compared_value": compared_text, "declared": compared_text}
    if tied or nearest is None:
        ledger.add("step_not_standard", (), notice, common)
        return None, None
    gap = abs(compared - nearest)
    if gap == 0:
        return None, None
    rounds = gap <= _tolerance(finest)
    if uncertainty is not None and gap > uncertainty:
        rounds = False
    start = _decimal_or_none(ingest.audit.strt_text)
    if start is None:
        if not rounds:
            ledger.add(
                "step_not_standard",
                (),
                notice,
                {**common, "observed": _dec_text(nearest)},
            )
        return None, None
    last = depths[-1]
    reconstructed = start + (Decimal(len(depths) - 1) * nearest)
    shift = last - reconstructed
    gate = abs(shift) <= (Decimal("0.5") * nearest)
    if rounds and gate:
        corrected = DerivedQuantity("corrected_step", _dec_text(nearest), unit)
        displacement = DerivedQuantity("displacement", _dec_text(shift), unit)
        assert_derived_unit(corrected, unit)
        assert_derived_unit(displacement, unit)
        ledger.add(
            "step_corrected",
            (),
            (
                f"{label} {shown} {unit}{origin} corrected to {_dec_text(nearest)} {unit}; "
                f"displacement {_dec_text(shift)} {unit}."
            ),
            {
                **common,
                "observed": _dec_text(nearest),
                "difference": _dec_text(shift),
            },
        )
        return corrected, displacement
    ledger.add(
        "step_not_standard",
        (),
        notice,
        {
            **common,
            "observed": _dec_text(nearest),
            "difference": _dec_text(abs(compared - nearest)),
        },
    )
    return None, None


def _nearest(step: Decimal, options: tuple[Decimal, ...]) -> tuple[Decimal | None, bool]:
    """The unique nearest standard step. A tie is not chosen."""

    ranked = sorted(((abs(option - step), option) for option in options), key=lambda item: (item[0], item[1]))
    best = ranked[0][0]
    tied = [option for distance, option in ranked if distance == best]
    if len(tied) != 1:
        return None, True
    return tied[0], False


def _endpoints(
    ledger: _Ledger,
    ingest: IngestResult,
    depths: tuple[Decimal, ...],
    step: Decimal | None,
    observed: DerivedQuantity | None,
    unit: str,
    standards: _Standards,
) -> tuple[DerivedQuantity | None, DerivedQuantity | None]:
    interval = _sample_interval(step, observed, unit)
    recorded_strt = _one_endpoint(
        ledger,
        label="STRT",
        declared_text=ingest.audit.strt_text,
        declared_unit=ingest.audit.strt_unit,
        observed=depths[0],
        interval=interval,
        depth_unit=unit,
        standards=standards,
    )
    recorded_stop = _one_endpoint(
        ledger,
        label="STOP",
        declared_text=ingest.audit.stop_text,
        declared_unit=ingest.audit.stop_unit,
        observed=depths[-1],
        interval=interval,
        depth_unit=unit,
        standards=standards,
    )
    return recorded_strt, recorded_stop


def _sample_interval(step: Decimal | None, observed: DerivedQuantity | None, unit: str) -> Decimal | None:
    if step is not None and step != 0:
        return abs(step)
    if observed is None:
        return None
    assert_derived_unit(observed, observed.unit)
    if observed.unit != unit and observed.unit.strip() != "":
        return None
    number = Decimal(observed.text)
    if number == 0:
        return None
    return abs(number)


def _one_endpoint(
    ledger: _Ledger,
    *,
    label: str,
    declared_text: str | None,
    declared_unit: str | None,
    observed: Decimal,
    interval: Decimal | None,
    depth_unit: str,
    standards: _Standards,
) -> DerivedQuantity | None:
    declared = _decimal_or_none(declared_text)
    observed_quantity = DerivedQuantity(f"observed_{label.lower()}", _dec_text(observed), depth_unit)
    assert_derived_unit(observed_quantity, depth_unit)
    if declared is None or declared_text is None:
        return observed_quantity
    if declared_unit is not None and declared_unit.strip() != "" and not _units_match(declared_unit, depth_unit, standards):
        ledger.add(
            "endpoint_unit_mismatch",
            (("endpoint", label),),
            f"{label} unit {declared_unit!r} is not the depth unit {depth_unit!r}. The values were not subtracted.",
            {"declared": declared_text, "observed": _dec_text(observed)},
        )
        return None
    difference = declared - observed
    if difference == 0:
        return observed_quantity
    if interval is None:
        ledger.add(
            "endpoint_not_compared",
            (("endpoint", label),),
            f"{label} {declared_text} was not compared with the data because no sample interval is established.",
            {"declared": declared_text, "observed": _dec_text(observed), "difference": _dec_text(difference)},
        )
        return observed_quantity
    intervals = difference / interval
    intervals_text = _dec_text(intervals)
    intervals_shown = _display_decimal(intervals, 2)
    fields = {
        "declared": declared_text.strip(),
        "observed": _dec_text(observed),
        "difference": _dec_text(difference),
        "difference_in_step_intervals": intervals_text,
    }
    if abs(difference) <= interval:
        ledger.add(
            "endpoint_within_step",
            (("endpoint", label),),
            (
                f"{label} {declared_text.strip()} differs from the data {_dec_text(observed)} "
                f"by {_dec_text(difference)} ({intervals_shown} step intervals)."
            ),
            fields,
        )
        return observed_quantity
    ledger.add(
        "endpoint_beyond_step",
        (("endpoint", label),),
        (
            f"{label} {declared_text.strip()} differs from the data {_dec_text(observed)} "
            f"by {_dec_text(difference)}, which is beyond one step."
        ),
        fields,
    )
    return None


def _units_match(declared_unit: str, depth_unit: str, standards: _Standards) -> bool:
    """True when the two unit texts are the same, or both name one depth family. D12.

    Identical text matches even when neither spelling is recognised, so the
    values can be subtracted. A recognised spelling still matches its family.
    """

    if declared_unit.strip().casefold() == depth_unit.strip().casefold():
        return True
    return _same_family(declared_unit, depth_unit, standards)


def _finish_structural(result: DepthAxisResult) -> DepthAxisResult:
    """A later endpoint failure overrides COMPLETE. Direction is not stamped."""

    if not any(item.block_type is BlockType.STRUCTURAL_DATA_FAILURE for item in result.findings):
        return result
    return DepthAxisResult(
        StageStatus.REFUSED_STRUCTURAL,
        result.findings,
        None,
        None,
        result.declared_step,
        result.observed_step,
        None,
        None,
        None,
        None,
        result.alias,
        None,
        result.depth_unit,
        result.depth_unit_decision_id,
    )
