"""Stage 1: classify the depth axis. It does not rewrite samples.

Implements D3 (F14, then ascending only), D8, D9 (record the alias, do not
copy the curve), D28, F10, D67 (record STEP 0, do not enforce it), D86, D88
as revised to ascending only, D89, C4, and contracts section 4 item 5.

Structure
---------
CONTRACT
    Stage 1 contract. alters_data is False. depends_on_stages is (0,).
depth_index_mnemonics
    The small index-alias set. It stays in this module (D28, F10).
DerivedQuantity
    A number stored with the unit it was derived under. C4.
classify
    Read an ingest result and return a DepthAxisResult.
registered_stage
    The same classification behind the stage runner.

Direction is stamped ASCENDING on an accepted result. DESCENDING is never
stamped. Depths are not reversed, re-sorted, or replaced. A corrected step
is a recorded quantity for a later export. This stage does not write it
onto the samples. Stages 0–4 cannot alter data (D24).

Irregular-but-ascending grids are accepted (D67, the revised D88). A declared
STEP that the samples do not meet is INFORMATIONAL, not a structural failure.
The build plan says the same (P09 item 4, grid-inconsistent bullet).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from polish.model.dataset import DepthDirection, DepthRegularity
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
        "Depths decrease, repeat, or the header STEP is negative.",
        "STRT or STOP disagrees with the first or last depth by more than one sample interval.",
    ),
    depends_on_stages=(0,),
    alters_data=False,
)


def registered_stage(ingest: IngestResult) -> RegisteredStage:
    """A runner stage that classifies ``ingest`` and does not replace it."""

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
    ("step_negative", BlockType.STRUCTURAL_DATA_FAILURE),
    ("depth_unit_unrecognised", BlockType.AMBIGUOUS_INTERPRETATION),
    ("step_zero", BlockType.INFORMATIONAL),
    ("step_blank_regular", BlockType.INFORMATIONAL),
    ("step_blank_irregular", BlockType.INFORMATIONAL),
    ("strt_blank", BlockType.INFORMATIONAL),
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


def classify(ingest: IngestResult) -> DepthAxisResult:
    """Classify the depth axis of one Stage 0 result.

    Implements D3, D8, D9, D28, F10, D67, D86, D88 (ascending only), D89, D96, C4.
    Does not reverse, re-sort, or copy a curve. Does not open the network.
    """

    if not isinstance(ingest, IngestResult):
        raise TypeError("ingest must be an IngestResult.")
    load_rule(_GRID_RULE_ID)
    standards = _standard_steps()
    ledger = _Ledger(ingest)
    curves = ingest.audit.curves
    tokens = ingest.audit.depth_tokens
    if not curves:
        ledger.add("depth_no_curve", (), "The file has no curve to use as a depth index.")
        return _refused(ledger)

    mnemonic = curves[0].mnemonic
    unit = curves[0].unit
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
        return _refused(ledger, alias=alias, depth_unit=unit)

    parsed = _decimals(tokens)
    if parsed is None:
        ledger.add(
            "depth_token_not_decimal",
            (),
            "A depth token is not a finite decimal, so the axis was not classified.",
        )
        return _refused(ledger, alias=alias, depth_unit=unit)
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
        return _refused(ledger, alias=alias, depth_unit=unit)

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
            f"STEP {ingest.audit.step_text} is negative. Descending depth is a structural failure.",
            {"declared": ingest.audit.step_text or ""},
        )
    if ledger.items and any(item.block_type is BlockType.STRUCTURAL_DATA_FAILURE for item in ledger.items):
        return _refused(ledger, alias=alias, depth_unit=unit)

    family = _family(unit, standards)
    if family is None:
        shown = unit if unit.strip() != "" else ""
        ledger.add(
            "depth_unit_unrecognised",
            (),
            f"Depth unit {shown!r} is absent or not recognised. Feet or metres were not assumed.",
            {"observed": shown},
        )

    finest = _finest_places(tokens)
    declared = _declared_step(ingest, step, unit, standards, family)
    observed: DerivedQuantity | None = None
    corrected: DerivedQuantity | None = None
    displacement: DerivedQuantity | None = None
    regularity: DepthRegularity | None = None
    primary = _primary(family, unit, standards) if family is not None else None

    if step is None:
        regularity, observed = _blank_step(ledger, depths, unit, finest)
    elif step == 0:
        regularity = DepthRegularity.IRREGULAR
        ledger.add(
            "step_zero",
            (),
            "STEP is 0. The header value was recorded. The grid was not given an invented step.",
            {"declared": "0"},
        )
        declared = DerivedQuantity("declared_step", "0", _step_unit(ingest, unit))
    else:
        regularity, observed = _declared_grid(ledger, ingest, depths, tokens, step, unit)
        if family is not None and _same_family(ingest.audit.step_unit, unit, standards):
            corrected, displacement = _standard_step(
                ledger, ingest, depths, step, unit, family, finest, standards
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
    )


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
    return format(number, "f")


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


def _on_declared_grid(depths: tuple[Decimal, ...], tokens: tuple[str, ...], start: Decimal, step: Decimal) -> bool:
    """True when each written depth is within one unit of its own last place."""

    for index, (depth, token) in enumerate(zip(depths, tokens)):
        places = _places(token)
        if places is None:
            return False
        grid = start + (Decimal(index) * step)
        if abs(depth - grid) > _tolerance(places):
            return False
    return True


def _increments(depths: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
    return tuple(depths[index + 1] - depths[index] for index in range(len(depths) - 1))


def _increments_constant(depths: tuple[Decimal, ...], finest: int | None) -> bool:
    if finest is None:
        return False
    increments = _increments(depths)
    if not increments:
        return False
    return max(increments) - min(increments) <= _tolerance(finest)


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
    if "unit_aliases" not in group or "steps" not in group or "primary" not in group:
        raise ValueError(f"{_STEP_RULE_ID} {key} is missing unit_aliases, steps, or primary.")
    aliases = group["unit_aliases"]
    steps = group["steps"]
    primary = group["primary"]
    if not isinstance(aliases, list) or not aliases or not all(isinstance(item, str) and item.strip() for item in aliases):
        raise ValueError(f"{_STEP_RULE_ID} {key}.unit_aliases must be a non-empty list of strings.")
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


def _blank_step(
    ledger: _Ledger,
    depths: tuple[Decimal, ...],
    unit: str,
    finest: int | None,
) -> tuple[DepthRegularity, DerivedQuantity | None]:
    if _increments_constant(depths, finest):
        derived = _increments(depths)[0]
        quantity = DerivedQuantity("observed_step", _dec_text(derived), unit)
        assert_derived_unit(quantity, unit)
        ledger.add(
            "step_blank_regular",
            (),
            f"STEP is blank. Data-derived step {_dec_text(derived)} {unit} was recorded.",
            {"observed": _dec_text(derived)},
        )
        return DepthRegularity.REGULAR, quantity
    ledger.add(
        "step_blank_irregular",
        (),
        "STEP is blank. Depth increments are not constant. The grid is irregular.",
    )
    return DepthRegularity.IRREGULAR, None


def _declared_grid(
    ledger: _Ledger,
    ingest: IngestResult,
    depths: tuple[Decimal, ...],
    tokens: tuple[str, ...],
    step: Decimal,
    unit: str,
) -> tuple[DepthRegularity | None, DerivedQuantity | None]:
    start = _decimal_or_none(ingest.audit.strt_text)
    step_text = ingest.audit.step_text or _dec_text(step)
    if start is None:
        ledger.add(
            "strt_blank",
            (),
            "STRT is blank, so the grid STRT + i × STEP was not reconstructed. The ascending samples were left as written.",
            {"declared": step_text},
        )
        return DepthRegularity.IRREGULAR, None
    if _on_declared_grid(depths, tokens, start, step):
        quantity = DerivedQuantity("observed_step", _dec_text(step), _step_unit(ingest, unit))
        return DepthRegularity.REGULAR, quantity
    ledger.add(
        "grid_off_declared_step",
        (),
        (
            f"Declared STEP {step_text} does not put every depth on STRT + i × STEP "
            "within the grid rounding tolerance. The ascending grid is irregular and was not modified."
        ),
        {"declared": step_text},
    )
    return DepthRegularity.IRREGULAR, None


def _standard_step(
    ledger: _Ledger,
    ingest: IngestResult,
    depths: tuple[Decimal, ...],
    step: Decimal,
    unit: str,
    family: str,
    finest: int | None,
    standards: _Standards,
) -> tuple[DerivedQuantity | None, DerivedQuantity | None]:
    """Record a D89 correction, or the non-standard notice. Samples stay put."""

    if finest is None:
        return None, None
    options = standards.feet_steps if family == "feet" else standards.metre_steps
    nearest, tied = _nearest(step, options)
    step_text = ingest.audit.step_text or _dec_text(step)
    step_unit = _step_unit(ingest, unit)
    if tied or nearest is None:
        ledger.add(
            "step_not_standard",
            (),
            f"step as written ({step_text} {step_unit}) is not a recognised standard interval; depths not modified.",
            {"declared": step_text},
        )
        return None, None
    tolerance = _tolerance(finest)
    gap = abs(step - nearest)
    if gap == 0:
        return None, None
    rounds = gap <= tolerance
    start = _decimal_or_none(ingest.audit.strt_text)
    if start is None:
        if not rounds:
            ledger.add(
                "step_not_standard",
                (),
                f"step as written ({step_text} {step_unit}) is not a recognised standard interval; depths not modified.",
                {"declared": step_text, "observed": _dec_text(nearest)},
            )
        return None, None
    last = depths[-1]
    reconstructed = start + (Decimal(len(depths) - 1) * nearest)
    shift = last - reconstructed
    gate = abs(shift) <= (Decimal("0.5") * nearest)
    if rounds and gate:
        corrected = DerivedQuantity("corrected_step", _dec_text(nearest), step_unit)
        displacement = DerivedQuantity("displacement", _dec_text(shift), step_unit)
        assert_derived_unit(corrected, step_unit)
        assert_derived_unit(displacement, step_unit)
        ledger.add(
            "step_corrected",
            (),
            (
                f"Declared step {step_text} {step_unit} corrected to {_dec_text(nearest)} {step_unit}; "
                f"displacement {_dec_text(shift)} {step_unit}."
            ),
            {
                "declared": step_text,
                "observed": _dec_text(nearest),
                "difference": _dec_text(shift),
            },
        )
        return corrected, displacement
    ledger.add(
        "step_not_standard",
        (),
        f"step as written ({step_text} {step_unit}) is not a recognised standard interval; depths not modified.",
        {
            "declared": step_text,
            "observed": _dec_text(nearest),
            "difference": _dec_text(abs(step - nearest)),
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
    if declared_unit is not None and declared_unit.strip() != "" and not _same_family(declared_unit, depth_unit, standards):
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
    fields = {
        "declared": declared_text.strip(),
        "observed": _dec_text(observed),
        "difference": _dec_text(difference),
        "difference_in_step_intervals": _dec_text(intervals),
    }
    if abs(difference) <= interval:
        ledger.add(
            "endpoint_within_step",
            (("endpoint", label),),
            (
                f"{label} {declared_text.strip()} differs from the data {_dec_text(observed)} "
                f"by {_dec_text(difference)} ({_dec_text(intervals)} step intervals)."
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
    )
