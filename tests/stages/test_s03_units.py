"""Stage 3 units. C1, C4, C6, D87, D90, OD-05.

Recognition counts and conversion factors are read from the shipped tables.
They are not re-typed as a second source of the same numbers.
"""

from __future__ import annotations

import datetime
import json
from fractions import Fraction
from pathlib import Path

import pytest

from polish.model.states import (
    BlockType,
    ProcessingEligibility,
    StageStatus,
    UnitInterpretation,
    UserAction,
)
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, Runner, StageOutcome
from polish.stages.s00_ingest import ingest
from polish.stages.s01_depth import DerivedQuantity, assert_derived_unit, classify
from polish.stages.s01_depth import registered_stage as depth_stage
from polish.stages.s02_identify import identify
from polish.stages.s02_identify import registered_stage as identify_stage
from polish.stages.s03_units import (
    CONTRACT,
    STAGE_4_DEPENDS_ON,
    TargetUnitSystem,
    UnitMismatch,
    declared_unit_state,
    declare_depth_conversion,
    incompatible_unit_actions,
    interpret,
    read_values,
    registered_stage,
    require_unit,
    units_eligibility,
)
from polish.tables.loader import _family_record, load_unit_spellings

_WHEN = datetime.datetime(2026, 10, 12, tzinfo=datetime.timezone.utc)


def _write(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "well.las"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _las(curves: str, data: str, unit: str = "M", step: str = "1.0", stop: str = "1.0") -> str:
    return (
        "~VERSION INFORMATION\n"
        "VERS. 2.0 : CWLS\n"
        "WRAP. NO : WRAP\n"
        "~WELL INFORMATION\n"
        f"STRT.{unit} 0.0 : START\n"
        f"STOP.{unit} {stop} : STOP\n"
        f"STEP.{unit} {step} : STEP\n"
        "NULL. -999.25 : NULL\n"
        "~CURVE INFORMATION\n"
        f"{curves}"
        "~A\n"
        f"{data}"
    )


def _curve_block(specs: list[tuple[str, str]], depth_unit: str = "M") -> str:
    lines = [f"DEPT.{depth_unit} : DEPTH"]
    for mnemonic, unit in specs:
        lines.append(f"{mnemonic}.{unit} : {mnemonic}")
    return "\n".join(lines) + "\n"


def _data(samples: list[list[str]]) -> str:
    return "\n".join(" ".join(row) for row in samples) + "\n"


def _run(directory: Path, specs: list[tuple[str, str]], samples: list[list[str]], depth_unit: str = "M"):
    step = "0.5" if depth_unit == "FT" else "1.0"
    stop = "0.5" if depth_unit == "FT" else "1.0"
    body = _las(_curve_block(specs, depth_unit), _data(samples), unit=depth_unit, step=step, stop=stop)
    ingested = ingest(_write(directory, body), read_at=_WHEN)
    assert ingested.dataset is not None
    before = tuple(curve.values.tobytes() for curve in ingested.dataset.curves)
    depth = classify(ingested)
    assert depth.status is StageStatus.COMPLETE
    local = directory / "no-local-tables"
    identified = identify(ingested, depth, local_tables_directory=local)
    assert identified.status is StageStatus.COMPLETE
    result = interpret(ingested, identified)
    assert result.status is StageStatus.COMPLETE
    assert result.target_system is TargetUnitSystem.AS_RECORDED
    assert tuple(curve.values.tobytes() for curve in ingested.dataset.curves) == before
    return ingested, identified, result


def _curve(result, mnemonic: str):
    matches = [item for item in result.curves if item.mnemonic == mnemonic]
    assert len(matches) == 1, mnemonic
    return matches[0]


def _findings(result, block: BlockType, mnemonic: str):
    return [
        finding
        for finding in result.findings
        if finding.block_type is block and finding.evidence.as_dict().get("mnemonic") == mnemonic
    ]


def test_contract_is_read_only_and_names_the_stage_4_dependency():
    assert CONTRACT.stage_number == 3
    assert CONTRACT.alters_data is False
    assert CONTRACT.depends_on_stages == (0, 1, 2)
    assert STAGE_4_DEPENDS_ON == ("unit_interpretation", "target_system")
    assert "AS_RECORDED" in CONTRACT.guarantees.units


def test_absent_empty_and_unrecognised_are_distinct(tmp_path: Path):
    assert declared_unit_state(None) is UnitInterpretation.ABSENT
    assert declared_unit_state("") is UnitInterpretation.EMPTY
    assert declared_unit_state("   ") is UnitInterpretation.EMPTY
    assert units_eligibility(ProcessingEligibility.ELIGIBLE, UnitInterpretation.ABSENT) is (
        ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED
    )
    ingested, _identified, result = _run(
        tmp_path,
        [("GR", ""), ("ZZ", "NOTAUNIT")],
        [["0.0", "1.0", "2.0"], ["1.0", "3.0", "4.0"]],
    )
    empty = _curve(result, "GR")
    unknown = _curve(result, "ZZ")
    assert empty.outcome is UnitInterpretation.EMPTY
    assert empty.canonical_unit is None
    assert empty.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED
    assert _findings(result, BlockType.AMBIGUOUS_INTERPRETATION, "GR")
    assert unknown.outcome is UnitInterpretation.UNRECOGNISED
    assert unknown.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_LOW_TIER
    notice = _findings(result, BlockType.AMBIGUOUS_INTERPRETATION, "ZZ")
    assert notice
    assert "NOTAUNIT" in notice[0].message
    curve = ingested.dataset.curves[1]
    object.__setattr__(curve, "original_unit", None)
    depth = classify(ingested)
    identified = identify(ingested, depth, local_tables_directory=tmp_path / "no-local-tables")
    absent = interpret(ingested, identified)
    gr = _curve(absent, "GR")
    assert gr.outcome is UnitInterpretation.ABSENT
    assert gr.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED
    assert _findings(absent, BlockType.AMBIGUOUS_INTERPRETATION, "GR")


def test_gamma_ray_gapi_and_cps_are_both_compatible_and_not_converted(tmp_path: Path):
    _ingested, _identified, gapi = _run(tmp_path / "gapi", [("GR", "GAPI")], [["0.0", "40.0"], ["1.0", "50.0"]])
    recognised = _curve(gapi, "GR")
    assert recognised.outcome is UnitInterpretation.RECOGNISED
    assert recognised.canonical_unit == "GAPI"
    assert recognised.dimension == "gamma_api"
    assert recognised.processing_eligibility is ProcessingEligibility.ELIGIBLE
    assert _findings(gapi, BlockType.AMBIGUOUS_INTERPRETATION, "GR") == []

    ingested, _identified, cps = _run(tmp_path / "cps", [("GR", "cps")], [["0.0", "80.0"], ["1.0", "90.0"]])
    count_rate = _curve(cps, "GR")
    assert count_rate.outcome is UnitInterpretation.RECOGNISED
    assert count_rate.canonical_unit == "CPS"
    assert count_rate.dimension == "count_rate"
    assert count_rate.processing_eligibility is ProcessingEligibility.ELIGIBLE
    before = ingested.dataset.curves[1].values.tobytes()
    view = read_values(ingested.dataset, count_rate, "CPS")
    assert view.conversion is None
    assert view.unit == "CPS"
    with pytest.raises(UnitMismatch) as mismatch:
        require_unit(view, "GAPI")
    assert mismatch.value.reason
    with pytest.raises(UnitMismatch) as refused:
        read_values(ingested.dataset, count_rate, "GAPI")
    assert "No fixed factor" in refused.value.reason
    assert "gamma_api" in refused.value.reason
    assert ingested.dataset.curves[1].values.tobytes() == before
    assert ingested.dataset.curves[1].values.flags.writeable is False


def test_api_on_gamma_ray_is_gamma_units_not_gravity(tmp_path: Path):
    _ingested, _identified, result = _run(tmp_path, [("GR", "API")], [["0.0", "40.0"], ["1.0", "41.0"]])
    gr = _curve(result, "GR")
    assert gr.outcome is UnitInterpretation.RECOGNISED
    assert gr.canonical_unit == "GAPI"
    assert gr.dimension == "gamma_api"
    assert gr.candidates == ()


def test_api_with_no_single_accepted_dimension_stays_unrecognised(tmp_path: Path):
    _ingested, _identified, result = _run(tmp_path, [("RHOB", "API")], [["0.0", "2.5"], ["1.0", "2.6"]])
    rhob = _curve(result, "RHOB")
    assert rhob.outcome is UnitInterpretation.UNRECOGNISED
    assert rhob.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED
    dimensions = {item.dimension for item in rhob.candidates}
    assert dimensions == {"gamma_api", "api_gravity"}
    notice = _findings(result, BlockType.AMBIGUOUS_INTERPRETATION, "RHOB")
    assert notice
    listed = json.loads(str(notice[0].evidence.as_dict()["candidates"]))
    assert {row["dimension"] for row in listed} == {"gamma_api", "api_gravity"}


def test_incompatible_unit_stays_eligible_and_offers_an_explicit_mapping(tmp_path: Path):
    _ingested, _identified, result = _run(tmp_path, [("RHOB", "GAPI")], [["0.0", "2.5"], ["1.0", "2.6"]])
    rhob = _curve(result, "RHOB")
    assert rhob.outcome is UnitInterpretation.INCOMPATIBLE_WITH_FAMILY
    assert rhob.dimension == "gamma_api"
    assert rhob.processing_eligibility is ProcessingEligibility.ELIGIBLE
    warning = _findings(result, BlockType.RULE_DEPENDENT_WARNING, "RHOB")
    assert len(warning) == 1
    assert warning[0].evidence.as_dict()["dimension"] == "gamma_api"
    assert UserAction.MAP_EXPLICITLY in incompatible_unit_actions()


def test_each_eligibility_branch(tmp_path: Path):
    assert units_eligibility(
        ProcessingEligibility.NOT_ELIGIBLE_LOW_TIER, UnitInterpretation.RECOGNISED
    ) is ProcessingEligibility.NOT_ELIGIBLE_LOW_TIER
    assert units_eligibility(
        ProcessingEligibility.NOT_ELIGIBLE_NO_FAMILY, UnitInterpretation.ABSENT
    ) is ProcessingEligibility.NOT_ELIGIBLE_NO_FAMILY
    assert units_eligibility(
        ProcessingEligibility.NOT_ELIGIBLE_ROLE, UnitInterpretation.RECOGNISED
    ) is ProcessingEligibility.NOT_ELIGIBLE_ROLE
    assert units_eligibility(ProcessingEligibility.ELIGIBLE, UnitInterpretation.EMPTY) is (
        ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED
    )
    assert units_eligibility(ProcessingEligibility.ELIGIBLE, UnitInterpretation.UNRECOGNISED) is (
        ProcessingEligibility.NOT_ELIGIBLE_UNIT_UNRESOLVED
    )
    assert units_eligibility(ProcessingEligibility.ELIGIBLE, UnitInterpretation.RECOGNISED) is (
        ProcessingEligibility.ELIGIBLE
    )
    assert units_eligibility(
        ProcessingEligibility.ELIGIBLE, UnitInterpretation.INCOMPATIBLE_WITH_FAMILY
    ) is ProcessingEligibility.ELIGIBLE

    local = tmp_path / "local-tables"
    local.mkdir()
    (local / "mnemonics.json").write_text(
        json.dumps({"entries": [{"mnemonic": "SHOP_FLAG", "user_label": "shop"}]}),
        encoding="utf-8",
    )
    body = _las(_curve_block([("SHOP_FLAG", "GAPI"), ("DT_SYNT", "US/FT")]), _data([["0.0", "1.0", "2.0"], ["1.0", "1.0", "2.0"]]))
    ingested = ingest(_write(tmp_path, body), read_at=_WHEN)
    assert ingested.dataset is not None
    depth = classify(ingested)
    identified = identify(ingested, depth, local_tables_directory=local)
    result = interpret(ingested, identified)
    flag = _curve(result, "SHOP_FLAG")
    synthetic = _curve(result, "DT_SYNT")
    assert flag.outcome is UnitInterpretation.RECOGNISED
    assert flag.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_NO_FAMILY
    assert synthetic.outcome is UnitInterpretation.RECOGNISED
    assert synthetic.processing_eligibility is ProcessingEligibility.NOT_ELIGIBLE_ROLE


def test_density_conversion_is_not_compared_in_the_old_unit(tmp_path: Path):
    ingested, _identified, result = _run(
        tmp_path,
        [("RHOB", "G/CC")],
        [["0.0", "2.5"], ["1.0", "2.5"]],
    )
    rhob = _curve(result, "RHOB")
    assert rhob.outcome is UnitInterpretation.RECOGNISED
    assert rhob.canonical_unit == "G/CC"
    before = tuple(curve.values.tobytes() for curve in ingested.dataset.curves)
    view = read_values(ingested.dataset, rhob, "KG/M3")
    assert view.unit == "KG/M3"
    assert view.conversion is not None
    assert view.conversion.factor == Fraction(1000, 1)
    assert view.values[0] == pytest.approx(2500.0)
    with pytest.raises(UnitMismatch) as mismatch:
        require_unit(view, "G/CC")
    assert "not coerced" in mismatch.value.reason
    inverse = load_unit_spellings().conversion("KG/M3", "G/CC", "density")
    assert inverse is not None
    restored = float(Fraction(float(view.values[0])) * inverse.factor + inverse.offset)
    assert restored == pytest.approx(2.5)
    assert tuple(curve.values.tobytes() for curve in ingested.dataset.curves) == before
    same = read_values(ingested.dataset, rhob, "g/cm3")
    assert same.unit == "G/CC"
    assert same.conversion is None
    require_unit(same, "G/CC")


def test_depth_converted_to_metres_invalidates_a_step_derived_in_feet(tmp_path: Path):
    ingested, _identified, result = _run(
        tmp_path,
        [("GR", "GAPI")],
        [["0.0", "10.0"], ["0.5", "11.0"]],
        depth_unit="FT",
    )
    depth = _curve(result, "DEPT")
    assert depth.outcome is UnitInterpretation.RECOGNISED
    assert depth.canonical_unit == "FT"
    step = DerivedQuantity("observed_step", "0.5", "FT")
    still_feet = declare_depth_conversion(step, "FT")
    assert still_feet.invalidated is False
    view = read_values(ingested.dataset, depth, "M")
    assert view.unit == "M"
    assert view.conversion is not None
    assert view.conversion.factor == Fraction(381, 1250)
    declaration = declare_depth_conversion(step, view.unit)
    assert declaration.invalidated is True
    assert "feet" in declaration.reason
    assert_derived_unit(step, "FT")
    with pytest.raises(AssertionError):
        assert_derived_unit(step, "M")
    assert ingested.dataset.curves[0].values.flags.writeable is False


def test_spellings_have_no_case_only_conflict_and_pu_has_no_conversion():
    spellings = load_unit_spellings()
    assert spellings.case_conflicts == ()
    assert spellings.meanings("gapi")[0].canonical_unit == "GAPI"
    pu = spellings.meanings("PU")
    assert len(pu) == 1
    assert pu[0].dimension == "porosity"
    assert spellings.conversion("PU", "V/V", "porosity") is None
    assert spellings.conversion("S/M", "OHMM", "electrical_resistivity") is None


def test_observed_fixture_spellings_resolve(tmp_path: Path):
    """Each spelling added from a fixture curve section. P11b item 6."""

    spellings = load_unit_spellings()
    expected = {
        "m3/m3": ("V/V", "porosity", ("30_3-5 S.las", "34_10-19.las")),
        "inches": ("IN", "length", ("WLC_PETRO_COMPUTED_INPUT_1_15_9-F-1B.las", "159-19A_LFP.las")),
        "b/elec": ("B/E", "photoelectric_factor", ("WLC_PETRO_COMPUTED_INPUT_1_15_9-F-1B.las",)),
        "v/v_decimal": ("V/V", "porosity", ("159-19A_LFP.las",)),
    }
    for text, (canonical, dimension, names) in expected.items():
        matches = spellings.meanings(text)
        assert len(matches) == 1
        assert (matches[0].canonical_unit, matches[0].dimension) == (canonical, dimension)
        assert matches[0].source_tier == "OBSERVED"
        for name in names:
            assert f"tests/fixtures/{name}, ~Curve section" in matches[0].source
    _ingested, _identified, result = _run(
        tmp_path,
        [("NPHI", "m3/m3"), ("CALI", "inches"), ("PEF", "b/elec"), ("NPOR", "v/v_decimal")],
        [["0.0", "0.2", "8.5", "3.0", "0.3"], ["1.0", "0.21", "8.6", "3.1", "0.31"]],
    )
    for mnemonic, (canonical, dimension) in {
        "NPHI": ("V/V", "porosity"),
        "CALI": ("IN", "length"),
        "PEF": ("B/E", "photoelectric_factor"),
        "NPOR": ("V/V", "porosity"),
    }.items():
        item = _curve(result, mnemonic)
        assert item.outcome is UnitInterpretation.RECOGNISED
        assert (item.canonical_unit, item.dimension) == (canonical, dimension)


def test_f_resolves_to_feet_or_fahrenheit_by_family(tmp_path: Path):
    """F is feet on a depth family and Fahrenheit on temperature. P11b item 7, B3."""

    spellings = load_unit_spellings()
    assert spellings.case_conflicts == ()
    meanings = {(row.canonical_unit, row.dimension) for row in spellings.meanings("f")}
    assert meanings == {("DEGF", "temperature"), ("FT", "length")}
    _ingested, _identified, result = _run(
        tmp_path,
        [("DEPTH", "F"), ("TEMP", "F"), ("ZZTOP", "F")],
        [["0.0", "100", "80", "1"], ["1.0", "101", "81", "2"]],
    )
    depth = _curve(result, "DEPTH")
    assert depth.outcome is UnitInterpretation.RECOGNISED
    assert (depth.canonical_unit, depth.dimension) == ("FT", "length")
    temperature = _curve(result, "TEMP")
    assert temperature.outcome is UnitInterpretation.RECOGNISED
    assert (temperature.canonical_unit, temperature.dimension) == ("DEGF", "temperature")
    unknown = _curve(result, "ZZTOP")
    assert unknown.outcome is UnitInterpretation.UNRECOGNISED
    findings = _findings(result, BlockType.AMBIGUOUS_INTERPRETATION, "ZZTOP")
    assert len(findings) == 1
    candidates = json.loads(findings[0].evidence.as_dict()["candidates"])
    got = {(row["canonical_unit"], row["dimension"]) for row in candidates}
    assert got == {("DEGF", "temperature"), ("FT", "length")}


def test_family_loader_rejects_a_unit_dimension_and_a_bad_list():
    base = {
        "id": "X",
        "display_name": "X",
        "tool_group": None,
        "interpolation_domain": None,
        "display_log_scale": None,
        "legacy_group": "x",
        "source": "s",
        "source_tier": "CITED",
        "accepted_dimensions_note": None,
    }
    with pytest.raises(ValueError, match="unit_dimension"):
        _family_record({**base, "unit_dimension": "density", "accepted_dimensions": ["density"]}, set())
    with pytest.raises(ValueError, match="non-empty"):
        _family_record({**base, "accepted_dimensions": []}, set())
    with pytest.raises(ValueError, match="duplicate"):
        _family_record({**base, "accepted_dimensions": ["density", "density"]}, set())
    with pytest.raises(ValueError, match="non-empty strings"):
        _family_record({**base, "accepted_dimensions": ["density", 1]}, set())


def test_runner_leaves_source_arrays_unchanged(tmp_path: Path):
    ingested, identified, _result = _run(tmp_path, [("GR", "GAPI")], [["0.0", "1.0"], ["1.0", "2.0"]])
    before = tuple(curve.values.tobytes() for curve in ingested.dataset.curves)

    def stage0() -> RegisteredStage:
        contract = StageContract(
            0,
            "Ingest",
            StageRequires("Units stay as written.", "The file has been read.", ()),
            StageGuarantees("Unchanged.", "Unchanged.", "Unchanged."),
            ("The file cannot be read.",),
            (),
            False,
        )

        def run(_context: object) -> StageOutcome:
            return StageOutcome(StageStatus.COMPLETE, ())

        def unmet(_context: object) -> tuple[str, ...]:
            return ()

        return RegisteredStage(contract, run, unmet)

    depth = classify(ingested)
    runner = Runner(
        (
            stage0(),
            depth_stage(ingested),
            identify_stage(ingested, depth, tmp_path / "no-local-tables"),
            registered_stage(ingested, identified),
        )
    )
    results = runner.run(ingested.dataset)
    assert [item.status for item in results] == [StageStatus.COMPLETE] * 4
    assert tuple(curve.values.tobytes() for curve in ingested.dataset.curves) == before
