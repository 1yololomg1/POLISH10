"""Stage 1 depth axis. D3, D8, D9, D28, F10, D67, D86, D88, D89, C4.

Synthetic files are the cases. The four LAS fixtures are the measured ones.
Counts of curves, steps, and findings are taken from the result, not typed in
as a stand-in for a measurement the file can supply.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from polish.decisions.apply import apply_decision, depth_unit_choices, listed_options
from polish.decisions.session import open_session
from polish.model.dataset import DepthDirection, DepthRegularity
from polish.model.depth_unit_choice import DepthUnitChoice
from polish.model.states import BlockType, StageStatus, UserAction
from polish.stages.runner import RegisteredStage, Runner, StageOutcome
from polish.stages.s00_ingest import CONTRACT as INGEST_CONTRACT
from polish.stages.s00_ingest import ingest
from polish.stages.s01_depth import (
    CONTRACT,
    DerivedQuantity,
    assert_derived_unit,
    classify,
    depth_index_mnemonics,
    registered_stage,
)

_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures"
_WHEN = datetime.datetime(2026, 9, 26, tzinfo=datetime.timezone.utc)


def _write(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "well.las"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _las(
    *,
    wrap: str = "NO",
    strt: str = "0.0",
    stop: str = "1.0",
    step: str = "1.0",
    well_unit: str = "M",
    curves: str = "DEPT.M : DEPTH\nGR.GAPI : GAMMA\n",
    data: str = "0.0 1.0\n1.0 2.0\n",
    step_line: str | None = None,
    strt_line: str | None = None,
    stop_line: str | None = None,
) -> str:
    if step_line is None:
        step_line = f"STEP.{well_unit} {step} : STEP\n"
    if strt_line is None:
        strt_line = f"STRT.{well_unit} {strt} : START\n"
    if stop_line is None:
        stop_line = f"STOP.{well_unit} {stop} : STOP\n"
    return (
        "~VERSION INFORMATION\n"
        "VERS. 2.0 : CWLS\n"
        f"WRAP. {wrap} : WRAP\n"
        "~WELL INFORMATION\n"
        f"{strt_line}"
        f"{stop_line}"
        f"{step_line}"
        "NULL. -999.25 : NULL\n"
        "~CURVE INFORMATION\n"
        f"{curves}"
        "~A\n"
        f"{data}"
    )


def _classify(directory: Path, body: str):
    return classify(ingest(_write(directory, body), read_at=_WHEN))


def _messages(result, block: BlockType) -> list[str]:
    return [finding.message for finding in result.findings if finding.block_type is block]


def test_contract_is_read_only_and_depends_on_stage_0():
    assert CONTRACT.stage_number == 1
    assert CONTRACT.alters_data is False
    assert CONTRACT.depends_on_stages == (0,)
    assert depth_index_mnemonics() == frozenset({"DEPT", "DEPTH", "MD", "TVD", "TVDSS"})


def test_ascending_well_is_regular_and_records_the_alias(tmp_path: Path):
    ingested = ingest(_write(tmp_path, _las()), read_at=_WHEN)
    assert ingested.dataset is not None
    before = ingested.dataset.grid.values.tobytes()
    second_before = ingested.dataset.curves[1].values.tobytes()
    result = classify(ingested)
    assert result.status is StageStatus.COMPLETE
    assert result.direction is DepthDirection.ASCENDING
    assert result.regularity is DepthRegularity.REGULAR
    assert result.alias is not None
    assert result.alias.canonical_name == "DEPT"
    assert result.alias.source.column_index == 0
    assert result.corrected_step is None
    assert ingested.dataset.grid.values.tobytes() == before
    assert ingested.dataset.curves[1].values.tobytes() == second_before
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)


def test_bottom_up_with_a_blank_depth_unit_omits_the_unit(tmp_path: Path):
    body = _las(
        curves="DEPT. : DEPTH\nGR.GAPI : GAMMA\n",
        strt="1.0",
        stop="0.0",
        step="-1.0",
        data="1.0 1.0\n0.0 2.0\n",
    )
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    found = [item for item in result.findings if item.check == "depth_bottom_up"]
    assert len(found) == 1
    assert found[0].message == (
        "This log is recorded bottom-up: depth decreases from 1.0 to 0.0. "
        "That is valid LAS, but POLISH supports increasing depth only, so the file was not processed. "
        "The file is unchanged."
    )


def test_descending_well_with_negative_step_is_structural(tmp_path: Path):
    body = _las(strt="1.0", stop="0.0", step="-1.0", data="1.0 1.0\n0.0 2.0\n")
    path = _write(tmp_path, body)
    ingested = ingest(path, read_at=_WHEN)
    assert ingested.dataset is not None
    assert ingested.dataset.grid.direction is DepthDirection.DESCENDING
    before = ingested.dataset.grid.values.tobytes()
    result = classify(ingested)
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert result.direction is not DepthDirection.DESCENDING
    assert result.direction is None
    assert ingested.dataset.grid.values.tobytes() == before
    structural = [item for item in result.findings if item.block_type is BlockType.STRUCTURAL_DATA_FAILURE]
    assert len(structural) == 1
    assert structural[0].check == "depth_bottom_up"
    assert "valid LAS" in structural[0].message
    assert "increasing depth only" in structural[0].message
    assert structural[0].message == (
        "This log is recorded bottom-up: depth decreases from 1.0 to 0.0 M. "
        "That is valid LAS, but POLISH supports increasing depth only, so the file was not processed. "
        "The file is unchanged."
    )
    from polish.pipeline.run import run_well

    tables = tmp_path / "tables"
    app = tmp_path / "app"
    tables.mkdir()
    app.mkdir()
    ran = run_well(path, read_at=_WHEN, local_tables_directory=tables, app_data=app)
    assert ran.depth is not None
    assert ran.depth.status is StageStatus.REFUSED_STRUCTURAL
    assert ran.identification is None
    assert ran.units is None
    assert ran.validity is None
    assert ingested.dataset.grid.values.tobytes() == before


def test_irregular_ascending_grid_is_accepted(tmp_path: Path):
    body = _las(strt="0.0", stop="3.0", step="1.0", data="0.0 1.0\n1.0 2.0\n3.0 3.0\n")
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.COMPLETE
    assert result.direction is DepthDirection.ASCENDING
    assert result.regularity is DepthRegularity.IRREGULAR
    assert result.corrected_step is None
    assert any("irregular" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)


def test_duplicate_depth_is_structural(tmp_path: Path):
    body = _las(strt="0.0", stop="1.0", step="1.0", data="0.0 1.0\n1.0 2.0\n1.0 3.0\n")
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert any("Duplicate depth" in message for message in _messages(result, BlockType.STRUCTURAL_DATA_FAILURE))


def test_non_monotonic_depth_is_structural(tmp_path: Path):
    body = _las(strt="0.0", stop="1.0", step="1.0", data="0.0 1.0\n2.0 2.0\n1.0 3.0\n")
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert any("decreases" in message for message in _messages(result, BlockType.STRUCTURAL_DATA_FAILURE))


def test_first_curve_that_is_not_depth_halts(tmp_path: Path):
    body = _las(
        curves="GR.GAPI : GAMMA\nDEPT.M : DEPTH\n",
        data="0.0 10.0\n1.0 11.0\n",
    )
    ingested = ingest(_write(tmp_path, body), read_at=_WHEN)
    assert ingested.dataset is not None
    before = tuple(curve.original_mnemonic for curve in ingested.dataset.curves)
    result = classify(ingested)
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert result.alias is None
    assert result.direction is None
    assert any("not a depth index" in message for message in _messages(result, BlockType.STRUCTURAL_DATA_FAILURE))
    assert tuple(curve.original_mnemonic for curve in ingested.dataset.curves) == before


def test_stop_within_one_step_is_informational(tmp_path: Path):
    body = _las(strt="0.0", stop="1.5", step="1.0", data="0.0 1.0\n1.0 2.0\n")
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.COMPLETE
    assert result.recorded_stop is not None
    assert result.recorded_stop.text == "1.0"
    informational = _messages(result, BlockType.INFORMATIONAL)
    assert any("STOP" in message and "step intervals" in message for message in informational)
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)


def test_stop_beyond_one_step_is_structural(tmp_path: Path):
    body = _las(strt="0.0", stop="3.0", step="1.0", data="0.0 1.0\n1.0 2.0\n")
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert any("beyond one step" in message for message in _messages(result, BlockType.STRUCTURAL_DATA_FAILURE))


def test_blank_step_with_regular_increments_records_the_data_step(tmp_path: Path):
    body = _las(
        strt="0.0",
        stop="1.0",
        step_line="STEP.M  : STEP\n",
        data="0.0 1.0\n0.5 2.0\n1.0 3.0\n",
    )
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert result.declared_step is None
    assert result.observed_step is not None
    assert result.observed_step.text == "0.5"
    assert result.observed_step.unit == "M"
    assert_derived_unit(result.observed_step, "M")
    assert any("Data-derived step" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)


def test_blank_step_with_irregular_increments_is_accepted(tmp_path: Path):
    body = _las(
        strt="0.0",
        stop="3.0",
        step_line="STEP.M  : STEP\n",
        data="0.0 1.0\n1.0 2.0\n3.0 3.0\n",
    )
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.IRREGULAR
    assert result.observed_step is None
    assert any("not constant" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)


def test_missing_depth_unit_is_ambiguous(tmp_path: Path):
    body = _las(curves="DEPT. : DEPTH\nGR.GAPI : GAMMA\n")
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.COMPLETE
    assert result.primary_standard_step is None
    assert result.corrected_step is None
    ambiguous = _messages(result, BlockType.AMBIGUOUS_INTERPRETATION)
    assert any("not recognised" in message and "not assumed" in message for message in ambiguous)
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)


def test_two_depth_curves_keep_the_second_unchanged(tmp_path: Path):
    body = _las(
        curves="DEPT.M : DEPTH\nDEPTH_MD.M : MEASURED\n",
        data="0.0 5.0\n1.0 6.0\n",
    )
    ingested = ingest(_write(tmp_path, body), read_at=_WHEN)
    assert ingested.dataset is not None
    second = ingested.dataset.curves[1]
    values = second.values.tobytes()
    mnemonic = second.original_mnemonic
    result = classify(ingested)
    assert result.status is StageStatus.COMPLETE
    assert result.alias is not None
    assert result.alias.source.column_index == 0
    assert ingested.dataset.curves[1].original_mnemonic == mnemonic == "DEPTH_MD"
    assert ingested.dataset.curves[1].values.tobytes() == values
    assert not any("DEPTH_MD" in message for message in _messages(result, BlockType.AMBIGUOUS_INTERPRETATION))


def test_non_standard_step_failing_the_d89_gate_is_not_corrected(tmp_path: Path):
    rows = [f"{index * DecimalText('0.30000000')} 1.0" for index in range(40)]
    body = _las(
        strt="0.00000000",
        stop=rows[-1].split()[0],
        step="0.30000000",
        data="\n".join(rows) + "\n",
    )
    ingested = ingest(_write(tmp_path, body), read_at=_WHEN)
    assert ingested.dataset is not None
    before = ingested.dataset.grid.values.tobytes()
    result = classify(ingested)
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert result.corrected_step is None
    assert result.direction is DepthDirection.ASCENDING
    assert ingested.dataset.grid.values.tobytes() == before
    assert any(
        "not a recognised standard interval" in message and "not modified" in message
        for message in _messages(result, BlockType.INFORMATIONAL)
    )


def test_non_standard_step_passing_the_d89_gate_is_corrected_on_the_record(tmp_path: Path):
    body = _las(strt="0.00", stop="0.30", step="0.15", data="0.00 1.0\n0.15 2.0\n0.30 3.0\n")
    ingested = ingest(_write(tmp_path, body), read_at=_WHEN)
    assert ingested.dataset is not None
    before = ingested.dataset.grid.values.tobytes()
    result = classify(ingested)
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert result.corrected_step is not None
    assert result.corrected_step.text == "0.1524"
    assert result.displacement is not None
    assert_derived_unit(result.corrected_step, result.corrected_step.unit)
    informational = _messages(result, BlockType.INFORMATIONAL)
    assert any("corrected to" in message and "displacement" in message for message in informational)
    assert ingested.dataset.grid.values.tobytes() == before


def test_derived_quantity_refuses_a_different_unit():
    quantity = DerivedQuantity("observed_step", "0.5", "FT")
    assert assert_derived_unit(quantity, "FT") == quantity.value()
    with pytest.raises(AssertionError, match="C4"):
        assert_derived_unit(quantity, "M")


def test_runner_does_not_reorder_or_replace_the_dataset(tmp_path: Path):
    ingested = ingest(_write(tmp_path, _las()), read_at=_WHEN)
    assert ingested.dataset is not None
    before = ingested.dataset.grid.values.tobytes()
    identity = id(ingested.dataset)

    def run(_context):
        return StageOutcome(StageStatus.COMPLETE, ())

    def unmet(_context):
        return ()

    stage0 = RegisteredStage(INGEST_CONTRACT, run, unmet)
    results = Runner((stage0, registered_stage(ingested))).run(ingested.dataset)
    assert results[1].status is StageStatus.COMPLETE
    assert id(ingested.dataset) == identity
    assert ingested.dataset.grid.values.tobytes() == before


def _fixture(name: str):
    return classify(ingest(_FIXTURES / name, read_at=_WHEN))


def _kind_counts(result) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in result.findings:
        counts[finding.block_type.value] = counts.get(finding.block_type.value, 0) + 1
    return counts


def _data_interval(name: str) -> str:
    ingested = ingest(_FIXTURES / name, read_at=_WHEN)
    tokens = ingested.audit.depth_tokens
    interval = (Decimal(tokens[-1].strip()) - Decimal(tokens[0].strip())) / Decimal(len(tokens) - 1)
    return format(interval, "f")


def test_fixture_34_10_19_is_regular_at_the_written_step():
    result = _fixture("34_10-19.las")
    assert result.status is StageStatus.COMPLETE
    assert result.direction is DepthDirection.ASCENDING
    assert result.regularity is DepthRegularity.REGULAR
    assert result.corrected_step is None
    assert result.observed_step is not None
    assert result.observed_step.text == "0.15200000"
    notice = next(item for item in result.findings if item.check == "step_not_standard")
    assert notice.evidence.as_dict()["compared_quantity"] == "implied_interval"
    assert notice.evidence.as_dict()["compared_value"] == _data_interval("34_10-19.las")
    assert notice.evidence.as_dict()["observed"] == "0.1524"
    assert "first and last depths in the data" in notice.message
    assert any("STOP" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert _kind_counts(result) == {"INFORMATIONAL": 2}


def test_fixture_30_3_5_s_matches_the_34_pattern():
    result = _fixture("30_3-5 S.las")
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert result.corrected_step is None
    assert result.observed_step is not None
    assert result.observed_step.text == "0.15200000"
    notice = next(item for item in result.findings if item.check == "step_not_standard")
    assert notice.evidence.as_dict()["compared_quantity"] == "implied_interval"
    assert notice.evidence.as_dict()["compared_value"] == _data_interval("30_3-5 S.las")
    assert notice.evidence.as_dict()["observed"] == "0.1524"
    assert "first and last depths in the data" in notice.message
    assert any("STOP" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert _kind_counts(result) == {"INFORMATIONAL": 2}


def test_fixture_159_is_regular_at_0_1524_after_the_wrapped_reread():
    result = _fixture("159-19A_LFP.las")
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert result.observed_step is not None
    assert result.observed_step.text == "0.1524"
    assert result.corrected_step is None
    assert result.findings == ()
    assert _kind_counts(result) == {}


def test_fixture_messages_keep_at_most_twelve_decimal_places():
    names = (
        "159-19A_LFP.las",
        "30_3-5 S.las",
        "34_10-19.las",
        "WLC_PETRO_COMPUTED_INPUT_1_15_9-F-1B.las",
        "new_file_null_minus_999.las",
    )
    for name in names:
        result = _fixture(name)
        for finding in result.findings:
            fractions = re.findall(r"\d+\.(\d+)", finding.message)
            assert all(len(fraction) <= 12 for fraction in fractions), (name, finding.message)


def test_fixture_wlc_is_regular_at_the_standard_tenth_of_a_metre():
    result = _fixture("WLC_PETRO_COMPUTED_INPUT_1_15_9-F-1B.las")
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert result.corrected_step is None
    assert result.observed_step is not None
    assert result.observed_step.text == "0.1000"
    assert result.primary_standard_step is not None
    assert result.primary_standard_step.text == "0.1524"
    assert not any("not a recognised standard interval" in finding.message for finding in result.findings)
    assert _kind_counts(result) == {}


def _checks(result) -> list[str]:
    return [item.check for item in result.findings]


def _series(start: str, step: str, count: int, skip: set[int] | None = None) -> str:
    """Depth rows from ``start``, stepping by ``step``, omitting indexes in ``skip``."""

    origin = Decimal(start)
    increment = Decimal(step)
    omitted = skip or set()
    rows = [format(origin + (increment * index), "f") for index in range(count) if index not in omitted]
    return "\n".join(f"{depth} 1.0" for depth in rows) + "\n"


def test_one_missing_sample_on_a_tenth_grid_is_irregular(tmp_path: Path):
    data = _series("100.0", "0.1", 12, {5})
    declared = _las(strt="100.0", stop="101.1", step="0.1", data=data)
    result = _classify(tmp_path, declared)
    assert result.regularity is DepthRegularity.IRREGULAR
    assert "grid_off_declared_step" in _checks(result)
    assert result.observed_step is None


def test_one_missing_sample_with_a_blank_step_records_the_median_and_validates_none(tmp_path: Path):
    data = _series("100.0", "0.1", 12, {5})
    body = _las(strt="100.0", stop="101.1", step_line="STEP.M  : STEP\n", data=data)
    result = _classify(tmp_path, body)
    assert result.regularity is DepthRegularity.IRREGULAR
    irregular = [item for item in result.findings if item.check == "step_blank_irregular"]
    assert len(irregular) == 1
    assert irregular[0].evidence.as_dict()["observed"] == "0.1"
    assert result.observed_step is None


def test_one_missing_integer_sample_is_irregular_for_both_step_paths(tmp_path: Path):
    data = _series("100", "1", 11, {5})
    declared = _las(strt="100", stop="110", step="1", data=data)
    declared_result = _classify(tmp_path, declared)
    assert declared_result.regularity is DepthRegularity.IRREGULAR
    assert declared_result.observed_step is None
    blank = _las(strt="100", stop="110", step_line="STEP.M  : STEP\n", data=data)
    blank_result = _classify(tmp_path / "blank", blank)
    assert blank_result.regularity is DepthRegularity.IRREGULAR
    assert blank_result.observed_step is None
    found = [item for item in blank_result.findings if item.check == "step_blank_irregular"]
    assert found[0].evidence.as_dict()["observed"] == "1"


def test_a_complete_tenth_grid_is_regular_with_and_without_step(tmp_path: Path):
    data = _series("100.0", "0.1", 12)
    declared = _classify(tmp_path, _las(strt="100.0", stop="101.1", step="0.1", data=data))
    assert declared.regularity is DepthRegularity.REGULAR
    assert "grid_off_declared_step" not in _checks(declared)
    blank = _classify(
        tmp_path / "blank",
        _las(strt="100.0", stop="101.1", step_line="STEP.M  : STEP\n", data=data),
    )
    assert blank.regularity is DepthRegularity.REGULAR
    assert blank.observed_step is not None
    assert blank.observed_step.text == "0.1"


def test_a_rounded_half_foot_grid_is_regular_until_a_sample_is_missing(tmp_path: Path):
    step = Decimal("0.1524")
    count = 8
    complete = [format((step * index).quantize(Decimal("0.001")), "f") for index in range(count)]
    body = _las(
        strt=complete[0],
        stop=complete[-1],
        step="0.1524",
        data="\n".join(f"{depth} 1.0" for depth in complete) + "\n",
    )
    regular = _classify(tmp_path, body)
    assert regular.regularity is DepthRegularity.REGULAR
    missing = [depth for index, depth in enumerate(complete) if index != 3]
    broken = _las(
        strt=missing[0],
        stop=missing[-1],
        step="0.1524",
        data="\n".join(f"{depth} 1.0" for depth in missing) + "\n",
    )
    irregular = _classify(tmp_path / "missing", broken)
    assert irregular.regularity is DepthRegularity.IRREGULAR
    assert irregular.observed_step is None


def test_disturbing_both_ends_of_a_blank_step_is_irregular(tmp_path: Path):
    data = _series("100.0", "0.1", 13, {1, 12})
    result = _classify(tmp_path, _las(strt="100.0", stop="101.1", step_line="STEP.M  : STEP\n", data=data))
    assert result.regularity is DepthRegularity.IRREGULAR
    assert result.observed_step is None


def test_strt_off_the_first_depth_does_not_make_a_regular_grid_irregular(tmp_path: Path):
    data = _series("100.0", "0.1", 12)
    result = _classify(tmp_path, _las(strt="100.06", stop="101.1", step="0.1", data=data))
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert "grid_off_declared_step" not in _checks(result)
    strt = [item for item in result.findings if item.check == "endpoint_within_step" and item.identity == (("endpoint", "STRT"),)]
    assert len(strt) == 1


def test_strt_within_one_step_is_informational(tmp_path: Path):
    result = _classify(tmp_path, _las(strt="0.4", stop="1.0", step="1.0"))
    assert result.status is StageStatus.COMPLETE
    assert any(item.check == "endpoint_within_step" and "STRT" in item.message for item in result.findings)
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)


def test_strt_beyond_one_step_is_structural(tmp_path: Path):
    result = _classify(tmp_path, _las(strt="2.5", stop="1.0", step="1.0"))
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert any(item.check == "endpoint_beyond_step" and "STRT" in item.message for item in result.findings)


def test_stop_differing_by_decimal_rounding_is_informational(tmp_path: Path):
    result = _classify(
        tmp_path,
        _las(strt="0.000", stop="1.001", step="0.500", data="0.000 1.0\n0.500 2.0\n1.000 3.0\n"),
    )
    assert result.status is StageStatus.COMPLETE
    stop = [item for item in result.findings if item.check == "endpoint_within_step" and "STOP" in item.message]
    assert len(stop) == 1
    assert stop[0].evidence.as_dict()["difference"] == "0.001"
    assert not any(item.check == "endpoint_beyond_step" for item in result.findings)


def test_feet_written_as_f_refuses_a_stop_beyond_one_step(tmp_path: Path):
    data = "100.0 1.0\n100.5 2.0\n"
    feet = _las(
        well_unit="FT",
        curves="DEPT.FT : DEPTH\nGR.GAPI : GAMMA\n",
        strt="100.0",
        stop="500.0",
        step="0.5",
        data=data,
    )
    short = _las(
        well_unit="F",
        curves="DEPT.F : DEPTH\nGR.GAPI : GAMMA\n",
        strt="100.0",
        stop="500.0",
        step="0.5",
        data=data,
    )
    feet_result = _classify(tmp_path, feet)
    short_result = _classify(tmp_path / "short", short)
    assert feet_result.status is StageStatus.REFUSED_STRUCTURAL
    assert short_result.status is StageStatus.REFUSED_STRUCTURAL
    assert any(item.check == "endpoint_beyond_step" for item in feet_result.findings)
    assert any(item.check == "endpoint_beyond_step" for item in short_result.findings)


def test_dept_f_with_a_matching_stop_is_complete(tmp_path: Path):
    result = _classify(
        tmp_path,
        _las(
            well_unit="F",
            curves="DEPT.F : DEPTH\nGR.GAPI : GAMMA\n",
            strt="100.0",
            stop="100.5",
            step="0.5",
            data="100.0 1.0\n100.5 2.0\n",
        ),
    )
    assert result.status is StageStatus.COMPLETE
    assert result.depth_unit == "F"
    assert "depth_unit_unrecognised" not in _checks(result)
    assert "endpoint_unit_mismatch" not in _checks(result)


def test_a_recorded_feet_choice_runs_the_standard_step_and_the_endpoints(tmp_path: Path):
    body = _las(
        well_unit="FT",
        curves="DEPT. : DEPTH\nGR.GAPI : GAMMA\n",
        strt="0.0",
        stop="1.4",
        step="0.5",
        data="0.0 1.0\n0.5 2.0\n1.0 3.0\n",
    )
    path = _write(tmp_path, body)
    ingested = ingest(path, read_at=_WHEN)
    asked = classify(ingested)
    question = next(item for item in asked.findings if item.check == "depth_unit_unrecognised")
    assert listed_options(question) == ("FT", "M")
    session = open_session(path, asked.findings)
    saved = apply_decision(
        session,
        question,
        UserAction.CHOOSE_OPTION,
        reason="the depth is in feet",
        chosen_option="FT",
    )
    choices = depth_unit_choices(saved)
    assert len(choices) == 1
    result = classify(ingested, depth_unit_choices=choices)
    assert result.status is StageStatus.COMPLETE
    assert result.depth_unit == "FT"
    assert result.depth_unit_decision_id == choices[0].decision_id
    assert result.primary_standard_step is not None
    assert result.primary_standard_step.text == "0.5"
    assert result.primary_standard_step.unit == "FT"
    assert "depth_unit_unrecognised" not in _checks(result)
    assert any(item.check == "endpoint_within_step" and "STOP" in item.message for item in result.findings)
    assert "endpoint_unit_mismatch" not in _checks(result)
    assert "step_not_standard" not in _checks(result)
    assert "step_corrected" not in _checks(result)


def test_a_depth_unit_choice_outside_the_listed_options_is_refused(tmp_path: Path):
    path = _write(tmp_path, _las(curves="DEPT. : DEPTH\nGR.GAPI : GAMMA\n"))
    asked = classify(ingest(path, read_at=_WHEN))
    question = next(item for item in asked.findings if item.check == "depth_unit_unrecognised")
    session = open_session(path, asked.findings)
    saved = apply_decision(
        session,
        question,
        UserAction.CHOOSE_OPTION,
        reason="feet",
        chosen_option="FT",
    )
    stale = replace(saved.decisions[0], chosen_option="FEET")
    with pytest.raises(ValueError, match="not an option this finding lists"):
        depth_unit_choices(replace(saved, decisions=(stale,)))
    with pytest.raises(ValueError, match="not an option this finding lists"):
        classify(ingest(path, read_at=_WHEN), depth_unit_choices=(DepthUnitChoice("FEET", "decision-1"),))


def test_a_held_session_applies_no_depth_unit_choice(tmp_path: Path):
    path = _write(tmp_path, _las(curves="DEPT. : DEPTH\nGR.GAPI : GAMMA\n"))
    ingested = ingest(path, read_at=_WHEN)
    asked = classify(ingested)
    question = next(item for item in asked.findings if item.check == "depth_unit_unrecognised")
    session = open_session(path, asked.findings)
    saved = apply_decision(
        session,
        question,
        UserAction.CHOOSE_OPTION,
        reason="feet",
        chosen_option="FT",
    )
    held = replace(saved, hold=True)
    assert depth_unit_choices(held) == ()
    result = classify(ingested, depth_unit_choices=depth_unit_choices(held))
    assert result.depth_unit_decision_id is None
    assert result.primary_standard_step is None
    assert "depth_unit_unrecognised" in _checks(result)


def test_the_standard_step_follows_the_implied_interval(tmp_path: Path):
    depths = [format(Decimal("0.152") * index, ".3f") for index in range(8)]
    result = _classify(
        tmp_path,
        _las(
            strt=depths[0],
            stop=depths[-1],
            step="0.1524",
            data="\n".join(f"{depth} 1.0" for depth in depths) + "\n",
        ),
    )
    assert result.status is StageStatus.COMPLETE
    found = [item for item in result.findings if item.check == "step_not_standard"]
    assert len(found) == 1
    evidence = found[0].evidence.as_dict()
    assert evidence["compared_quantity"] == "implied_interval"
    assert evidence["compared_value"] == "0.152"
    assert evidence["observed"] == "0.1524"
    assert "first and last depths in the data" in found[0].message
    assert "step_corrected" not in _checks(result)
    assert result.corrected_step is None


def test_a_short_stop_on_a_standard_grid_is_only_an_endpoint_finding(tmp_path: Path):
    result = _classify(
        tmp_path,
        _las(strt="0.0", stop="1.04", step="1.0", data="0.0 1.0\n1.0 2.0\n2.0 3.0\n"),
    )
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert "step_not_standard" not in _checks(result)
    assert "step_corrected" not in _checks(result)
    assert result.corrected_step is None
    stop = [item for item in result.findings if item.check == "endpoint_within_step" and "STOP" in item.message]
    assert len(stop) == 1
    assert stop[0].evidence.as_dict()["difference"] == "-0.96"


def test_exponential_depths_are_not_called_irregular(tmp_path: Path):
    rows = [f"{format(Decimal('100') + (Decimal(index) * Decimal('0.5')), '.4E')} 1.0" for index in range(3)]
    result = _classify(tmp_path, _las(strt="100.0", stop="101.0", step="0.5", data="\n".join(rows) + "\n"))
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is None
    assert _checks(result) == ["depth_exponential"]
    assert "grid_off_declared_step" not in _checks(result)


def test_a_strt_value_with_a_colon_is_not_called_blank(tmp_path: Path):
    data = _series("100.0", "0.1", 12)
    body = _las(
        strt_line="STRT.M 100.0 : START DEPTH: KB\n",
        stop="101.1",
        step="0.1",
        data=data,
    )
    result = _classify(tmp_path, body)
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    found = [item for item in result.findings if item.check == "strt_not_a_number"]
    assert len(found) == 1
    assert found[0].message == "STRT is not a number: '100.0 : START DEPTH'."
    assert "strt_blank" not in _checks(result)


def test_a_decrease_that_is_not_bottom_up_keeps_its_message(tmp_path: Path):
    result = _classify(
        tmp_path,
        _las(strt="100.0", stop="100.7", step="0.5", data="100.0 1.0\n100.5 2.0\n100.2 3.0\n100.7 4.0\n"),
    )
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    found = [item for item in result.findings if item.check == "depth_decreases"]
    assert len(found) == 1
    assert found[0].message == "Depth decreases from 100.5 to 100.2 at indexes 1 and 2."
    assert "depth_bottom_up" not in _checks(result)


def test_a_negative_step_with_rising_depths_names_the_contradiction(tmp_path: Path):
    result = _classify(
        tmp_path,
        _las(strt="100.0", stop="101.0", step="-0.5", data="100.0 1.0\n100.5 2.0\n101.0 3.0\n"),
    )
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    found = [item for item in result.findings if item.check == "step_negative"]
    assert len(found) == 1
    assert found[0].message == (
        "STEP -0.5 is negative, but the depths increase. "
        "The header contradicts the data, so the file was not processed."
    )
    assert "depth_bottom_up" not in _checks(result)


class DecimalText:
    """Build an exact decimal spelling without binary float noise."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __rmul__(self, index: int) -> str:
        from decimal import Decimal

        return format(Decimal(index) * Decimal(self.text), "f")
