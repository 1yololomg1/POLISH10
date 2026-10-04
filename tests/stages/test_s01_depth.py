"""Stage 1 depth axis. D3, D8, D9, D28, F10, D67, D86, D88, D89, C4.

Synthetic files are the cases. The four LAS fixtures are the measured ones.
Counts of curves, steps, and findings are taken from the result, not typed in
as a stand-in for a measurement the file can supply.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from polish.model.dataset import DepthDirection, DepthRegularity
from polish.model.states import BlockType, StageStatus
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
    path = directory / "well.las"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _las(
    *,
    wrap: str = "NO",
    strt: str = "0.0",
    stop: str = "1.0",
    step: str = "1.0",
    curves: str = "DEPT.M : DEPTH\nGR.GAPI : GAMMA\n",
    data: str = "0.0 1.0\n1.0 2.0\n",
    step_line: str | None = None,
) -> str:
    if step_line is None:
        step_line = f"STEP.M {step} : STEP\n"
    return (
        "~VERSION INFORMATION\n"
        "VERS. 2.0 : CWLS\n"
        f"WRAP. {wrap} : WRAP\n"
        "~WELL INFORMATION\n"
        f"STRT.M {strt} : START\n"
        f"STOP.M {stop} : STOP\n"
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


def test_descending_well_with_negative_step_is_structural(tmp_path: Path):
    body = _las(strt="1.0", stop="0.0", step="-1.0", data="1.0 1.0\n0.0 2.0\n")
    ingested = ingest(_write(tmp_path, body), read_at=_WHEN)
    assert ingested.dataset is not None
    assert ingested.dataset.grid.direction is DepthDirection.DESCENDING
    before = ingested.dataset.grid.values.tobytes()
    result = classify(ingested)
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert result.direction is not DepthDirection.DESCENDING
    assert result.direction is None
    assert ingested.dataset.grid.values.tobytes() == before
    structural = _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert any("decreases" in message for message in structural)
    assert any("negative" in message for message in structural)


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


def test_fixture_34_10_19_is_regular_at_the_written_step():
    result = _fixture("34_10-19.las")
    assert result.status is StageStatus.COMPLETE
    assert result.direction is DepthDirection.ASCENDING
    assert result.regularity is DepthRegularity.REGULAR
    assert result.corrected_step is None
    assert result.observed_step is not None
    assert result.observed_step.text == "0.15200000"
    assert any("not a recognised standard interval" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert any("STOP" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert _kind_counts(result) == {"INFORMATIONAL": 2}


def test_fixture_30_3_5_s_matches_the_34_pattern():
    result = _fixture("30_3-5 S.las")
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert result.corrected_step is None
    assert result.observed_step is not None
    assert result.observed_step.text == "0.15200000"
    assert any("not a recognised standard interval" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert any("STOP" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert _kind_counts(result) == {"INFORMATIONAL": 2}


def test_fixture_159_is_regular_at_0_1524_after_the_wrapped_reread():
    result = _fixture("159-19A_LFP.las")
    assert result.status is StageStatus.COMPLETE
    assert result.regularity is DepthRegularity.REGULAR
    assert result.corrected_step is None
    assert result.observed_step is not None
    assert result.observed_step.text == "0.1524"
    assert not any("not a recognised standard interval" in finding.message for finding in result.findings)
    assert not _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert _kind_counts(result) == {}


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


class DecimalText:
    """Build an exact decimal spelling without binary float noise."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __rmul__(self, index: int) -> str:
        from decimal import Decimal

        return format(Decimal(index) * Decimal(self.text), "f")
