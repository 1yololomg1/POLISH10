"""Stage 0 ingest. D64, D65 (PARTIAL), D66 (PARTIAL, F19), D68 (PARTIAL).

Conformance cases are small synthetic LAS strings. Evidence level is in the
test id. D65, D66 and D68 are not verified against the CWLS specification
(OD-24).
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from polish.model.null_choice import NullChoice
from polish.model.states import BlockType, StageStatus
from polish.stages.s00_ingest import CONTRACT, UnclassifiedIngest, ingest

_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures"
_GOLDEN = _REPO / "tests" / "golden"
_WHEN = datetime.datetime(2026, 9, 26, tzinfo=datetime.timezone.utc)


def _write(directory: Path, body: str, name: str = "well.las") -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _read(path: Path):
    return ingest(path, read_at=_WHEN)


def _blocks(result) -> list[BlockType]:
    return [finding.block_type for finding in result.findings]


def _messages(result, block: BlockType) -> list[str]:
    return [finding.message for finding in result.findings if finding.block_type is block]


def _las(
    *,
    vers: str = "2.0",
    wrap: str = "NO",
    strt: str = "0.0",
    stop: str = "1.0",
    step: str = "1.0",
    null: str | None = "-999.25",
    curves: str = "DEPT.M : DEPTH\nGR.GAPI : GAMMA\n",
    well_extra: str = "",
    between: str = "",
    data_header: str = "~A",
    data: str = "0.0 1.0\n1.0 2.0\n",
    version_header: str = "~VERSION INFORMATION",
    well_header: str = "~WELL INFORMATION",
    curve_header: str = "~CURVE INFORMATION",
) -> str:
    null_line = "" if null is None else f"NULL. {null} : NULL\n"
    return (
        f"{version_header}\n"
        f"VERS. {vers} : CWLS\n"
        f"WRAP. {wrap} : WRAP\n"
        f"{well_header}\n"
        f"STRT.M {strt} : START\n"
        f"STOP.M {stop} : STOP\n"
        f"STEP.M {step} : STEP\n"
        f"{null_line}"
        f"{well_extra}"
        f"{curve_header}\n"
        f"{curves}"
        f"{between}"
        f"{data_header}\n"
        f"{data}"
    )


def test_stage_contract_is_read_only_and_has_no_dependencies():
    assert CONTRACT.stage_number == 0
    assert CONTRACT.alters_data is False
    assert CONTRACT.depends_on_stages == ()
    assert any("not strictly monotonic" in line for line in CONTRACT.declines_when)


@pytest.mark.parametrize("evidence", ["PARTIAL-D65-first-char-after-tilde"])
def test_partial_d65_first_character_after_tilde_selects_the_section(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    # ~CURVE and ~ASCII are the long titles. Only the first character counts.
    body = _las(
        version_header="~VERSION INFORMATION",
        well_header="~WELL INFORMATION",
        curve_header="~CURVE INFORMATION",
        data_header="~ASCII LOG DATA",
    )
    result = _read(_write(tmp_path, body))
    assert [mark.letter for mark in result.audit.sections] == ["V", "W", "C", "A"]
    assert result.dataset is not None
    assert [curve.original_mnemonic for curve in result.dataset.curves] == ["DEPT", "GR"]
    assert result.dataset.grid.values.size == 2
    assert BlockType.STRUCTURAL_DATA_FAILURE not in _blocks(result)


@pytest.mark.parametrize("evidence", ["PARTIAL-D65-ascii-section-is-last"])
def test_partial_d65_data_section_is_the_last_section_whose_letter_is_a(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(data="0.0 1.0\n1.0 2.0\n~PARAMETER INFORMATION\nX. 1 : Y\n")
    result = _read(_write(tmp_path, body))
    assert result.audit.ascii_is_last is False
    # ~A not last is now an informational ledger entry. The dataset is still built.
    assert any("not the last section" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert result.dataset is not None
    assert result.dataset.grid.values.size == 2
    assert result.audit.depth_step_count == 2


@pytest.mark.parametrize("evidence", ["PARTIAL-D65-first-period-last-colon"])
def test_partial_d65_header_splits_on_first_period_and_last_colon(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(well_extra="MARK.HH:MM  01:02 : LEFT:RIGHT\n")
    result = _read(_write(tmp_path, body))
    mark = next(line for line in result.audit.header_lines if line.mnemonic == "MARK")
    assert mark.unit == "HH:MM"
    assert mark.value_text == "01:02 : LEFT"
    assert mark.description == "RIGHT"


@pytest.mark.parametrize("evidence", ["PARTIAL-D65-column-count-equals-curve-count"])
def test_partial_d65_column_count_matches_the_curve_count(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    result = _read(_write(tmp_path, _las()))
    assert result.dataset is not None
    assert len(result.dataset.curves) == 2
    assert result.dataset.grid.values.size == 2
    assert result.audit.column_mismatches == ()
    assert BlockType.STRUCTURAL_DATA_FAILURE not in _blocks(result)


@pytest.mark.parametrize("evidence", ["PARTIAL-D65-wrap-yes"])
def test_partial_d65_wrap_yes_joins_lines_into_depth_steps(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(wrap="YES", data="0.0\n1.5\n1.0\n2.5\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.audit.wrapped is True
    assert result.dataset.grid.values.size == 2
    assert result.dataset.curves[1].values.tolist() == [1.5, 2.5]


@pytest.mark.parametrize("evidence", ["PARTIAL-D65-comments-and-blanks"])
def test_partial_d65_comments_and_blank_lines_are_not_data(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(data="0.0 1.0\n\n# a comment\n1.0 2.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.dataset.grid.values.size == 2
    assert result.audit.column_mismatches == ()


@pytest.mark.parametrize("evidence", ["PARTIAL-D65-column-mismatch-recorded"])
def test_partial_d65_column_mismatch_is_recorded_and_not_padded(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(data="0.0 1.0 9.0\n1.0 2.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is None
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    messages = _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert len(messages) == 1
    assert "3" in messages[0]
    assert "~C count is 2" in messages[0]
    assert result.audit.column_mismatches
    assert result.audit.column_mismatches[0][1:] == (3, 2)


@pytest.mark.parametrize("evidence", ["PARTIAL-D65-negative-step"])
def test_partial_d65_negative_step_is_kept(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(strt="2.0", stop="1.0", step="-1.0", data="2.0 10\n1.0 11\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.dataset.grid.declared_step == -1.0
    # Stage 0 no longer classifies regularity. None means not yet classified.
    assert result.dataset.grid.regularity is None
    assert result.dataset.grid.direction.value == "DESCENDING"
    assert result.dataset.grid.values.tolist() == [2.0, 1.0]
    assert BlockType.STRUCTURAL_DATA_FAILURE not in _blocks(result)


@pytest.mark.parametrize("evidence", ["PARTIAL-D66-null-tolerance"])
def test_partial_d66_null_within_tolerance_is_absent_and_pass2_disagreement_is_structural(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    # -999.2500004 matches -999.25 at rel_tol 1e-6. lasio keeps the number.
    # That value change is the approved structural disagreement.
    body = _las(data="0.0 -999.2500004\n1.0 5.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert math.isnan(result.dataset.curves[1].values[0])
    assert result.dataset.curves[1].values[1] == 5.0
    assert result.null_sample_indexes[1] == (0,)
    assert result.status is StageStatus.COMPLETE
    notes = [item for item in result.findings if item.check == "null_within_tolerance"]
    assert len(notes) == 1
    assert notes[0].message == (
        "1 sample(s) in GR are within the NULL tolerance of -999.25 "
        "without being exactly it (first: -999.2500004 at index 0). They were read as absent."
    )
    assert not any("disagree" in item.message for item in result.findings)


@pytest.mark.parametrize("evidence", ["PARTIAL-D66-outside-tolerance-stays-number"])
def test_partial_d66_value_outside_null_tolerance_stays_a_number(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(data="0.0 -998.0\n1.0 5.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.dataset.curves[1].values[0] == -998.0
    assert result.null_sample_indexes[1] == ()
    assert BlockType.RULE_DEPENDENT_WARNING not in _blocks(result)
    assert BlockType.STRUCTURAL_DATA_FAILURE not in _blocks(result)


@pytest.mark.parametrize("evidence", ["PARTIAL-D66-undeclared-null"])
def test_partial_d66_undeclared_null_converts_nothing(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(null=None, data="0.0 -999.25\n1.0 1.5\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.audit.null_number is None
    assert result.dataset.curves[1].values.tolist() == [-999.25, 1.5]
    assert result.null_sample_indexes[1] == ()
    ambiguous = _messages(result, BlockType.AMBIGUOUS_INTERPRETATION)
    assert any("NULL is not declared" in message for message in ambiguous)
    notes = [item for item in result.findings if item.check == "sentinel_candidate"]
    assert len(notes) == 1
    assert notes[0].message == (
        "Column 1 'GR' has 1 sample(s) equal to -999.25, a common NULL value "
        "that is not this file's declared NULL. They were kept as numbers."
    )
    assert json.loads(next(item for item in result.findings if item.check == "null_undeclared").evidence.as_dict()["options"]) == [
        "-999.25",
        "NONE",
    ]
    assert BlockType.STRUCTURAL_DATA_FAILURE not in _blocks(result)


@pytest.mark.parametrize("evidence", ["PARTIAL-D66-undeclared-sentinel-not-converted"])
def test_partial_d66_sentinel_that_is_not_the_declared_null_stays_a_number(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(null="-999.25", data="0.0 -999\n1.0 1.5\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.dataset.curves[1].values[0] == -999.0
    assert not math.isnan(result.dataset.curves[1].values[0])
    notes = [item for item in result.findings if item.check == "sentinel_candidate"]
    assert len(notes) == 1
    assert "-999" in notes[0].message
    assert "kept as numbers" in notes[0].message
    assert notes[0].block_type is BlockType.INFORMATIONAL
    assert result.null_sample_indexes[1] == ()


@pytest.mark.parametrize("evidence", ["PARTIAL-D66-scientific-notation-is-not-nonconforming"])
def test_partial_d66_scientific_notation_is_a_number(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(data="0.0 1.5e-3\n1.0 2.0E+01\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.dataset.curves[1].values.tolist() == pytest.approx([0.0015, 20.0])
    blob = " ".join(finding.message.lower() for finding in result.findings)
    assert "nonconforming" not in blob
    assert "scientific" not in blob


@pytest.mark.parametrize("evidence", ["PARTIAL-D68-version-3-not-parsed"])
def test_partial_d68_version_3_is_reported_and_not_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str):
    assert evidence.startswith("PARTIAL-")
    called: list[object] = []

    def spy(*args, **kwargs):
        called.append(args)
        raise AssertionError("lasio.read should not run on a LAS 3.0 version")

    monkeypatch.setattr("lasio.read", spy)
    body = _las(vers="3.0", data="0.0 1.0\n1.0 2.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is None
    assert called == []
    warnings = _messages(result, BlockType.RULE_DEPENDENT_WARNING)
    assert any("major version 3" in message for message in warnings)
    assert result.status is StageStatus.COMPLETE


@pytest.mark.parametrize("evidence", ["PARTIAL-D68-las3-feature-in-2-0-file"])
def test_partial_d68_las3_section_is_not_parsed_as_curves(tmp_path: Path, evidence: str):
    assert evidence.startswith("PARTIAL-")
    body = _las(between="~X_DEFINITION\nNOPE.M : NOT A CURVE\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert [curve.original_mnemonic for curve in result.dataset.curves] == ["DEPT", "GR"]
    assert any("LAS 3.0" in message for message in _messages(result, BlockType.RULE_DEPENDENT_WARNING))
    assert any("unrecognised section" in message for message in _messages(result, BlockType.INFORMATIONAL))


def test_header_line_that_does_not_split_is_informational(tmp_path: Path):
    body = _las(well_extra="THIS LINE HAS NO PERIOD\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    messages = _messages(result, BlockType.INFORMATIONAL)
    assert any("does not split" in message for message in messages)


def test_unrecognised_section_is_informational(tmp_path: Path):
    body = _las(between="~Z NOTES\nHELLO THERE\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    messages = _messages(result, BlockType.INFORMATIONAL)
    assert any("unrecognised section" in message and "~Z" in message for message in messages)


def test_encoding_undeterminable_is_structural(tmp_path: Path):
    path = tmp_path / "bad.las"
    path.write_bytes(b"\xff\x00 not utf-8")
    result = _read(path)
    assert result.dataset is None
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert result.audit.encoding_name is None
    message = next(item.message for item in result.findings if item.check == "encoding_undetermined")
    assert message.startswith("Line 1 has byte 0xFF, which is not valid UTF-8")
    assert "No encoding was assumed, so the file was not processed." in message


def test_missing_ascii_section_is_structural(tmp_path: Path):
    body = _las(data_header="", data="")
    # The helper still emits a blank data header line when data_header is empty,
    # which is not a section. Build the file without ~A directly.
    body = (
        "~V\nVERS. 2.0 : CWLS\nWRAP. NO : WRAP\n"
        "~W\nSTRT.M 0.0 : START\nSTOP.M 1.0 : STOP\nSTEP.M 1.0 : STEP\nNULL. -999.25 : NULL\n"
        "~C\nDEPT.M : DEPTH\nGR.GAPI : GAMMA\n"
    )
    result = _read(_write(tmp_path, body))
    assert result.dataset is None
    assert any("No ~A section" in message for message in _messages(result, BlockType.STRUCTURAL_DATA_FAILURE))


def test_ascii_section_with_no_data_rows_is_structural(tmp_path: Path):
    body = _las(data="\n# nothing here\n\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is None
    assert any("no data rows" in message for message in _messages(result, BlockType.STRUCTURAL_DATA_FAILURE))
    assert any("no adjacent pair" in message for message in _messages(result, BlockType.STRUCTURAL_DATA_FAILURE))
    assert result.status is StageStatus.REFUSED_STRUCTURAL


def test_duplicate_mnemonics_keep_two_curve_refs(tmp_path: Path):
    body = _las(
        curves="DEPT.M : DEPTH\nGR.GAPI : ONE\nGR.API : TWO\n",
        data="0.0 1.0 3.0\n1.0 2.0 4.0\n",
    )
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.audit.duplicate_mnemonics == ("GR",)
    first, second = result.dataset.curves[1], result.dataset.curves[2]
    assert first.original_mnemonic == second.original_mnemonic == "GR"
    assert first.ref != second.ref
    assert first.ref.column_index == 1
    assert second.ref.column_index == 2
    assert first.original_description == " ONE"
    assert second.original_description == " TWO"
    assert first.values.tolist() == [1.0, 2.0]
    assert second.values.tolist() == [3.0, 4.0]


def test_ignore_header_errors_is_used_only_after_the_line_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import lasio

    calls: list[bool] = []
    real = lasio.read

    def spy(path, **kwargs):
        calls.append(kwargs.get("ignore_header_errors"))
        return real(path, **kwargs)

    monkeypatch.setattr(lasio, "read", spy)
    clean = _read(_write(tmp_path, _las(), name="clean.las"))
    assert calls == [False]
    assert clean.dataset is not None

    calls.clear()
    body = _las(well_extra="\x00\n")
    result = _read(_write(tmp_path, body, name="nullbyte.las"))
    assert True in calls
    assert result.dataset is not None
    assert any("does not split" in message for message in _messages(result, BlockType.INFORMATIONAL))


def test_input_hash_is_unchanged_and_opens_are_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = _write(tmp_path, _las())
    before = path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    modes: list[str] = []
    real_open = Path.open

    def spy(self, mode="r", *args, **kwargs):
        modes.append(mode if isinstance(mode, str) else repr(mode))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    result = _read(path)
    assert path.read_bytes() == before
    assert result.source_file.file_hash.hex_digest == digest
    assert modes
    for mode in modes:
        assert "w" not in mode
        assert "a" not in mode
        assert "x" not in mode
        assert "+" not in mode
    assert result.dataset is not None
    assert result.dataset.curves[0].values.flags.writeable is False
    assert "strict UTF-8" in result.audit.encoding_method


def _fixture_paths() -> list[Path]:
    return sorted(path for path in _FIXTURES.glob("*.las") if "keough" not in path.name.lower())


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda path: path.name)
def test_fixture_headline_facts_match_the_golden_file(path: Path):
    facts = json.loads((_GOLDEN / f"{path.stem}.facts.json").read_text(encoding="utf-8"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = _read(path)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert after == before == facts["sha256"]
    assert result.dataset is not None
    assert result.status is StageStatus.COMPLETE
    assert len(result.dataset.curves) == len(facts["curves"])
    assert int(result.dataset.grid.values.size) == facts["depth_step_count"]
    assert result.audit.null_number == pytest.approx(facts["null"]["number"])
    assert result.audit.strt_number == pytest.approx(facts["strt"]["number"])
    assert result.audit.stop_number == pytest.approx(facts["stop"]["number"])
    assert result.audit.step_number == pytest.approx(facts["step"]["number"])
    assert result.dataset.grid.declared_step == pytest.approx(facts["step"]["number"])
    # Stage 0 no longer classifies regularity.
    assert result.dataset.grid.regularity is None
    structural = _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert structural == []
    for curve, golden in zip(result.dataset.curves, facts["curves"], strict=True):
        assert curve.original_mnemonic == golden["mnemonic"]
        assert curve.original_unit == golden["unit"]
        assert curve.values.flags.writeable is False
        assert len(result.null_sample_indexes[curve.ref.column_index]) == golden["null_count"]


def test_cli_ingest_json_writes_no_file(tmp_path: Path):
    path = _write(tmp_path, _las(), name="cli.las")
    before = sorted(item.name for item in tmp_path.iterdir())
    completed = subprocess.run(
        [sys.executable, "-m", "polish", "ingest", str(path), "--json"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    after = sorted(item.name for item in tmp_path.iterdir())
    assert completed.returncode == 0
    assert after == before
    payload = json.loads(completed.stdout)
    assert payload["dataset"]["curve_count"] == 2
    assert payload["dataset"]["row_count"] == 2
    assert payload["null"]["number"] == -999.25
    assert payload["status"] == "COMPLETE"
    assert payload["dataset"]["regularity"] is None
    assert payload["dataset"]["declared_step"] == 1.0
    assert payload["duplicate_sections"] == []
    assert payload["absent_sections"] == []
    assert payload["preamble_lines"] == []
    assert payload["repeated_well_mnemonics"] == []
    assert payload["repeated_version_mnemonics"] == []
    assert "ledger" in payload


def test_non_monotonic_depth_is_a_structural_refusal(tmp_path: Path):
    body = _las(data="0.0 1\n2.0 1\n1.0 1\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is None
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    messages = _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert any("Non-monotonic depth at indexes 1 and 2" in message and "2.0" in message and "1.0" in message for message in messages)


def test_duplicate_depth_is_a_structural_refusal(tmp_path: Path):
    body = _las(data="0.0 1\n1.0 1\n1.0 2\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is None
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    messages = _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert any("Duplicate depth at indexes 1 and 2" in message and "1.0" in message for message in messages)


def test_one_depth_row_is_a_structural_refusal(tmp_path: Path):
    body = _las(data="0.0 1.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is None
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    messages = _messages(result, BlockType.STRUCTURAL_DATA_FAILURE)
    assert any("index 0" in message and "0.0" in message and "no adjacent pair" in message for message in messages)


def test_blank_step_builds_a_dataset_without_a_step_finding(tmp_path: Path):
    body = _las(step="", data="0.0 1.0\n1.0 2.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.status is StageStatus.COMPLETE
    assert result.audit.step_number is None
    assert result.audit.step_text == ""
    assert result.dataset.grid.declared_step is None
    assert result.dataset.grid.regularity is None
    assert not any("STEP" in message for message in _messages(result, BlockType.STRUCTURAL_DATA_FAILURE))
    assert not any("STEP is" in message for message in (finding.message for finding in result.findings))


def test_non_numeric_step_builds_a_dataset_and_keeps_the_header_text(tmp_path: Path):
    body = _las(step="N/A", data="0.0 1.0\n1.0 2.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.audit.step_text == "N/A"
    assert result.audit.step_number is None
    assert result.dataset.grid.declared_step is None
    assert result.status is StageStatus.COMPLETE


def test_repeated_null_uses_the_first_value(tmp_path: Path):
    body = _las(null="-999.25", well_extra="NULL. -999 : SECOND\n", data="0.0 -999.25\n1.0 -999\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.audit.repeated_well_mnemonics == ("NULL",)
    assert math.isnan(result.dataset.curves[1].values[0])
    assert result.dataset.curves[1].values[1] == -999.0
    messages = _messages(result, BlockType.AMBIGUOUS_INTERPRETATION)
    assert any("NULL is repeated" in message and "-999.25" in message and "first value" in message for message in messages)


def test_repeated_vers_and_wrap_are_stored_and_the_first_is_used(tmp_path: Path):
    body = _las().replace(
        "WRAP. NO : WRAP\n",
        "WRAP. NO : WRAP\nVERS. 1.2 : SECOND\nWRAP. YES : SECOND\n",
        1,
    )
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.audit.vers_text == "2.0"
    assert result.audit.wrap_text == "NO"
    assert result.audit.wrapped is False
    assert result.audit.repeated_version_mnemonics == ("VERS", "WRAP")
    messages = _messages(result, BlockType.AMBIGUOUS_INTERPRETATION)
    assert any("VERS is repeated" in message and "2.0" in message for message in messages)
    assert any("WRAP is repeated" in message and "NO" in message for message in messages)


def test_non_finite_null_converts_nothing(tmp_path: Path):
    body = _las(null="N/A", data="0.0 1.0\n1.0 2.0\n")
    result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.audit.null_text == "N/A"
    assert result.audit.null_number is None
    assert result.dataset.curves[1].values.tolist() == [1.0, 2.0]
    messages = _messages(result, BlockType.AMBIGUOUS_INTERPRETATION)
    assert any("not a finite number" in message and "N/A" in message for message in messages)


def test_duplicate_section_absent_section_and_preamble_are_informational(tmp_path: Path):
    duplicate = (
        "~V\nVERS. 2.0 : CWLS\nWRAP. NO : WRAP\n"
        "~W\nSTRT.M 0 : S\nSTOP.M 1 : S\nSTEP.M 1 : S\nNULL. -999.25 : N\n"
        "~W\nSTRT.M 9 : LATER\n"
        "~C\nDEPT.M : D\nGR.GAPI : G\n"
        "~A\n0 1\n1 2\n"
    )
    result = _read(_write(tmp_path, duplicate, name="dup.las"))
    assert result.dataset is not None
    assert "W" in result.audit.duplicate_section_letters
    assert result.audit.strt_number == 0.0
    assert any("Section ~W appears 2 times" in message for message in _messages(result, BlockType.INFORMATIONAL))
    assert any("STRT is repeated" in message and "0" in message for message in _messages(result, BlockType.AMBIGUOUS_INTERPRETATION))

    preamble = "before any section\n" + _las()
    preamble_result = _read(_write(tmp_path, preamble, name="preamble.las"))
    assert preamble_result.audit.preamble_lines == (1,)
    assert any("before the first section" in message for message in _messages(preamble_result, BlockType.INFORMATIONAL))

    missing_curve = (
        "~V\nVERS. 2.0 : CWLS\nWRAP. NO : WRAP\n"
        "~W\nSTRT.M 0 : S\nSTOP.M 1 : S\nSTEP.M 1 : S\nNULL. -999.25 : N\n"
        "~A\n0 1\n1 2\n"
    )
    missing = _read(_write(tmp_path, missing_curve, name="noc.las"))
    assert "C" in missing.audit.absent_section_letters
    assert any("Section ~C is absent." in message for message in _messages(missing, BlockType.INFORMATIONAL))

    missing_version = (
        "preface\n"
        "~W\nSTRT.M 0 : S\nSTOP.M 1 : S\nSTEP.M 1 : S\nNULL. -999.25 : N\n"
        "~C\nDEPT.M : D\nGR.GAPI : G\n"
        "~A\n0 1\n1 2\n"
    )
    path = _write(tmp_path, missing_version, name="nov.las")
    with pytest.raises(UnclassifiedIngest) as caught:
        _read(path)
    partial = caught.value.partial
    assert partial.status is StageStatus.BLOCKED
    assert partial.dataset is None
    assert any("Section ~V is absent." in message for message in _messages(partial, BlockType.INFORMATIONAL))
    assert any("VERS is not declared" in message for message in _messages(partial, BlockType.INFORMATIONAL))


def test_unclassified_halt_is_blocked_and_chooses_no_block_type(tmp_path: Path):
    # WRAP is neither YES nor NO. That condition has no approved block type.
    body = _las(wrap="MAYBE", data="0.0 1.0\n1.0 2.0\n")
    path = _write(tmp_path, body)
    with pytest.raises(UnclassifiedIngest) as caught:
        _read(path)
    partial = caught.value.partial
    assert partial.status is StageStatus.BLOCKED
    assert partial.dataset is None
    assert BlockType.STRUCTURAL_DATA_FAILURE not in _blocks(partial)
    completed = subprocess.run(
        [sys.executable, "-m", "polish", "ingest", str(path), "--json"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "BLOCKED"
    assert payload["dataset"] is None


def test_wrapped_file_is_reread_with_the_normal_engine_without_a_finding(tmp_path: Path, caplog):
    body = _las(
        wrap="YES",
        strt="0.0",
        stop="1.0",
        step="1.0",
        data="0.0\n10.0\n1.0\n20.0\n",
    )
    with caplog.at_level("WARNING", logger="lasio.las"):
        result = _read(_write(tmp_path, body))
    assert result.dataset is not None
    assert result.audit.depth_tokens == ("0.0", "1.0")
    assert not any("engine" in finding.message for finding in result.findings)
    assert "Only engine='normal' can read wrapped files" not in caplog.text


def test_declared_version_two_is_recorded_as_parsed_like_las_2(tmp_path: Path):
    result = _read(_write(tmp_path, _las()))
    messages = _messages(result, BlockType.INFORMATIONAL)
    assert any("not major version 3" in message and "parsed as LAS 2.0" in message for message in messages)
    assert result.status is StageStatus.COMPLETE
    assert result.dataset is not None


def test_an_html_error_page_is_not_a_las_file(tmp_path: Path):
    page = (
        "<!DOCTYPE html><html><head><title>502 Proxy Error</title></head>"
        "<body><h1>502 Proxy Error</h1></body></html>\n"
    )
    path = tmp_path / "proxy.las"
    path.write_bytes(page.encode("utf-8"))
    result = ingest(path, read_at=_WHEN)
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert result.dataset is None
    sections = [item for item in result.findings if item.check == "no_sections"]
    assert len(sections) == 1
    assert sections[0].message == (
        "This file has no LAS sections (no line starting with ~), so it is not a LAS file and was not processed."
    )
    assert [item.check for item in result.findings] == ["no_sections"]


def test_any_lasio_exception_is_pass2_no_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def broken(*_args, **_kwargs):
        raise KeyError("No ~ sections found")

    monkeypatch.setattr("lasio.read", broken)
    result = _read(_write(tmp_path, _las()))
    notes = [item for item in result.findings if item.check == "pass2_no_parse"]
    assert len(notes) == 1
    assert "KeyError" in notes[0].message
    assert result.dataset is not None


def _lasio_source_is_text(source: object) -> bool:
    """Pass 2 may receive decoded text or a stream. Never a path or a URL."""

    if isinstance(source, io.StringIO):
        return True
    return isinstance(source, str) and "\n" in source


def test_ingest_given_a_url_shaped_file_name_never_calls_urlopen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[object] = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    import lasio

    received: list[object] = []
    real = lasio.read

    def spy(source, *args, **kwargs):
        received.append(source)
        assert _lasio_source_is_text(source)
        assert kwargs.get("null_policy") == "none"
        return real(source, *args, **kwargs)

    monkeypatch.setattr(lasio, "read", spy)
    with pytest.raises(OSError):
        ingest("http://example.invalid/well.las", read_at=_WHEN)
    result = ingest(_write(tmp_path, _las()), read_at=_WHEN)
    assert result.status is StageStatus.COMPLETE
    assert received
    assert all(_lasio_source_is_text(item) for item in received)
    assert calls == []


def test_depths_from_9997_raise_no_sentinel_finding(tmp_path: Path):
    depths = (9997.0, 9998.0, 9999.0, 10000.0, 10000.5)
    data = "".join(f"{depth} 1.0\n" for depth in depths)
    body = _las(strt="9997.0", stop="10000.5", step="1.0", data=data)
    result = _read(_write(tmp_path, body))
    assert result.status is StageStatus.COMPLETE
    assert result.dataset is not None
    assert result.dataset.grid.values.tolist() == list(depths)
    assert not any(item.check == "sentinel_candidate" for item in result.findings)


def test_null_zero_keeps_a_zero_depth(tmp_path: Path):
    body = _las(null="0", data="0.0 1.0\n1.0 2.0\n")
    result = _read(_write(tmp_path, body))
    assert result.status is StageStatus.COMPLETE
    assert result.dataset is not None
    assert result.dataset.grid.values.tolist() == [0.0, 1.0]
    assert result.dataset.curves[1].values.tolist() == [1.0, 2.0]


def test_a_trailing_ctrl_z_is_ignored_and_the_hash_covers_it(tmp_path: Path):
    body = _las().encode("utf-8") + b"\x1a"
    path = tmp_path / "eof.las"
    path.write_bytes(body)
    result = ingest(path, read_at=_WHEN)
    assert result.status is StageStatus.COMPLETE
    assert result.dataset is not None
    notes = [item.message for item in result.findings if item.check == "dos_end_of_file"]
    assert notes == ["The file ends with a DOS end-of-file byte (Ctrl-Z). It was ignored; no value changed."]
    assert result.source_file.file_hash.hex_digest == hashlib.sha256(body).hexdigest()


def test_a_ctrl_z_that_is_not_trailing_is_not_stripped(tmp_path: Path):
    body = _las().encode("utf-8").replace(b"0.0 1.0", b"0.0\x1a 1.0", 1)
    path = tmp_path / "mid.las"
    path.write_bytes(body)
    result = ingest(path, read_at=_WHEN)
    assert not any(item.check == "dos_end_of_file" for item in result.findings)
    assert any(item.check == "data_token_not_numeric" for item in result.findings)


@pytest.mark.parametrize("token", ["5_0.0", "nan", "inf", "-1.#IND"])
def test_a_data_token_that_is_not_a_las_number_is_structural(tmp_path: Path, token: str):
    body = _las(data=f"0.0 {token}\n1.0 2.0\n")
    result = _read(_write(tmp_path, body))
    assert result.status is StageStatus.REFUSED_STRUCTURAL
    assert result.dataset is None
    findings = [item for item in result.findings if item.check == "data_token_not_numeric"]
    assert len(findings) == 1
    assert findings[0].message.startswith("Line ")
    assert f"has '{token}' where a number should be, so the file was not processed." in findings[0].message


def test_a_byte_that_is_not_utf8_names_the_line_and_windows_1252(tmp_path: Path):
    raw = bytearray()
    for number in range(1, 14):
        raw += f"# line {number}\n".encode("ascii")
    raw += b"degree \xb0\n"
    path = tmp_path / "latin.las"
    path.write_bytes(bytes(raw))
    result = ingest(path, read_at=_WHEN)
    message = next(item.message for item in result.findings if item.check == "encoding_undetermined")
    assert message == (
        "Line 14 has byte 0xB0, which is not valid UTF-8 (in Windows-1252 it would be '\u00b0'). "
        "LAS 2.0 allows plain ASCII only. No encoding was assumed, so the file was not processed."
    )
    assert result.status is StageStatus.REFUSED_STRUCTURAL


def test_an_undefined_windows_1252_byte_is_named(tmp_path: Path):
    path = tmp_path / "undefined.las"
    path.write_bytes(b"~\x81\n")
    result = ingest(path, read_at=_WHEN)
    message = next(item.message for item in result.findings if item.check == "encoding_undetermined")
    assert "Line 1 has byte 0x81" in message
    assert "not defined in Windows-1252" in message


def test_a_missing_null_can_be_chosen_and_an_unlisted_option_is_refused(tmp_path: Path):
    body = _las(null=None, data="0.0 -999.25\n1.0 1.5\n")
    path = _write(tmp_path, body)
    asked = ingest(path, read_at=_WHEN)
    question = next(item for item in asked.findings if item.check == "null_undeclared")
    assert json.loads(question.evidence.as_dict()["options"]) == ["-999.25", "NONE"]
    chosen = ingest(path, read_at=_WHEN, null_choices=(NullChoice("-999.25", "null-1"),))
    assert chosen.status is StageStatus.COMPLETE
    assert chosen.null_decision_id == "null-1"
    assert chosen.dataset is not None
    assert math.isnan(chosen.dataset.curves[1].values[0])
    assert chosen.dataset.curves[1].values[1] == 1.5
    assert not any(item.check in ("null_undeclared", "null_not_finite") for item in chosen.findings)
    none = ingest(path, read_at=_WHEN, null_choices=(NullChoice("NONE", "null-2"),))
    assert none.dataset is not None
    assert none.dataset.curves[1].values.tolist() == [-999.25, 1.5]
    assert none.null_decision_id == "null-2"
    with pytest.raises(ValueError, match="not an option this finding lists"):
        ingest(path, read_at=_WHEN, null_choices=(NullChoice("-999", "null-3"),))
