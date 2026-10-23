"""Describe report: fixtures, schema, refusals, and the command. D95, D19, C5, C6."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from polish.provenance.describe import NOTICE, OutputRefused, build_report, write_report
from tests.arch.import_walk import imported_names

_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures"
_GOLDEN = _REPO / "tests" / "golden"
_SCHEMA = _REPO / "polish" / "provenance" / "schema" / "describe.schema.json"
_LAYERS = ("stages", "model", "decisions")


def _facts(path: Path) -> dict:
    return json.loads((_GOLDEN / f"{path.stem}.facts.json").read_text(encoding="utf-8"))


def _schema() -> dict:
    return json.loads(_SCHEMA.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _names(directory: Path) -> list[str]:
    return sorted(item.name for item in directory.iterdir())


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "polish", *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("path", sorted(_FIXTURES.glob("*.las")), ids=lambda item: item.name)
def test_each_fixture_validates_and_matches_golden_facts(path: Path, tmp_path: Path):
    facts = _facts(path)
    report = build_report(path, local_tables_directory=tmp_path / "no-local-tables")
    document = report.document
    jsonschema.validate(document, _schema())
    stage0 = document["stage0"]
    assert stage0["dataset"]["curve_count"] == len(facts["curves"])
    assert len(document["curves"]) == len(facts["curves"])
    assert stage0["dataset"]["row_count"] == facts["depth_step_count"]
    assert stage0["null"]["text"] == facts["null"]["text"]
    assert stage0["null"]["number"] == facts["null"]["number"]
    for key in ("strt", "stop", "step"):
        assert stage0[key]["text"] == facts[key]["text"]
        assert stage0[key]["unit"] == facts[key]["unit"]
        assert stage0[key]["number"] == facts[key]["number"]
    assert document["schema_version"] == "polish-describe/1"
    assert document["notice"] == NOTICE
    assert document["default_passes"] == []
    assert document["file"]["sha256"] == facts["sha256"]
    assert document["local_tables"]["found"] is False
    assert document["local_tables"]["read"] is False
    unverified = {item["file"] for item in document["not_verified"]}
    assert "null_comparison.json" in unverified
    assert "null_sentinels.json" in unverified
    markdown = report.markdown().splitlines()
    assert markdown[0] == document["result"]
    assert markdown[1] == NOTICE
    text = report.markdown()
    assert "## Refusals and non-eligibility" in text
    if not document["refusals"]:
        assert "\nNo refusals\n" in text
    for group in document["findings_by_block"]:
        for finding in group["findings"]:
            assert finding["open"] == "Open: no decision recorded"


def test_a_structural_refusal_is_written_and_the_command_exits_2(tmp_path: Path):
    source = tmp_path / "no-ascii.las"
    source.write_text(
        "~V\nVERS. 2.0 : CWLS\nWRAP. NO : WRAP\n"
        "~W\nSTRT.M 0.0 : START\nSTOP.M 1.0 : STOP\nSTEP.M 1.0 : STEP\nNULL. -999.25 : NULL\n"
        "~C\nDEPT.M : DEPTH\nGR.GAPI : GAMMA\n",
        encoding="utf-8",
        newline="\n",
    )
    out = tmp_path / "out"
    completed = _run("describe", str(source), "--out", str(out))
    assert completed.returncode == 2
    written = sorted(out.iterdir())
    assert [item.name for item in written] == ["no-ascii.describe.json", "no-ascii.describe.md"]
    markdown = written[1].read_text(encoding="utf-8")
    assert markdown.splitlines()[0].startswith("Refused:")
    assert "No ~A section" in markdown.splitlines()[0]
    assert "\nNo refusals\n" not in markdown
    assert completed.stdout.splitlines() == [str(written[0].resolve()), str(written[1].resolve())]
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    jsonschema.validate(payload, _schema())
    assert payload["structural_refusal"] is True
    assert payload["stage0"]["dataset"] is None


def test_refusals_section_says_no_refusals_when_there_are_none(tmp_path: Path):
    path = _FIXTURES / "new_file_null_minus_999.las"
    report = build_report(path, local_tables_directory=tmp_path / "no-local-tables")
    assert report.document["refusals"] == []
    assert "\nNo refusals\n" in report.markdown()
    assert "## Refusals and non-eligibility" in report.markdown()


def test_output_equal_to_the_input_is_refused(tmp_path: Path):
    source = tmp_path / "well.las"
    source.write_text("~V\n", encoding="utf-8", newline="\n")
    before = source.read_bytes()
    with pytest.raises(OutputRefused):
        write_report(
            build_report(_FIXTURES / "new_file_null_minus_999.las", local_tables_directory=tmp_path / "absent"),
            source,
            source,
        )
    assert source.read_bytes() == before
    assert _names(tmp_path) == ["well.las"]


def test_an_existing_target_is_refused_unless_overwrite_is_given(tmp_path: Path):
    source = _FIXTURES / "new_file_null_minus_999.las"
    out = tmp_path / "out"
    out.mkdir()
    json_path = out / "new_file_null_minus_999.describe.json"
    markdown_path = out / "new_file_null_minus_999.describe.md"
    json_path.write_text("OLD JSON", encoding="utf-8", newline="\n")
    markdown_path.write_text("OLD MD", encoding="utf-8", newline="\n")
    report = build_report(source, local_tables_directory=tmp_path / "absent")
    with pytest.raises(OutputRefused):
        write_report(report, source, out, overwrite=False)
    assert json_path.read_text(encoding="utf-8") == "OLD JSON"
    assert markdown_path.read_text(encoding="utf-8") == "OLD MD"
    written = write_report(report, source, out, overwrite=True)
    assert written == (json_path.resolve(), markdown_path.resolve())
    assert json_path.read_text(encoding="utf-8") != "OLD JSON"
    assert markdown_path.read_text(encoding="utf-8") != "OLD MD"
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == "polish-describe/1"


def test_command_exit_codes_are_0_1_and_2(tmp_path: Path):
    source = _FIXTURES / "new_file_null_minus_999.las"
    out = tmp_path / "out"
    ok = _run("describe", str(source), "--out", str(out))
    assert ok.returncode == 0
    assert ok.stderr == ""
    assert len(ok.stdout.splitlines()) == 2
    missing = _run("describe", str(tmp_path / "missing.las"), "--out", str(tmp_path / "unused"))
    assert missing.returncode == 1
    assert missing.stderr.strip() != ""
    assert not (tmp_path / "unused").exists()
    blocked = _run("describe", str(source), "--out", str(out))
    assert blocked.returncode == 1
    assert "already exists" in blocked.stderr
    refused = tmp_path / "no-ascii.las"
    refused.write_text(
        "~V\nVERS. 2.0 : CWLS\nWRAP. NO : WRAP\n"
        "~W\nSTRT.M 0.0 : START\nSTOP.M 1.0 : STOP\nSTEP.M 1.0 : STEP\nNULL. -999.25 : NULL\n"
        "~C\nDEPT.M : DEPTH\nGR.GAPI : GAMMA\n",
        encoding="utf-8",
        newline="\n",
    )
    structural = _run("describe", str(refused), "--out", str(tmp_path / "structural"))
    assert structural.returncode == 2
    assert structural.stdout.splitlines()[0].endswith("no-ascii.describe.json")


def test_describe_does_not_change_the_input_or_write_anything_else(tmp_path: Path):
    source = _FIXTURES / "new_file_null_minus_999.las"
    before_names = _names(source.parent)
    before_hash = _sha256(source)
    out = tmp_path / "out"
    completed = _run("describe", str(source), "--out", str(out))
    assert completed.returncode == 0
    assert _sha256(source) == before_hash
    assert _names(source.parent) == before_names
    assert _names(out) == [
        "new_file_null_minus_999.describe.json",
        "new_file_null_minus_999.describe.md",
    ]
    assert not list(out.glob("*.partial"))


def test_engine_layers_do_not_import_provenance():
    offenders: list[str] = []
    for folder in _LAYERS:
        for path, line, name in imported_names(_REPO / "polish" / folder):
            if name == "polish.provenance" or name.startswith("polish.provenance."):
                offenders.append(f"{path}:{line} {name}")
    assert offenders == []
