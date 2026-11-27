"""Committed facts files match a fresh read of each fixture. D27.

A mismatch means the fixture bytes or the extractor changed. It does not mean
the facts have been accepted as petrophysically correct.
"""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tests.golden.extract_facts import (
    FactsError,
    canonical_json,
    extract_facts,
    facts_path_for,
    is_keough,
    iter_fixtures,
    write_fixture_facts,
)

_GOLDEN = Path(__file__).resolve().parent
_FIXTURES = _GOLDEN.parent / "fixtures"
_EXTRACTOR = _GOLDEN / "extract_facts.py"

# Names this reader must not pull in. The facts come from the file text.
_BANNED_IMPORTS = {
    "polish",
    "polish_ui",
    "core",
    "ui",
    "petrophysics",
    "lasio",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "tkinter",
    "advanced_preprocessing_system10",
}


def _fixture_paths() -> list[Path]:
    return iter_fixtures(_FIXTURES)


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda path: path.name)
def test_committed_facts_match_a_fresh_read(path: Path):
    committed_path = facts_path_for(path, _GOLDEN)
    assert committed_path.is_file()
    fresh = canonical_json(extract_facts(path))
    committed = committed_path.read_text(encoding="utf-8")
    assert fresh == committed


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda path: path.name)
def test_recorded_sha256_is_the_file_bytes(path: Path):
    facts = json.loads(facts_path_for(path, _GOLDEN).read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert facts["sha256"] == digest
    assert facts["fixture"] == path.name


def test_extractor_source_does_not_import_polish_lasio_or_the_monolith():
    tree = ast.parse(_EXTRACTOR.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
    banned = sorted({name for name in found if name in _BANNED_IMPORTS})
    assert banned == []


def test_keough_names_are_excluded():
    assert is_keough(Path("KEOUGH_12-34.las"))
    assert is_keough(Path("1052970244_keough.LAS"))
    assert not is_keough(Path("159-19A_LFP.las"))
    assert all(not is_keough(path) for path in _fixture_paths())


def _write_las(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


_HEADER = """\
~VERSION INFORMATION
VERS. 2.0 : CWLS
WRAP. {wrap} : LINES
~WELL INFORMATION
STRT.M 0.0 : START
STOP.M 1.0 : STOP
STEP.M 1.0 : STEP
NULL. -999.25 : NULL
~CURVE INFORMATION
DEPT.M : DEPTH
GR.GAPI : GAMMA
RHOB.G/C3 : DENSITY
~A
"""


def test_wrapped_file_joins_lines_into_depth_steps(tmp_path: Path):
    # Three curves. Each step is split: depth alone, then the two log values.
    body = _HEADER.format(wrap="YES") + (
        " 0.0\n"
        " 10.0 -999.25\n"
        " 1.0\n"
        " 20.0 2.5\n"
    )
    facts = extract_facts(_write_las(tmp_path, "wrapped.las", body))
    assert facts["wrap"]["wrapped"] is True
    assert facts["depth_step_count"] == 2
    # First data step is 0.0, 10.0, -999.25. Second is 1.0, 20.0, 2.5.
    gr = facts["curves"][1]
    rhob = facts["curves"][2]
    assert gr["null_count"] == 0
    assert gr["min_non_null"] == 10.0
    assert gr["max_non_null"] == 20.0
    assert rhob["null_count"] == 1
    assert rhob["first_non_null_depth"] == 1.0
    assert rhob["last_non_null_depth"] == 1.0
    assert rhob["min_non_null"] == 2.5
    assert rhob["max_non_null"] == 2.5
    assert facts["curves"][0]["null_count"] == 0
    assert facts["curves"][0]["min_non_null"] == 0.0
    assert facts["curves"][0]["max_non_null"] == 1.0


def test_unwrapped_file_keeps_one_line_per_depth_step(tmp_path: Path):
    body = _HEADER.format(wrap="NO") + (
        "0.0 10.0 -999.25\n"
        "1.0 20.0 2.5\n"
    )
    facts = extract_facts(_write_las(tmp_path, "unwrapped.las", body))
    assert facts["wrap"]["wrapped"] is False
    assert facts["depth_step_count"] == 2
    assert facts["curves"][1]["null_count"] == 0
    assert facts["curves"][1]["min_non_null"] == 10.0
    assert facts["curves"][2]["null_count"] == 1
    assert facts["curves"][2]["first_non_null_depth"] == 1.0
    assert facts["curves"][2]["min_non_null"] == 2.5
    assert facts["curves"][2]["max_non_null"] == 2.5


def test_unwrapped_short_line_is_not_joined_to_the_next(tmp_path: Path):
    body = _HEADER.format(wrap="NO") + "0.0 10.0\n-999.25\n"
    with pytest.raises(FactsError, match="WRAP NO"):
        extract_facts(_write_las(tmp_path, "short.las", body))


def test_wrapped_leftover_values_are_refused(tmp_path: Path):
    body = _HEADER.format(wrap="YES") + "0.0 10.0 -999.25 1.0\n"
    with pytest.raises(FactsError, match="left over"):
        extract_facts(_write_las(tmp_path, "leftover.las", body))


def test_relative_null_tolerance_is_one_part_in_a_million(tmp_path: Path):
    # |Δ| = 0.0005; 1e-6 * 999.25 ≈ 0.000999, so the near value is NULL.
    # |Δ| = 0.002 is outside that band, so it is a real sample.
    body = (
        "~V\n"
        "VERS. 2.0 :\n"
        "WRAP. NO :\n"
        "~W\n"
        "STRT.M 0 :\n"
        "STOP.M 2 :\n"
        "STEP.M 1 :\n"
        "NULL. -999.25 :\n"
        "~C\n"
        "DEPT.M : D\n"
        "GR.GAPI : G\n"
        "~A\n"
        "0 -999.25\n"
        "1 -999.2505\n"
        "2 -999.252\n"
    )
    facts = extract_facts(_write_las(tmp_path, "tolerance.las", body))
    gr = facts["curves"][1]
    assert gr["null_count"] == 2
    assert gr["first_non_null_depth"] == 2.0
    assert gr["last_non_null_depth"] == 2.0
    assert gr["min_non_null"] == pytest.approx(-999.252)
    assert facts["null"]["text"] == "-999.25"


def test_all_null_curve_has_no_span(tmp_path: Path):
    body = _HEADER.format(wrap="NO") + "0.0 -999.25 -999.25\n1.0 -999.25 -999.25\n"
    facts = extract_facts(_write_las(tmp_path, "all_null.las", body))
    gr = facts["curves"][1]
    assert gr["null_count"] == 2
    assert gr["first_non_null_depth"] is None
    assert gr["last_non_null_depth"] is None
    assert gr["min_non_null"] is None
    assert gr["max_non_null"] is None


@pytest.mark.parametrize(
    ("curve_line", "description"),
    [
        ("RES.OHMM  1.0 : Resistivity: deep", " deep"),
        ("GR.GAPI : GAMMA RAY", " GAMMA RAY"),
    ],
)
def test_description_is_the_text_after_the_last_colon(tmp_path: Path, curve_line: str, description: str):
    # D65 / CWLS LAS 2.0: a colon inside the description is not the delimiter.
    # A line with one colon is unchanged.
    body = (
        "~V\nVERS. 2.0 : version\nWRAP. NO : wrap\n"
        "~W\nSTRT.M 0.0 : START\nSTOP.M 1.0 : STOP\nSTEP.M 1.0 : STEP\n"
        "NULL. -999.25 : NULL\n"
        "~C\nDEPT.M : DEPTH\n"
        f"{curve_line}\n"
        "~A\n0.0 1.0\n1.0 2.0\n"
    )
    facts = extract_facts(_write_las(tmp_path, "last_colon.las", body))
    curve = facts["curves"][1]
    assert curve["description"] == description
    assert curve["mnemonic"] == curve_line.split(".", 1)[0].strip()


def test_curve_description_keeps_the_text_after_the_colon(tmp_path: Path):
    body = (
        "~V\nVERS. 2.0 : version text\nWRAP. NO :\n"
        "~W\nSTRT.FT 1000.0 : START DEPTH\nSTOP.FT 1000.5 : STOP DEPTH\n"
        "STEP.FT 0.5 : STEP\nNULL. -999 : NULL VALUE\n"
        "~C\nDEPT.FT : DEPTH\nGR  .GAPI : GAMMA RAY\n"
        "~A\n1000.0 80.0\n1000.5 90.0\n"
    )
    facts = extract_facts(_write_las(tmp_path, "verbatim.las", body))
    assert facts["vers"]["text"] == "2.0"
    assert facts["null"]["text"] == "-999"
    assert facts["null"]["number"] == -999.0
    assert facts["strt"]["unit"] == "FT"
    assert facts["strt"]["text"] == "1000.0"
    assert facts["curves"][1]["mnemonic"] == "GR"
    assert facts["curves"][1]["unit"] == "GAPI"
    assert facts["curves"][1]["description"] == " GAMMA RAY"
    assert facts["depth_step_count"] == 2


def test_missing_null_is_refused(tmp_path: Path):
    body = "~V\nVERS. 2.0 :\nWRAP. NO :\n~W\nSTRT.M 0 :\nSTOP.M 1 :\nSTEP.M 1 :\n~C\nDEPT.M : D\n~A\n0\n"
    with pytest.raises(FactsError, match="NULL"):
        extract_facts(_write_las(tmp_path, "no_null.las", body))


def test_round_trip_write_matches_canonical_json(tmp_path: Path):
    body = _HEADER.format(wrap="NO") + "0.0 10.0 2.5\n"
    fixture = _write_las(tmp_path, "one_step.las", body)
    written = write_fixture_facts(fixture, tmp_path)
    assert written.read_text(encoding="utf-8") == canonical_json(extract_facts(fixture))
