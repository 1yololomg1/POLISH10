"""Export behaviour of the old app, loaded from source.

Importing advanced_preprocessing_system10 installs global handlers and
selects the TkAgg backend, so these tests compile the three symbols they
need out of the file with ast and call them on a stand-in object.
Synthetic LAS only. The fixtures under tests/fixtures are not read.
"""

from __future__ import annotations

import ast
import os
import re
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import lasio
import numpy as np
import pandas as pd

_APP = Path(__file__).resolve().parents[2] / "advanced_preprocessing_system10.py"

_DEPTHS = [3500.0183, 3500.1707, 3500.3232]
_EXPONENT = re.compile(r"\d\.?\d*[eE][+-]?\d")

_SOURCE = """\
~Version
VERS. 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP. NO : ONE LINE PER DEPTH STEP
~Well
STRT.M 3500.0183 : START
STOP.M 3500.3232 : STOP
STEP.M 0.1524 : STEP
NULL. -999.25 : NULL VALUE
COMP. ACME OIL : COMPANY
WELL. {well} : WELL
UWI. {uwi} : UNIQUE WELL ID
DATE. 2024-03-15 : LOG DATE
~Curve
DEPT.M : Depth
GR.GAPI : Gamma Ray
RHOB. : Bulk Density
~Parameter
LOC. PAD 12 : LOCATION
~Ascii
3500.0183 45.5 2.45
3500.1707 -999.25 2.50
3500.3232 10000.25 2.55
"""


# The writer calls these. They are compiled with it so the app is still not imported.
_COMPANIONS = ("_parse_las_header_line", "_read_source_las_header")


def _load(name: str):
    """Compile one class or function out of the old app. Do not import it."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    found: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name not in found:
            found[node.name] = node
    if name not in found:
        raise AssertionError(f"{name} was not found in the old app")
    body = [found[item] for item in _COMPANIONS if item != name and item in found]
    body.append(found[name])
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Optional": Optional,
        "json": __import__("json"),
        "warnings": __import__("warnings"),
        "os": os,
        "np": np,
        "pd": pd,
        "lasio": lasio,
    }
    exec(compile(module, str(_APP), "exec"), namespace)
    return namespace[name]


def _write_source(directory: Path, well: str, filename: str, uwi: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(_SOURCE.format(well=well, uwi=uwi), encoding="utf-8", newline="\n")
    return path


def _frame() -> pd.DataFrame:
    # GR is first on purpose. The writer has to move the depth column first.
    return pd.DataFrame(
        {
            "GR": [45.5, np.nan, 10000.25],
            "DEPT": _DEPTHS,
            "RHOB": [2.45, 2.50, 2.55],
        }
    )


def _curve_info() -> dict:
    return {
        "DEPT": {"unit": "M", "description": "Depth"},
        "GR": {"unit": "GAPI", "description": "Gamma Ray"},
        "RHOB": {"unit": "", "description": "Bulk Density"},
    }


class _PathVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


def _stand_in(other_path: str) -> SimpleNamespace:
    """A self that would leak the wrong well if the writer still read it."""
    return SimpleNamespace(
        file_path_var=_PathVar(other_path),
        original_las_header="LEAKED HEADER",
        processing_results={"resampled_spacing": 9.9},
    )


def _export(source: Path, stand_in: SimpleNamespace, **kwargs: object) -> str:
    writer = _load("_generate_las_text_from_dataframe")
    return writer(
        stand_in,
        _frame(),
        _curve_info(),
        "-999.25",
        source_path=str(source),
        **kwargs,
    )


def test_output_path_accepts_a_new_file_and_refuses_the_rest(tmp_path: Path) -> None:
    handler = _load("SafeFileHandler")

    plain = tmp_path / "out"
    plain.mkdir()
    new_file = plain / "export.las"
    accepted = handler.validate_output_path(str(new_file))
    assert accepted == new_file.resolve()
    assert not new_file.exists()

    spaced = tmp_path / "a folder"
    spaced.mkdir()
    spaced_file = spaced / "new file.las"
    forward = str(spaced_file).replace("\\", "/")
    accepted_forward = handler.validate_output_path(forward)
    assert accepted_forward == spaced_file.resolve()
    assert not spaced_file.exists()

    missing_parent = tmp_path / "missing" / "export.las"
    assert handler.validate_output_path(str(missing_parent)) is None
    assert not missing_parent.parent.exists()

    existing_dir = tmp_path / "already"
    existing_dir.mkdir()
    assert handler.validate_output_path(str(existing_dir)) is None

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.las"
    escaped = allowed / ".." / "outside.las"
    assert handler.validate_output_path(str(escaped), allowed_dir=str(allowed)) is None
    assert not outside.exists()

    inside = allowed / "kept.las"
    assert handler.validate_output_path(str(inside), allowed_dir=str(allowed)) == inside.resolve()
    assert not inside.exists()

    assert handler.validate_file_path(str(tmp_path / "absent.las")) is None


def test_writer_round_trips_units_header_and_values(tmp_path: Path) -> None:
    folder = tmp_path / "dir with space" / "nested"
    source = _write_source(folder, "ALPHA-1", "well one.las", "100123456789")
    # Point self at a different existing file. The export must ignore it.
    decoy = _write_source(folder, "DECOY", "decoy.las", "000")
    text = _export(source, _stand_in(str(decoy)))

    assert "dir with space" not in text
    assert "nested" not in text
    assert "decoy.las" not in text
    assert "LEAKED HEADER" not in text
    assert "well one.las" in text
    assert _EXPONENT.search(text) is None
    assert "10000.25" in text
    assert not any(line.startswith("FILE.") for line in text.splitlines())

    out = tmp_path / "alpha.las"
    out.write_text(text, encoding="utf-8", newline="\n")
    loaded = lasio.read(str(out))

    assert loaded.well.WELL.value == "ALPHA-1"
    assert loaded.well.COMP.value == "ACME OIL"
    assert loaded.well.UWI.value == "100123456789"
    assert loaded.well.DATE.value == "2024-03-15"
    assert loaded.params.LOC.value == "PAD 12"
    assert loaded.params.LOC.descr == "LOCATION"
    assert float(loaded.well.STRT.value) == _DEPTHS[0]
    assert float(loaded.well.STOP.value) == _DEPTHS[-1]
    assert float(loaded.well.STEP.value) == 0.1524
    assert float(loaded.well.NULL.value) == -999.25
    assert loaded.well.STRT.unit == "M"

    assert [(curve.mnemonic, curve.unit) for curve in loaded.curves] == [
        ("DEPT", "M"),
        ("GR", "GAPI"),
        ("RHOB", ""),
    ]
    assert np.array_equal(np.asarray(loaded["DEPT"], dtype=float), np.asarray(_DEPTHS))
    assert np.array_equal(
        np.asarray(loaded["GR"], dtype=float),
        np.asarray([45.5, np.nan, 10000.25]),
        equal_nan=True,
    )
    assert np.array_equal(np.asarray(loaded["RHOB"], dtype=float), np.asarray([2.45, 2.5, 2.55]))


def test_two_wells_keep_their_own_names(tmp_path: Path) -> None:
    folder = tmp_path / "batch folder"
    alpha = _write_source(folder, "ALPHA-1", "alpha.las", "111")
    beta = _write_source(folder, "BETA-2", "beta.las", "222")
    alpha_text = _export(alpha, _stand_in(str(beta)))
    beta_text = _export(beta, _stand_in(str(alpha)))

    alpha_out = tmp_path / "alpha_out.las"
    beta_out = tmp_path / "beta_out.las"
    alpha_out.write_text(alpha_text, encoding="utf-8", newline="\n")
    beta_out.write_text(beta_text, encoding="utf-8", newline="\n")

    assert lasio.read(str(alpha_out)).well.WELL.value == "ALPHA-1"
    assert lasio.read(str(beta_out)).well.WELL.value == "BETA-2"
    assert "batch folder" not in alpha_text
    assert "batch folder" not in beta_text
    assert "BETA-2" not in alpha_text
    assert "ALPHA-1" not in beta_text


def test_missing_source_still_writes_the_depth_block(tmp_path: Path) -> None:
    missing = tmp_path / "gone folder" / "missing.las"
    text = _export(missing, _stand_in(str(missing)))
    assert "gone folder" not in text
    out = tmp_path / "missing_out.las"
    out.write_text(text, encoding="utf-8", newline="\n")
    loaded = lasio.read(str(out))
    assert float(loaded.well.STRT.value) == _DEPTHS[0]
    assert float(loaded.well.STOP.value) == _DEPTHS[-1]
    assert float(loaded.well.STEP.value) == 0.0
    assert float(loaded.well.NULL.value) == -999.25


def test_resampled_spacing_overrides_the_header_step(tmp_path: Path) -> None:
    source = tmp_path / "feet.las"
    source.write_text(
        "\n".join(
            [
                "~Version",
                "VERS. 2.0 : v",
                "WRAP. NO : w",
                "~Well",
                "STRT.FT 1000 : START",
                "STOP.FT 1001 : STOP",
                "STEP.FT 0.5 : STEP",
                "NULL. -999.25 : NULL",
                "WELL. FEETWELL : WELL",
                "~Curve",
                "DEPT.FT : Depth",
                "GR.GAPI : Gamma",
                "~Ascii",
                "1000 1",
                "1000.5 2",
                "1001 3",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    frame = pd.DataFrame({"DEPT": [1000.0, 1000.5, 1001.0], "GR": [1.0, 2.0, 3.0]})
    info = {
        "DEPT": {"unit": "FT", "description": "Depth"},
        "GR": {"unit": "GAPI", "description": "Gamma"},
    }
    writer = _load("_generate_las_text_from_dataframe")
    text = writer(
        _stand_in(str(source)),
        frame,
        info,
        "-999.25",
        source_path=str(source),
        resampled_spacing=0.25,
    )
    out = tmp_path / "feet_out.las"
    out.write_text(text, encoding="utf-8", newline="\n")
    loaded = lasio.read(str(out))
    assert float(loaded.well.STEP.value) == 0.25
    assert loaded.well.STEP.unit == "FT"
    assert loaded.well.WELL.value == "FEETWELL"


def test_resampled_grid_stays_on_the_step() -> None:
    """159-19A's depth range. The count and last depth are the spec, not a fixture."""
    resample = _load("resample_to_standard_spacing")
    frame = pd.DataFrame({"DEPT": [3500.0183, 4094.9878]})
    stand = SimpleNamespace(
        processed_data=frame,
        processing_results={},
        log_processing=lambda _message: None,
    )
    resample(stand, "DEPT", 0.1524)
    depths = stand.processed_data["DEPT"].to_numpy()
    assert len(depths) == 3905
    assert depths[-1] == 4094.9879
    assert stand.processing_results["resampled_spacing"] == 0.1524


def test_gap_total_skips_the_spacing_record() -> None:
    count_gaps = _load("_calculate_total_gaps_filled")
    stand = SimpleNamespace(
        processing_results={
            "resampled_spacing": 0.1524,
            "GR": {"gap_filling": {"quality_metrics": {"total_points_filled": 4}}},
        }
    )
    assert count_gaps(stand) == 4


_LATIN = """\
~Version
VERS. 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP. NO : ONE LINE PER DEPTH STEP
~Well
# Nota de campo
STRT.M 3500.0183 : START DEPTH
STOP.M 3500.3232 : STOP DEPTH
STEP.M 0.1524 : STEP
NULL. -999.25 : NULL VALUE
WELL. POZO ÁLAMO-1 : WELL
FLD . 007 : FIELD
LIC . 0012345 : LICENCE
DATE. 2024-03-15 10:30:00 : LOG DATE
~Curve
DEPT.M : Depth
GR.GAPI : Gamma Ray
~Parameter
BHT.DEGC 085.50 : temp
~A
3500.0183 45.5
3500.1707 -999.25
3500.3232 10000.25
"""

_KEPT_LATIN = (
    "# Nota de campo",
    "WELL. POZO ÁLAMO-1 : WELL",
    "FLD . 007 : FIELD",
    "LIC . 0012345 : LICENCE",
    "DATE. 2024-03-15 10:30:00 : LOG DATE",
    "BHT.DEGC 085.50 : temp",
)


def _write_las(path: Path, body: str, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body.encode(encoding))
    return path


def _las_text(well_lines: list[str], curve_lines: list[str]) -> str:
    return "\n".join(
        [
            "~Version",
            "VERS. 2.0 : v",
            "WRAP. NO : w",
            "~Well",
            *well_lines,
            "~Curve",
            *curve_lines,
            "~A",
            "0 0",
            "",
        ]
    )


def _emit(
    directory: Path,
    filename: str,
    body: str,
    frame: pd.DataFrame,
    info: dict,
    **kwargs: object,
) -> str:
    path = _write_las(directory / filename, body)
    writer = _load("_generate_las_text_from_dataframe")
    return writer(
        _stand_in(str(path)),
        frame,
        info,
        "-999.25",
        source_path=str(path),
        **kwargs,
    )


def _mnemonic_lines(text: str, mnemonic: str) -> list[str]:
    found = []
    for line in text.splitlines():
        parsed = _load("_parse_las_header_line")(line)
        if parsed is not None and parsed[0] == mnemonic:
            found.append(line)
    return found


def test_verbatim_header_and_latin1_encoding(tmp_path: Path) -> None:
    source = _write_las(tmp_path / "pozo.las", _LATIN, encoding="latin-1")
    sections, encoding = _load("_read_source_las_header")(str(source))
    assert encoding == "latin-1"
    assert sections is not None
    assert "# Nota de campo" in sections["W"]
    assert "FLD . 007 : FIELD" in sections["W"]
    assert "BHT.DEGC 085.50 : temp" in sections["P"]

    text = _export(source, _stand_in(str(tmp_path / "other.las")))
    written = text.splitlines()
    for line in _KEPT_LATIN:
        assert line in written
    assert "STEP.M 0.1524 : STEP" in written
    assert "FLD. 7" not in text
    assert "LIC. 12345" not in text


def test_feet_step_converts_when_depth_is_exported_in_metres(tmp_path: Path) -> None:
    body = _las_text(
        [
            "STRT.FT 1000 : START",
            "STOP.FT 1001 : STOP",
            "STEP.FT 0.5 : STEP",
            "NULL. -999.25 : NULL",
            "WELL. FEETWELL : WELL",
        ],
        ["DEPT.FT : Depth", "GR.GAPI : Gamma"],
    )
    feet = [Decimal("1000"), Decimal("1000.5"), Decimal("1001")]
    metres = [float(item * Decimal("0.3048")) for item in feet]
    frame = pd.DataFrame({"DEPT": metres, "GR": [1.0, 2.0, 3.0]})
    info = {
        "DEPT": {"unit": "M", "description": "Depth"},
        "GR": {"unit": "GAPI", "description": "Gamma"},
    }
    text = _emit(tmp_path, "feet.las", body, frame, info)
    assert "STEP.M 0.1524 : STEP" in text.splitlines()


def test_step_in_feet_is_kept_when_the_export_stays_in_feet(tmp_path: Path) -> None:
    body = _las_text(
        [
            "STRT.F 100 : START",
            "STEP.F 0.5 : STEP",
            "NULL. -999.25 : NULL",
        ],
        ["DEPT.F : Depth"],
    )
    frame = pd.DataFrame({"DEPT": [100.0, 101.0, 103.0]})
    info = {"DEPT": {"unit": "F", "description": "Depth"}}
    text = _emit(tmp_path, "unconverted.las", body, frame, info)
    assert "STEP.F 0.5 : STEP" in text.splitlines()


def test_mismatched_step_unit_falls_through_to_zero(tmp_path: Path) -> None:
    body = _las_text(
        [
            "STRT.M 100 : START",
            "STEP.F 0.5 : STEP",
            "NULL. -999.25 : NULL",
        ],
        ["DEPT.M : Depth"],
    )
    frame = pd.DataFrame({"DEPT": [100.0, 100.5, 101.1]})
    info = {"DEPT": {"unit": "M", "description": "Depth"}}
    text = _emit(tmp_path, "mismatch.las", body, frame, info)
    assert "STEP.M 0 : STEP" in text.splitlines()


def test_step_and_strt_without_units_give_zero(tmp_path: Path) -> None:
    body = _las_text(
        [
            "STRT. 10 : START",
            "STEP. 0.5 : STEP",
            "NULL. -999.25 : NULL",
        ],
        ["DEPT. : Depth"],
    )
    frame = pd.DataFrame({"DEPT": [10.0, 10.2, 10.7]})
    info = {"DEPT": {"unit": "", "description": "Depth"}}
    text = _emit(tmp_path, "nounit.las", body, frame, info)
    assert "STEP. 0 : STEP" in text.splitlines()


def test_header_step_of_zero_uses_the_equal_gap(tmp_path: Path) -> None:
    body = _las_text(
        [
            "STRT.M 100 : START",
            "STEP.M 0 : STEP",
            "NULL. -999.25 : NULL",
        ],
        ["DEPT.M : Depth"],
    )
    frame = pd.DataFrame({"DEPT": [100.0, 100.25, 100.5]})
    info = {"DEPT": {"unit": "M", "description": "Depth"}}
    text = _emit(tmp_path, "zero.las", body, frame, info)
    assert "STEP.M 0.25 : STEP" in text.splitlines()


def test_decreasing_depth_writes_a_negative_step(tmp_path: Path) -> None:
    body = _las_text(
        [
            "STRT.M 100 : START",
            "STEP.M 0.5 : STEP",
            "NULL. -999.25 : NULL",
        ],
        ["DEPT.M : Depth"],
    )
    frame = pd.DataFrame({"DEPT": [100.0, 99.5, 99.0]})
    info = {"DEPT": {"unit": "M", "description": "Depth"}}
    text = _emit(tmp_path, "down.las", body, frame, info)
    written = text.splitlines()
    assert "STEP.M -0.5 : STEP" in written
    assert "STRT.M 100 : START DEPTH" in written
    assert "STOP.M 99 : STOP DEPTH" in written


def test_preview_stop_is_the_last_depth_of_the_whole_column(tmp_path: Path) -> None:
    depths = [float(Decimal("100") + Decimal("0.5") * index) for index in range(10)]
    body = _las_text(
        [
            "STRT.M 100 : START",
            "STOP.M 1 : STOP",
            "STEP.M 0.5 : STEP",
            "NULL. -999.25 : NULL",
        ],
        ["DEPT.M : Depth"],
    )
    frame = pd.DataFrame({"DEPT": depths})
    info = {"DEPT": {"unit": "M", "description": "Depth"}}
    text = _emit(tmp_path, "preview.las", body, frame, info, max_rows=3)
    written = text.splitlines()
    assert "STOP.M 104.5 : STOP DEPTH" in written
    data = [line for line in written[written.index("~Ascii") + 1 :] if line.strip()]
    assert len(data) == 3
    assert data[-1].split()[0] != "104.5"


def test_non_las_sources_carry_no_header(tmp_path: Path) -> None:
    reader = _load("_read_source_las_header")
    csv_path = tmp_path / "curves.csv"
    csv_path.write_bytes(b"~Well\nFLD . 007 : FIELD\n")
    dlis_path = tmp_path / "huge.dlis"
    dlis_path.write_bytes(b"~Well\nFLD . 007 : FIELD\n")
    for path in (csv_path, dlis_path):
        assert reader(str(path)) == (None, None)
    assert reader("") == (None, None)
    assert reader(None) == (None, None)

    writer = _load("_generate_las_text_from_dataframe")
    for path in (csv_path, dlis_path):
        text = writer(
            _stand_in(str(path)),
            _frame(),
            _curve_info(),
            "-999.25",
            source_path=str(path),
        )
        assert "FLD . 007" not in text
        assert "STRT.M" in text


def test_duplicate_strt_is_written_once_and_a_missing_null_is_added(tmp_path: Path) -> None:
    body = _las_text(
        [
            "STRT.M 100 : START",
            "STRT.M 200 : START AGAIN",
            "STOP.M 101 : STOP",
            "STEP.M 1 : STEP",
            "WELL. DUP : WELL",
        ],
        ["DEPT.M : Depth"],
    )
    frame = pd.DataFrame({"DEPT": [100.0, 101.0]})
    info = {"DEPT": {"unit": "M", "description": "Depth"}}
    text = _emit(tmp_path, "dup.las", body, frame, info)
    assert _mnemonic_lines(text, "STRT") == ["STRT.M 100 : START DEPTH"]
    assert _mnemonic_lines(text, "NULL") == ["NULL. -999.25 : NULL VALUE"]


def test_null_has_no_unit_and_version_has_no_dlm(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "ALPHA-1", "well.las", "100123456789")
    text = _export(source, _stand_in(str(source)))
    assert _mnemonic_lines(text, "NULL") == ["NULL. -999.25 : NULL VALUE"]
    version = text.split("~Well", 1)[0]
    assert "DLM" not in version
    assert not any(line.strip().upper().startswith("DLM") for line in text.splitlines())
