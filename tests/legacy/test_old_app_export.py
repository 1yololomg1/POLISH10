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


def _load(name: str):
    """Compile one class or function out of the old app. Do not import it."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            found = node
            break
    if found is None:
        raise AssertionError(f"{name} was not found in the old app")
    module = ast.Module(body=[found], type_ignores=[])
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
