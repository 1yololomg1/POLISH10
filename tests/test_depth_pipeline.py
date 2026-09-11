"""Depth-pipeline tests for the Step 3 switch and DEPT-copy changes.

STRUCTURE:
- _Var / _Root / DepthHost: Tk-free host binding Application and ReservoirDepthManager methods
- test_dept_well_export_columns_unchanged_after_depth_copy: item 1
Later items append tests to this module.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from advanced_preprocessing_system10 import (
    AdvancedPreprocessingApplication,
    ReservoirDepthManager,
)


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _Root:
    def after(self, *args, **kwargs):
        return None


def _las_curve_names(text: str) -> list[str]:
    names = []
    in_curve = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("~C"):
            in_curve = True
            continue
        if in_curve and stripped.startswith("~"):
            break
        if in_curve and stripped and not stripped.startswith("#"):
            names.append(stripped.split(".", 1)[0].strip())
    return names


class ExportHost:
    _generate_las_text_from_dataframe = (
        AdvancedPreprocessingApplication._generate_las_text_from_dataframe
    )
    _dataframe_with_uncertainty_bands = (
        AdvancedPreprocessingApplication._dataframe_with_uncertainty_bands
    )

    def handle_ui_error(self, *args, **kwargs):
        return None


def make_export_host():
    host = ExportHost()
    host.file_path_var = _Var("test.las")
    host.null_value_var = _Var("-999.25")
    host.processing_results = {}
    return host


def test_dept_well_export_columns_unchanged_after_depth_copy():
    """A well already named DEPT must export the same curves after the copy.

    The copy used to land in DEPTH_PRIMARY, which then appeared in LAS output.
    Writing under DEPT overwrites the same column, so the export set is unchanged.
    """
    original = pd.DataFrame(
        {
            "DEPT": [1000.0, 1000.5, 1001.0],
            "GR": [40.0, 50.0, 60.0],
        }
    )
    curve_info = {
        "DEPT": {"unit": "FT", "description": "DEPTH"},
        "GR": {"unit": "GAPI", "description": "GAMMA RAY"},
    }
    before_columns = list(original.columns)

    data = original.copy()
    manager = ReservoirDepthManager()
    selected, _metadata = manager.standardize_depth_reference(data, curve_info)
    assert selected == "DEPT"
    assert "DEPTH_PRIMARY" not in data.columns
    assert "DEPT" in data.columns

    host = make_export_host()
    export_df = host._dataframe_with_uncertainty_bands(data)
    las_text = host._generate_las_text_from_dataframe(export_df, curve_info, "-999.25")
    exported = _las_curve_names(las_text)
    assert exported == before_columns
    assert "DEPTH_PRIMARY" not in exported
