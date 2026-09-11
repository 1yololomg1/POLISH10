"""Depth-pipeline tests for the Step 3 switch and DEPT-copy changes.

STRUCTURE:
- _Var / _Root / DepthHost: Tk-free host binding Application and ReservoirDepthManager methods
- test_dept_well_export_columns_unchanged_after_depth_copy: item 1
Later items append tests to this module.
"""

from __future__ import annotations

import inspect

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


def test_depth_standardization_precedes_resample_in_process_thread():
    """process_data_thread must finish depth copy before _uniformize_data resamples."""
    thread_src = inspect.getsource(AdvancedPreprocessingApplication.process_data_thread)
    assert thread_src.index("_validate_and_standardize_depth") < thread_src.index(
        "_uniformize_data"
    )
    uni_src = inspect.getsource(AdvancedPreprocessingApplication._uniformize_data)
    assert "resample_to_standard_spacing" in uni_src


class UniformizeHost:
    """Tk-free host for _uniformize_data / resample_to_standard_spacing."""

    _uniformize_data = AdvancedPreprocessingApplication._uniformize_data
    resample_to_standard_spacing = AdvancedPreprocessingApplication.resample_to_standard_spacing

    def uniformize_curves(self):
        return None

    def log_processing(self, msg):
        self.logs.append(str(msg))


class _Label:
    def config(self, **kwargs):
        return None


def test_md_well_has_dept_when_resampling_is_reached():
    """A well whose depth column is MD must already have DEPT at resample time."""
    data = pd.DataFrame(
        {
            "MD": [1000.0, 1000.5, 1001.0, 1001.5],
            "GR": [40.0, 50.0, 60.0, 70.0],
        }
    )
    curve_info = {
        "MD": {"unit": "FT", "description": "MEASURED DEPTH", "curve_type": "DEPTH"},
        "GR": {"unit": "GAPI", "description": "GAMMA RAY"},
    }
    manager = ReservoirDepthManager()
    selected, _metadata = manager.standardize_depth_reference(data, curve_info)
    assert selected == "MD"
    assert "MD" in data.columns

    seen = {}

    class RecordingHost(UniformizeHost):
        def resample_to_standard_spacing(self, depth_column, target_spacing):
            seen["columns"] = list(self.processed_data.columns)
            seen["depth_column"] = depth_column
            return AdvancedPreprocessingApplication.resample_to_standard_spacing(
                self, depth_column, target_spacing
            )

    host = RecordingHost()
    host.logs = []
    host.root = _Root()
    host.status_label = _Label()
    host.processed_data = data
    host.curve_info = curve_info
    host.rename_curves_var = _Var(True)
    host.standardize_units_var = _Var(False)
    host.depth_spacing_var = _Var(0.5)

    host._uniformize_data()
    assert seen, "resample_to_standard_spacing was not reached"
    assert "DEPT" in seen["columns"]
    assert seen["depth_column"] == "DEPT"
