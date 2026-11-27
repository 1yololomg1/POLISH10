"""Permanent pytest cases for the cubic quality-gate fixes.

These replace the session debug logs in advanced_preprocessing_system10.py.
They bind Application methods onto a Tk-free host except for the headless
load_data test, which needs the real constructor.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from advanced_preprocessing_system10 import AdvancedPreprocessingApplication
from core.curve_identification import CurveIdentificationEngine


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Root:
    def after(self, *args, **kwargs):
        return None


class QualityHost:
    detect_outliers_for_curve = AdvancedPreprocessingApplication.detect_outliers_for_curve
    apply_range_validation = AdvancedPreprocessingApplication.apply_range_validation
    apply_comprehensive_data_quality_validation = (
        AdvancedPreprocessingApplication.apply_comprehensive_data_quality_validation
    )
    _range_validation_skip_decision = AdvancedPreprocessingApplication._range_validation_skip_decision
    _record_range_validation_outcome = AdvancedPreprocessingApplication._record_range_validation_outcome
    _declared_curve_unit = AdvancedPreprocessingApplication._declared_curve_unit
    _typical_range_unit_compatible = AdvancedPreprocessingApplication._typical_range_unit_compatible
    _emit_range_validation_summary = AdvancedPreprocessingApplication._emit_range_validation_summary

    def log_processing(self, msg):
        self.logs.append(str(msg))


def make_host(unit="GAPI", curve_name="GR"):
    host = QualityHost()
    host.logs = []
    host.curve_identifier = CurveIdentificationEngine()
    host.outlier_detection_var = _Var(True)
    host.range_validation_var = _Var(True)
    host.range_validation_outcomes = []
    host.root = _Root()
    host.progress_bar = SimpleNamespace(configure=lambda **k: None)
    host.status_label = SimpleNamespace(config=lambda **k: None)
    if unit is None:
        host.curve_info = {}
    else:
        host.curve_info = {
            curve_name: {"curve_type": "GAMMA_RAY_TOTAL", "unit": unit},
        }
    if curve_name == "GR":
        host.curve_identifier.create_comprehensive_curve_info("GR", unit=unit or "")
    return host


def _skip_reasons(host):
    return [o for o in host.range_validation_outcomes if o.get("outcome") == "skipped"]


def test_gr_gapi_563_not_flagged_by_physical_bounds():
    host = make_host("GAPI")
    data = np.array([40.0, 80.0, 563.0], dtype=float)
    mask = host.detect_outliers_for_curve("GR", data)
    assert not bool(mask[2])
    applied = [o for o in host.range_validation_outcomes if "physical_bounds applied" in o.get("reason", "")]
    assert applied, host.range_validation_outcomes
    assert "GAPI" in applied[0]["reason"]


def test_gr_gapi_5000_flagged_by_physical_bounds():
    host = make_host("GAPI")
    data = np.array([40.0, 80.0, 5000.0], dtype=float)
    mask = host.detect_outliers_for_curve("GR", data)
    assert bool(mask[2])
    applied = [o for o in host.range_validation_outcomes if "physical_bounds applied" in o.get("reason", "")]
    assert applied and applied[0]["removed"] >= 1


def test_gr_cps_2500_not_flagged_and_skip_recorded():
    host = make_host("CPS")
    data = np.array([80.0, 2500.0, 40.0], dtype=float)
    mask = host.detect_outliers_for_curve("GR", data)
    assert not np.any(mask)
    skipped = _skip_reasons(host)
    assert skipped, host.range_validation_outcomes
    assert "CPS" in skipped[0]["reason"]
    assert skipped[0]["outcome"] == "skipped"
    host._emit_range_validation_summary()
    assert any("CPS" in line and "skipped" in line.lower() for line in host.logs) or any(
        "incompatible" in line for line in host.logs
    )


def test_gr_missing_unit_declines_physical_bounds_and_records_skip():
    host = make_host("")
    data = np.array([80.0, 5000.0, 40.0], dtype=float)
    mask = host.detect_outliers_for_curve("GR", data)
    assert not np.any(mask)
    skipped = _skip_reasons(host)
    assert skipped, host.range_validation_outcomes
    assert "missing" in skipped[0]["reason"]


def test_physical_bounds_flags_positive_and_negative_inf():
    host = make_host("GAPI")
    data = np.array([80.0, 200.0, np.inf, -np.inf, 40.0], dtype=float)
    mask = host.detect_outliers_for_curve("GR", data)
    assert bool(mask[2]) and bool(mask[3])


def test_physical_bounds_does_not_flag_nan():
    host = make_host("GAPI")
    data = np.array([80.0, np.nan, 40.0], dtype=float)
    mask = host.detect_outliers_for_curve("GR", data)
    assert not bool(mask[1])
    applied = [o for o in host.range_validation_outcomes if "physical_bounds applied" in o.get("reason", "")]
    assert applied
    assert applied[0]["removed"] == 0


def test_range_validation_converts_infinities_to_nan():
    host = make_host("GAPI")
    data = np.array([80.0, np.inf, -np.inf, np.nan], dtype=float)
    cleaned = host.apply_range_validation("GR", data)
    assert int(np.isinf(cleaned).sum()) == 0
    assert int(np.isnan(cleaned).sum()) == 3


def test_unknown_mnemonic_range_validation_skips_and_keeps_values():
    host = make_host("GAPI")
    host.curve_info["TBHV"] = {"curve_type": "UNKNOWN", "unit": "FT3"}
    data = np.array([10.0, 20.0, 30.0], dtype=float)
    out = host.apply_range_validation("TBHV", data)
    np.testing.assert_array_equal(out, data)
    skipped = [o for o in host.range_validation_outcomes if o.get("curve") == "TBHV"]
    assert skipped and skipped[0]["outcome"] == "skipped"
    assert skipped[0]["reason"]


def test_second_validation_pass_reports_only_its_own_curves():
    host = make_host("GAPI")
    host.range_validation_outcomes = [{"curve": "PRIOR", "outcome": "passed", "reason": "stale"}]
    host.apply_comprehensive_data_quality_validation(
        {
            "GR": np.array([40.0, 80.0, 120.0], dtype=float),
        }
    )
    curves = [o.get("curve") for o in host.range_validation_outcomes]
    assert "PRIOR" not in curves
    assert "GR" in curves


def test_headless_load_data_adopts_new_file_null(tmp_path):
    fixture = Path(__file__).resolve().parent / "fixtures" / "new_file_null_minus_999.las"
    assert fixture.is_file(), fixture
    app = AdvancedPreprocessingApplication()
    try:
        app.root.withdraw()
        app.well_datasets = {
            "OLD": {"well_info": {"null_value": "-999.25", "well_name": "OLD"}},
        }
        app.active_well_id = "OLD"
        app.null_value_var.set("-999.25")
        app.well_info = {"well_name": "OLD", "null_value": "-999.25"}
        app.load_data(str(fixture))
        assert app.null_value_var.get() == "-999"
    finally:
        app.root.destroy()
