"""Headless render model: bins, hit test, annotation. D74, D78, C7.2, C7.4."""

from __future__ import annotations

import numpy
import pytest

from polish.model.states import OutputInclusion, SampleStatus
from polish.render import annotate, bin_curve, hit_test

_M = SampleStatus.MEASURED
_A = SampleStatus.ABSENT
_I = SampleStatus.INVALID


def test_an_absent_sample_never_bins_to_zero():
    depths = [0.0, 1.0, 2.0, 3.0]
    values = [5.0, numpy.nan, numpy.nan, 7.0]
    bins = bin_curve(depths, values, (_M, _A, _A, _M), 0.0, 4.0, 4)
    assert bins.minimum[0] == 5.0 and bins.maximum[0] == 5.0
    assert numpy.isnan(bins.minimum[1]) and numpy.isnan(bins.maximum[1])
    assert numpy.isnan(bins.minimum[2]) and numpy.isnan(bins.maximum[2])
    assert list(bins.all_absent) == [False, True, True, False]
    assert list(bins.any_absent) == [False, True, True, False]
    assert not numpy.any(bins.minimum == 0.0)


def test_a_pixel_with_no_sample_is_not_absent_data():
    bins = bin_curve([0.0, 3.0], [1.0, 2.0], (_M, _M), 0.0, 4.0, 4)
    assert list(bins.sample_count) == [1, 0, 0, 1]
    assert list(bins.all_absent) == [False, False, False, False]
    assert numpy.isnan(bins.minimum[1])


def test_samples_are_placed_by_depth_not_by_row_index():
    depths = [0.0, 0.1, 0.2, 9.9]
    bins = bin_curve(depths, [1.0, 2.0, 3.0, 4.0], (_M, _M, _M, _M), 0.0, 10.0, 10)
    assert list(bins.sample_count) == [3, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert bins.minimum[0] == 1.0 and bins.maximum[0] == 3.0
    assert bins.minimum[9] == 4.0


def test_an_invalid_sample_flags_its_pixel_and_stays_out_of_the_measured_range():
    bins = bin_curve([0.0, 0.5, 1.5], [50.0, 0.002, 60.0], (_M, _I, _M), 0.0, 2.0, 2)
    assert list(bins.any_invalid) == [True, False]
    assert bins.minimum[0] == 50.0 and bins.maximum[0] == 50.0


def test_one_created_sample_among_measured_marks_the_pixel_modified():
    statuses = (_M,) * 30 + (SampleStatus.CREATED,) + (_M,) * 30
    depths = numpy.arange(61, dtype=float)
    bins = bin_curve(depths, numpy.ones(61), statuses, 0.0, 61.0, 1)
    assert list(bins.any_modified) == [True]
    assert list(bins.sample_count) == [61]


def test_the_window_includes_both_ends_and_excludes_the_rest():
    bins = bin_curve([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], (_M, _M, _M, _M), 1.0, 2.0, 1)
    assert list(bins.sample_count) == [2]
    assert bins.minimum[0] == 2.0 and bins.maximum[0] == 3.0


def test_bins_are_read_only_and_bad_input_is_refused():
    bins = bin_curve([0.0, 1.0], [1.0, 2.0], (_M, _M), 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        bins.minimum[0] = 0.0
    with pytest.raises(ValueError, match="ascending"):
        bin_curve([1.0, 0.0], [1.0, 2.0], (_M, _M), 0.0, 1.0, 2)
    with pytest.raises(ValueError, match="same length"):
        bin_curve([0.0, 1.0], [1.0], (_M, _M), 0.0, 1.0, 2)
    with pytest.raises(ValueError, match="greater than"):
        bin_curve([0.0, 1.0], [1.0, 2.0], (_M, _M), 1.0, 1.0, 2)
    with pytest.raises(ValueError, match="pixel_count"):
        bin_curve([0.0, 1.0], [1.0, 2.0], (_M, _M), 0.0, 1.0, 0)


def test_hit_test_returns_an_actual_sample_never_a_position_between():
    depths = [100.0, 100.5, 102.0]
    assert hit_test(depths, 100.2) == 0
    assert hit_test(depths, 100.3) == 1
    assert hit_test(depths, 100.25) == 0
    assert hit_test(depths, 101.9) == 2
    assert hit_test(depths, 50.0) == 0
    assert hit_test(depths, 500.0) == 2
    assert isinstance(hit_test(depths, 101.0), int)


def test_annotation_reports_no_value_for_an_absent_sample():
    depths = [10.0, 11.0]
    values = [numpy.nan, 0.002]
    absent = annotate(depths, values, 0, depth_unit="FT", unit="US/FT", status=_A)
    assert absent.value is None and absent.depth == 10.0 and absent.status is _A
    invalid = annotate(
        depths,
        values,
        1,
        depth_unit="FT",
        unit="US/FT",
        status=_I,
        output_inclusion=OutputInclusion.PENDING_DECISION,
        invalid_rules=("SONIC_COMPRESSIONAL",),
    )
    assert invalid.value == 0.002
    assert invalid.output_inclusion is OutputInclusion.PENDING_DECISION
    assert invalid.invalid_rules == ("SONIC_COMPRESSIONAL",)
    not_assessed = annotate(depths, values, 1, depth_unit="FT", unit="US/FT", status=None)
    assert not_assessed.status is None
    with pytest.raises(ValueError, match="not a sample"):
        annotate(depths, values, 2, depth_unit="FT", unit="US/FT", status=_M)
