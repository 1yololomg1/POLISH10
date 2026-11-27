"""Display bins and sample lookup for one curve. D73, D74, D78, D3.

This is the part of P17 the interface shell needs: bins, hit testing, and the
hover annotation. Interval bands, the full provenance record, zoom thresholds
(OD-25), and the performance target are not built here.

Everything is positioned by depth, never by row index (C7.2). An absent
sample is never binned as zero (C7.4). A pixel that holds a created, changed,
or resampled sample is marked modified and is not averaged with measured
samples (D74). ``any_modified`` is the OR of ``any_created``, ``any_changed``
and ``any_resampled`` (P17.1, D74). At this milestone Stages 0-4 produce
MEASURED, INVALID, and ABSENT only, so those flags are false on real data.
Statuses are mapped to small integer codes once per curve; the masks are
numpy comparisons, not a Python loop per flag (P17.6).

Structure
---------
PixelBins
    Minimum and maximum of the measured samples in each pixel, plus flags.
bin_curve
    Build PixelBins for a depth window and a pixel count.
hit_test
    Index of the nearest actual sample. Never a position between samples.
SampleAnnotation, annotate
    Depth, value, unit, sample status, and output inclusion of one sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from polish.model.states import OutputInclusion, SampleStatus

# One integer per status, assigned once per curve. The masks below are numpy
# comparisons on this array (P17.6: no Python loop per flag).
_CODE = {
    SampleStatus.MEASURED: 0,
    SampleStatus.INVALID: 1,
    SampleStatus.ABSENT: 2,
    SampleStatus.CREATED: 3,
    SampleStatus.CHANGED: 4,
    SampleStatus.RESAMPLED: 5,
}
_MEASURED = _CODE[SampleStatus.MEASURED]
_INVALID = _CODE[SampleStatus.INVALID]
_ABSENT = _CODE[SampleStatus.ABSENT]
_CREATED = _CODE[SampleStatus.CREATED]
_CHANGED = _CODE[SampleStatus.CHANGED]
_RESAMPLED = _CODE[SampleStatus.RESAMPLED]


@dataclass(frozen=True, eq=False)
class PixelBins:
    """Per-pixel bins of one curve over one depth window. D74, UXC6.

    ``minimum`` and ``maximum`` cover MEASURED samples only and are NaN in a
    pixel that has none. ``sample_count`` is every sample in the pixel,
    whatever its status.     ``all_absent`` is true only for a pixel that holds
    samples and all of them are ABSENT. A pixel with no sample is not absent
    data; its count is 0. ``any_created``, ``any_changed`` and ``any_resampled``
    are separate (P17.1). ``any_modified`` is their OR (D74).
    """

    depth_top: float
    depth_bottom: float
    minimum: numpy.ndarray
    maximum: numpy.ndarray
    sample_count: numpy.ndarray
    any_invalid: numpy.ndarray
    any_absent: numpy.ndarray
    all_absent: numpy.ndarray
    any_created: numpy.ndarray
    any_changed: numpy.ndarray
    any_resampled: numpy.ndarray
    any_modified: numpy.ndarray


def _ascending_depths(depths: object) -> numpy.ndarray:
    array = numpy.asarray(depths, dtype=numpy.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("depths must be a one-dimensional array with at least one sample.")
    if not bool(numpy.all(numpy.isfinite(array))):
        raise ValueError("depths must be finite.")
    if array.size > 1 and not bool(numpy.all(numpy.diff(array) > 0)):
        raise ValueError("depths must be strictly ascending (D3).")
    return array


def bin_curve(
    depths: object,
    values: object,
    statuses: tuple[SampleStatus, ...],
    depth_top: float,
    depth_bottom: float,
    pixel_count: int,
) -> PixelBins:
    """Bin one curve into ``pixel_count`` pixels between two depths. D74, UXC6, D3.

    A sample belongs to the pixel its depth falls in, so row index plays no
    part (C7.2). Only MEASURED samples feed the minimum and maximum. An ABSENT
    sample sets flags and never contributes a number (C7.4).

    Parameters
    ----------
    depths:
        Strictly ascending depths, one per sample.
    values:
        Stored values, one per sample. NaN where the sample is absent.
    statuses:
        One SampleStatus per sample, as Stage 4 recorded it.
    depth_top, depth_bottom:
        The window. ``depth_bottom`` is greater than ``depth_top``. Both ends
        are inside the window.
    pixel_count:
        Number of pixels, at least 1.

    Raises
    ------
    ValueError
        The arrays differ in length, the window is empty, or the pixel count
        is below 1.
    """

    depth_array = _ascending_depths(depths)
    value_array = numpy.asarray(values, dtype=numpy.float64)
    if value_array.shape != depth_array.shape:
        raise ValueError("values and depths must have the same length.")
    if type(statuses) is not tuple or len(statuses) != depth_array.size:
        raise ValueError("statuses must be a tuple with one SampleStatus per sample.")
    if type(pixel_count) is not int or pixel_count < 1:
        raise ValueError("pixel_count must be an int of at least 1.")
    top = float(depth_top)
    bottom = float(depth_bottom)
    if not (numpy.isfinite(top) and numpy.isfinite(bottom)) or bottom <= top:
        raise ValueError("depth_bottom must be greater than depth_top.")

    codes = _status_codes(statuses)
    measured = codes == _MEASURED
    invalid = codes == _INVALID
    absent = codes == _ABSENT
    created = codes == _CREATED
    changed = codes == _CHANGED
    resampled = codes == _RESAMPLED

    inside = (depth_array >= top) & (depth_array <= bottom)
    position = (depth_array[inside] - top) / (bottom - top) * pixel_count
    pixel = numpy.minimum(position.astype(numpy.int64), pixel_count - 1)

    counts = numpy.bincount(pixel, minlength=pixel_count)
    absent_counts = numpy.bincount(pixel, weights=absent[inside], minlength=pixel_count)
    invalid_counts = numpy.bincount(pixel, weights=invalid[inside], minlength=pixel_count)
    created_counts = numpy.bincount(pixel, weights=created[inside], minlength=pixel_count)
    changed_counts = numpy.bincount(pixel, weights=changed[inside], minlength=pixel_count)
    resampled_counts = numpy.bincount(pixel, weights=resampled[inside], minlength=pixel_count)

    minimum = numpy.full(pixel_count, numpy.inf)
    maximum = numpy.full(pixel_count, -numpy.inf)
    usable = measured[inside] & numpy.isfinite(value_array[inside])
    numpy.minimum.at(minimum, pixel[usable], value_array[inside][usable])
    numpy.maximum.at(maximum, pixel[usable], value_array[inside][usable])
    empty = ~numpy.isfinite(minimum)
    minimum[empty] = numpy.nan
    maximum[empty] = numpy.nan

    arrays = (
        minimum,
        maximum,
        counts.astype(numpy.int64),
        invalid_counts > 0,
        absent_counts > 0,
        (counts > 0) & (absent_counts == counts),
        created_counts > 0,
        changed_counts > 0,
        resampled_counts > 0,
        (created_counts > 0) | (changed_counts > 0) | (resampled_counts > 0),
    )
    for array in arrays:
        array.flags.writeable = False
    return PixelBins(top, bottom, *arrays)


def _status_codes(statuses: tuple[SampleStatus, ...]) -> numpy.ndarray:
    """Map each sample status to a small integer, once per curve. P17.6, D74.

    Raises
    ------
    TypeError
        A member is not a SampleStatus. The check is this one pass, not a
        second walk of the samples.
    """

    code_of = _CODE

    def code(item: object) -> int:
        if isinstance(item, SampleStatus):
            return code_of[item]
        raise TypeError("statuses must contain SampleStatus values.")

    return numpy.fromiter((code(item) for item in statuses), dtype=numpy.int8, count=len(statuses))


def hit_test(depths: object, depth: float) -> int:
    """Index of the actual sample nearest to ``depth``. D78, D3.

    The result is always a sample that exists. A depth between two samples
    returns the nearer one, and the shallower one on an exact tie. A depth
    outside the data returns the first or the last sample.

    Parameters
    ----------
    depths:
        Strictly ascending depths, one per sample.
    depth:
        The depth under the pointer.
    """

    depth_array = _ascending_depths(depths)
    target = float(depth)
    if not numpy.isfinite(target):
        raise ValueError("depth must be finite.")
    right = int(numpy.searchsorted(depth_array, target, side="left"))
    if right <= 0:
        return 0
    if right >= depth_array.size:
        return int(depth_array.size - 1)
    left = right - 1
    if target - depth_array[left] <= depth_array[right] - target:
        return left
    return right


@dataclass(frozen=True)
class SampleAnnotation:
    """What a hover readout states about one sample. D78, D44.

    ``value`` is None for an ABSENT sample. ``status`` is None when Stage 4
    did not run, which is "not assessed", not MEASURED. The units are the
    text the file wrote. No conversion is applied here.
    """

    sample_index: int
    depth: float
    depth_unit: str
    value: float | None
    unit: str
    status: SampleStatus | None
    output_inclusion: OutputInclusion | None
    invalid_rules: tuple[str, ...]


def annotate(
    depths: object,
    values: object,
    sample_index: int,
    *,
    depth_unit: str,
    unit: str,
    status: SampleStatus | None,
    output_inclusion: OutputInclusion | None = None,
    invalid_rules: tuple[str, ...] = (),
) -> SampleAnnotation:
    """Annotation for one actual sample. D78, D44.

    The value is read from the stored array at ``sample_index``. A NaN is
    reported as no value, never as zero (C7.4).

    Parameters
    ----------
    depths, values:
        The dataset's depth grid and one curve's stored values.
    sample_index:
        An index ``hit_test`` returned.
    depth_unit, unit:
        Unit text as written in the file.
    status:
        The Stage 4 sample status, or None when Stage 4 did not run.
    output_inclusion, invalid_rules:
        As Stage 4 recorded them for this sample.
    """

    depth_array = numpy.asarray(depths, dtype=numpy.float64)
    value_array = numpy.asarray(values, dtype=numpy.float64)
    if type(sample_index) is not int or not 0 <= sample_index < depth_array.size:
        raise ValueError("sample_index is not a sample of this curve.")
    if value_array.shape != depth_array.shape:
        raise ValueError("values and depths must have the same length.")
    if status is not None and not isinstance(status, SampleStatus):
        raise TypeError("status must be a SampleStatus or None.")
    number = float(value_array[sample_index])
    return SampleAnnotation(
        sample_index=sample_index,
        depth=float(depth_array[sample_index]),
        depth_unit=depth_unit,
        value=None if numpy.isnan(number) else number,
        unit=unit,
        status=status,
        output_inclusion=output_inclusion,
        invalid_rules=tuple(invalid_rules),
    )
