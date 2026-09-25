"""Sparse overlay on a dataset. The source arrays are not written.

Implements D43 for fills, derivations, and smoothing. Resampling is not an
overlay (OD-03). ``sample_status`` is ``states.SampleStatus``.

Reading with the overlay returns a new array. Reading original only returns
the source array object, still read-only.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy

from polish.model.dataset import Curve
from polish.model.ids import SampleRef
from polish.model.states import SampleStatus


def _nonempty(text: str) -> bool:
    return isinstance(text, str) and text.strip() != ""


@dataclass(frozen=True)
class OverlayEntry:
    """One replaced sample. D44, D43.

    ``provenance_record_id`` is required. A created or changed sample without
    provenance is not representable here.
    """

    new_value: float
    sample_status: SampleStatus
    provenance_record_id: str

    def __post_init__(self) -> None:
        if isinstance(self.new_value, bool) or not isinstance(self.new_value, numbers.Real):
            raise TypeError("new_value must be a real number")
        if not isinstance(self.sample_status, SampleStatus):
            raise TypeError("sample_status must be a SampleStatus")
        if not _nonempty(self.provenance_record_id):
            raise ValueError("An overlay entry needs a provenance_record_id.")
        object.__setattr__(self, "new_value", float(self.new_value))


class Overlay:
    """Sparse map from SampleRef to OverlayEntry. D43, C5.

    A second put at the same SampleRef is refused, so an entry is not
    silently replaced.
    """

    def __init__(self) -> None:
        self._entries: dict[SampleRef, OverlayEntry] = {}

    def put(
        self,
        sample_ref: SampleRef,
        new_value: float,
        sample_status: SampleStatus,
        provenance_record_id: str,
    ) -> None:
        """Record one overlay sample. Does not write the source array."""

        if not isinstance(sample_ref, SampleRef):
            raise TypeError("sample_ref must be a SampleRef")
        if sample_ref in self._entries:
            raise ValueError(f"Overlay already has an entry for {sample_ref}.")
        self._entries[sample_ref] = OverlayEntry(new_value, sample_status, provenance_record_id)

    def snapshot(self) -> tuple[tuple[str, float, str, str], ...]:
        """Stable copy of the entries, for a contract check. Not a source array."""

        rows = [
            (str(ref), entry.new_value, entry.sample_status.value, entry.provenance_record_id)
            for ref, entry in self._entries.items()
        ]
        rows.sort()
        return tuple(rows)

    def read_original(self, curve: Curve) -> numpy.ndarray:
        """Return the source array itself. C5. The overlay is not applied."""

        if not isinstance(curve, Curve):
            raise TypeError("curve must be a Curve")
        return curve.values

    def read_with_overlay(self, curve: Curve) -> numpy.ndarray:
        """Return a new array with overlay entries applied. The source is not written. D43."""

        if not isinstance(curve, Curve):
            raise TypeError("curve must be a Curve")
        output = numpy.array(curve.values, dtype=numpy.float64, copy=True)
        for sample_ref, entry in self._entries.items():
            if sample_ref.curve_ref != curve.ref:
                continue
            if sample_ref.sample_index >= int(output.size):
                raise ValueError(f"Overlay sample index is outside {curve.ref}.")
            output[sample_ref.sample_index] = entry.new_value
        return output
