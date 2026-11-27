"""In-memory well dataset. Source arrays are read-only.

Implements D43 as resolved by OD-03 (a child dataset per resample, structure
only), D17, C5, and D30. Numpy holds the values. This module does not resample
and does not read LAS.

Structure
---------
SourceFile
    Path, hash, size, and the time the caller says it was read.
Curve
    Verbatim mnemonic, unit, and description, plus a read-only float64 array.
DepthGrid
    Read-only depths, a required direction, an optional regularity, and an
    optional declared step.
LineageRecord
    The recorded parent link for a child dataset.
Dataset
    Identity, optional parent, source file, grid, curves, and lineage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy

from polish.model.ids import CurveRef, DatasetId, FileHash


class DepthDirection(Enum):
    """Direction of a depth grid. D3, D88. Not a sample or stage status."""

    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"


class DepthRegularity(Enum):
    """Whether the grid is regular. D67. Not a sample or stage status.

    The value is the caller's classification. Comparing it to the samples
    would need a tolerance, and no tolerance is approved, so this module
    does not recompute it. A grid may carry None instead of either value.
    None means the regularity has not been classified yet. It is not a
    third enum value.
    """

    REGULAR = "REGULAR"
    IRREGULAR = "IRREGULAR"


def _as_readonly_float64(values: object, what: str) -> numpy.ndarray:
    """Copy numeric values into a one-dimensional read-only float64 array.

    The copy is a new buffer. The caller's array is not locked and is not
    written. C5, D17.
    """

    array = numpy.asarray(values)
    if array.dtype == object or not numpy.issubdtype(array.dtype, numpy.number):
        raise TypeError(f"{what} must be numeric")
    owned = numpy.array(array, dtype=numpy.float64, copy=True)
    if owned.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional")
    if owned.size == 0:
        raise ValueError(f"{what} must contain at least one sample")
    owned.flags.writeable = False
    return owned


def _require_str(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str")
    return value


@dataclass(frozen=True, eq=False)
class SourceFile:
    """The input file a dataset was read from. D17, D46.

    Construction does not open the path. ``read`` opens it read-only.
    """

    path: Path
    file_hash: FileHash
    size_bytes: int
    read_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if not isinstance(self.file_hash, FileHash):
            raise TypeError("file_hash must be a FileHash")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be an int >= 0")
        if not isinstance(self.read_at, datetime):
            raise TypeError("read_at must be a datetime")

    @classmethod
    def read(cls, path: Path | str, read_at: datetime) -> SourceFile:
        """Hash and stat the file. The open mode is read-only. D17, C5."""

        file_path = Path(path)
        file_hash = FileHash.from_path(file_path)
        size_bytes = file_path.stat().st_size
        return cls(path=file_path, file_hash=file_hash, size_bytes=size_bytes, read_at=read_at)


@dataclass(frozen=True, eq=False)
class Curve:
    """One curve as read. original_* are stored verbatim. D17, C5, D30.

    ``values`` is float64 and ``flags.writeable`` is False. Assignment raises.
    """

    ref: CurveRef
    original_mnemonic: str
    original_unit: str
    original_description: str
    values: numpy.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.ref, CurveRef):
            raise TypeError("ref must be a CurveRef")
        mnemonic = _require_str(self.original_mnemonic, "original_mnemonic")
        unit = _require_str(self.original_unit, "original_unit")
        description = _require_str(self.original_description, "original_description")
        owned = _as_readonly_float64(self.values, "Curve values")
        object.__setattr__(self, "original_mnemonic", mnemonic)
        object.__setattr__(self, "original_unit", unit)
        object.__setattr__(self, "original_description", description)
        object.__setattr__(self, "values", owned)

    def __eq__(self, other: object) -> bool:
        if type(other) is not Curve:
            return NotImplemented
        return (
            self.ref == other.ref
            and self.original_mnemonic == other.original_mnemonic
            and self.original_unit == other.original_unit
            and self.original_description == other.original_description
            and numpy.array_equal(self.values, other.values, equal_nan=True)
        )


@dataclass(frozen=True, eq=False)
class DepthGrid:
    """Depth samples for one dataset. D3, D88, D67.

    Direction is required. It is checked by strict pairwise comparison, which
    needs no tolerance. ``regularity`` is stored as given. None means it has
    not been classified; Stage 1 owns that classification. ``declared_step``
    is the header step the caller supplies, including a negative step or zero.
    None means the header did not state a finite step. The step is not
    recomputed from the samples.
    """

    values: numpy.ndarray
    direction: DepthDirection
    regularity: DepthRegularity | None
    declared_step: float | None
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.direction, DepthDirection):
            raise TypeError("direction must be a DepthDirection")
        if self.regularity is not None and not isinstance(self.regularity, DepthRegularity):
            raise TypeError("regularity must be a DepthRegularity or None")
        if self.declared_step is None:
            step = None
        elif type(self.declared_step) is bool or not isinstance(self.declared_step, (int, float)):
            raise TypeError("declared_step must be a real number or None")
        else:
            step = float(self.declared_step)
            if not math.isfinite(step):
                raise TypeError("declared_step must be a finite real number or None")
        unit = _require_str(self.unit, "unit")
        owned = _as_readonly_float64(self.values, "Depth grid")
        if owned.size < 2:
            raise ValueError("Depth direction is not determined for fewer than 2 samples.")
        diffs = numpy.diff(owned)
        if self.direction is DepthDirection.ASCENDING and not bool(numpy.all(diffs > 0)):
            raise ValueError("ASCENDING depth grid is not strictly increasing.")
        if self.direction is DepthDirection.DESCENDING and not bool(numpy.all(diffs < 0)):
            raise ValueError("DESCENDING depth grid is not strictly decreasing.")
        object.__setattr__(self, "regularity", self.regularity)
        object.__setattr__(self, "declared_step", step)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "values", owned)

    def __eq__(self, other: object) -> bool:
        if type(other) is not DepthGrid:
            return NotImplemented
        return (
            self.direction is other.direction
            and self.regularity is other.regularity
            and self.declared_step == other.declared_step
            and self.unit == other.unit
            and numpy.array_equal(self.values, other.values, equal_nan=True)
        )


@dataclass(frozen=True)
class LineageRecord:
    """Recorded link from a child dataset to a parent. OD-03, D43.

    ``statement`` is the caller's description of the relation. This record
    does not resample and does not copy arrays.
    """

    parent_id: DatasetId
    statement: str

    def __post_init__(self) -> None:
        if not isinstance(self.parent_id, DatasetId):
            raise TypeError("parent_id must be a DatasetId")
        if not isinstance(self.statement, str) or self.statement.strip() == "":
            raise ValueError("A lineage record needs a non-empty statement.")


@dataclass(frozen=True, eq=False)
class Dataset:
    """One dataset on its own grid. OD-03, D43, C5.

    A source dataset has no parent and an empty lineage, and its id is the
    file hash. A child has a parent id, its own grid, and at least one
    lineage record. Construction stores the curves it is given. Each curve
    has already copied its values into a new read-only array, so a child
    cannot alias a parent's buffer.
    """

    id: DatasetId
    parent_id: DatasetId | None
    source_file: SourceFile
    grid: DepthGrid
    curves: tuple[Curve, ...]
    lineage: tuple[LineageRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, DatasetId):
            raise TypeError("id must be a DatasetId")
        if self.parent_id is not None and not isinstance(self.parent_id, DatasetId):
            raise TypeError("parent_id must be a DatasetId or None")
        if not isinstance(self.source_file, SourceFile):
            raise TypeError("source_file must be a SourceFile")
        if not isinstance(self.grid, DepthGrid):
            raise TypeError("grid must be a DepthGrid")
        if type(self.curves) is not tuple or not all(isinstance(curve, Curve) for curve in self.curves):
            raise TypeError("curves must be a tuple of Curve")
        if type(self.lineage) is not tuple or not all(isinstance(item, LineageRecord) for item in self.lineage):
            raise TypeError("lineage must be a tuple of LineageRecord")
        if len(self.curves) == 0:
            raise ValueError("A dataset has at least one curve.")
        self._check_identity()
        self._check_curves_match_grid()

    def _check_identity(self) -> None:
        if self.parent_id is None:
            if self.lineage:
                raise ValueError("A source dataset has no lineage records.")
            if self.id.parent_id is not None or self.id.token != self.source_file.file_hash.hex_digest:
                raise ValueError("A source dataset id must be the input file hash (D30, D46).")
            return
        if self.id.parent_id != self.parent_id:
            raise ValueError("A child dataset id must record the same parent.")
        if not self.lineage:
            raise ValueError("A child dataset needs a lineage record (OD-03).")
        if any(record.parent_id != self.parent_id for record in self.lineage):
            raise ValueError("Every lineage record must name this dataset's parent.")

    def _check_curves_match_grid(self) -> None:
        sample_count = int(self.grid.values.size)
        for position, curve in enumerate(self.curves):
            if curve.ref.dataset_id != self.id:
                raise ValueError("Curve ref dataset id does not match the dataset id.")
            if curve.ref.column_index != position:
                raise ValueError(
                    f"Curve column_index {curve.ref.column_index} is not position {position}."
                )
            if int(curve.values.size) != sample_count:
                raise ValueError("Curve length does not match the depth grid.")
            if curve.values.flags.writeable:
                raise ValueError("Curve values must be read-only.")
        if self.grid.values.flags.writeable:
            raise ValueError("Depth grid values must be read-only.")
