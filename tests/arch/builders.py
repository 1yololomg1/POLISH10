"""Small in-memory datasets for architecture tests. Not a LAS reader."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy

from polish.model.dataset import (
    Curve,
    Dataset,
    DepthDirection,
    DepthGrid,
    DepthRegularity,
    LineageRecord,
    SourceFile,
)
from polish.model.ids import CurveRef, DatasetId
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, Runner, StageOutcome
from polish.model.states import StageStatus


def source_dataset(path: Path, mnemonics: tuple[str, ...] = ("GR", "GR")) -> Dataset:
    """A source dataset whose file identity is ``path``. Values are not parsed from it.

    Two curves share a mnemonic on purpose: the mnemonic is not the identity.
    """

    source = SourceFile.read(path, datetime(2026, 9, 24, tzinfo=timezone.utc))
    dataset_id = DatasetId.for_source(source.file_hash)
    depths = numpy.array([1000.0, 1000.5, 1001.0], dtype=numpy.float64)
    grid = DepthGrid(depths, DepthDirection.ASCENDING, DepthRegularity.REGULAR, 0.5, "FT")
    curves = tuple(
        Curve(CurveRef(dataset_id, index), mnemonic, " GAPI", " gamma ray", depths + index)
        for index, mnemonic in enumerate(mnemonics)
    )
    return Dataset(dataset_id, None, source, grid, curves, ())


def child_dataset(parent: Dataset, token: str = "child-1") -> Dataset:
    """A child with its own grid and a lineage record. Arrays are new buffers. OD-03."""

    child_id = DatasetId.for_child(parent.id, token)
    depths = numpy.array([1000.0, 1001.0], dtype=numpy.float64)
    grid = DepthGrid(depths, DepthDirection.ASCENDING, DepthRegularity.REGULAR, 1.0, "FT")
    parent_curve = parent.curves[0]
    curve = Curve(
        CurveRef(child_id, 0),
        parent_curve.original_mnemonic,
        parent_curve.original_unit,
        parent_curve.original_description,
        numpy.array([1.0, 2.0], dtype=numpy.float64),
    )
    lineage = (LineageRecord(parent.id, "Child grid recorded. Resampling is not performed here."),)
    return Dataset(child_id, parent.id, parent.source_file, grid, (curve,), lineage)


def stage_contract(
    stage_number: int,
    *,
    alters_data: bool = False,
    depends_on: tuple[int, ...] = (),
    name: str | None = None,
) -> StageContract:
    return StageContract(
        stage_number=stage_number,
        name=name or f"stage {stage_number}",
        requires=StageRequires(
            units="any if declared",
            completeness="may see absent samples; may not see synthesised values",
            preconditions=(),
        ),
        guarantees=StageGuarantees(
            row_count="unchanged",
            units="unchanged",
            validity="mask unchanged",
        ),
        declines_when=(),
        depends_on_stages=depends_on,
        alters_data=alters_data,
    )


def noop_stage(stage_number: int = 0, depends_on: tuple[int, ...] = ()) -> RegisteredStage:
    def run(_context: object) -> StageOutcome:
        return StageOutcome(StageStatus.COMPLETE, ())

    def unmet(_context: object) -> tuple[str, ...]:
        return ()

    return RegisteredStage(stage_contract(stage_number, depends_on=depends_on), run, unmet)


def run_noop(dataset: Dataset) -> None:
    Runner((noop_stage(),)).run(dataset)
