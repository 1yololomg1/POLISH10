"""Identity, dataset, overlay, and export projection. D30, D43, D46, D79."""

from datetime import datetime, timezone
from pathlib import Path

import numpy
import pytest

from polish.model.dataset import DepthDirection, DepthGrid, DepthRegularity
from polish.model.findings import (
    DecisionActor,
    DecisionRecord,
    DecisionScope,
    Evidence,
    FindingRecord,
    evidence_digest,
    finding_key,
    project_for_export,
)
from polish.model.ids import DatasetId, FileHash
from polish.model.states import (
    BlockType,
    Disposition,
    IssueDisposition,
    UserAction,
    export_permission,
)
from tests.arch.builders import child_dataset, source_dataset

_ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_file_hash_streams_the_bytes(tmp_path: Path):
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    hashed = FileHash.from_path(path)
    assert hashed.hex_digest == _ABC_SHA256
    assert str(hashed) == _ABC_SHA256
    assert FileHash(hashed.hex_digest) == hashed


def test_dataset_id_round_trips_for_source_and_child(tmp_path: Path):
    path = tmp_path / "in.las"
    path.write_bytes(b"abc")
    dataset = source_dataset(path)
    source_id = dataset.id
    assert source_id == DatasetId.for_source(dataset.source_file.file_hash)
    assert DatasetId.parse(source_id.serialise()) == source_id
    child_id = DatasetId.for_child(source_id, "step: 0.5 / pass")
    assert DatasetId.parse(child_id.serialise()) == child_id
    assert child_id.parent_id == source_id
    assert str(dataset.curves[0]) == str(dataset.curves[1]) or dataset.curves[0].ref != dataset.curves[1].ref
    assert dataset.curves[0].original_mnemonic == dataset.curves[1].original_mnemonic
    assert dataset.curves[0].ref.column_index == 0
    assert dataset.curves[1].ref.column_index == 1


def test_verbatim_header_text_is_not_stripped(tmp_path: Path):
    path = tmp_path / "in.las"
    path.write_bytes(b"abc")
    dataset = source_dataset(path)
    curve = dataset.curves[0]
    assert curve.original_mnemonic == "GR"
    assert curve.original_unit == " GAPI"
    assert curve.original_description == " gamma ray"


def test_descending_grid_accepts_a_negative_declared_step():
    grid = DepthGrid(
        numpy.array([10.0, 9.5], dtype=numpy.float64),
        DepthDirection.DESCENDING,
        DepthRegularity.IRREGULAR,
        -0.5,
        "M",
    )
    assert grid.direction is DepthDirection.DESCENDING
    assert grid.declared_step == -0.5
    assert grid.values.flags.writeable is False


def test_unclassified_step_and_regularity_are_stored_as_none():
    # None means the header stated no finite step, and regularity is not classified.
    grid = DepthGrid(
        numpy.array([0.0, 1.0], dtype=numpy.float64),
        DepthDirection.ASCENDING,
        None,
        None,
        "M",
    )
    assert grid.regularity is None
    assert grid.declared_step is None
    assert grid.direction is DepthDirection.ASCENDING


def test_non_finite_declared_step_is_refused():
    with pytest.raises(TypeError, match="finite"):
        DepthGrid(
            numpy.array([0.0, 1.0], dtype=numpy.float64),
            DepthDirection.ASCENDING,
            None,
            float("nan"),
            "M",
        )


def test_non_monotonic_depth_is_refused():
    with pytest.raises(ValueError, match="strictly increasing"):
        DepthGrid(
            numpy.array([1.0, 2.0, 2.0], dtype=numpy.float64),
            DepthDirection.ASCENDING,
            DepthRegularity.IRREGULAR,
            0.0,
            "M",
        )


def test_child_keeps_its_own_grid(tmp_path: Path):
    path = tmp_path / "in.las"
    path.write_bytes(b"abc")
    parent = source_dataset(path)
    child = child_dataset(parent)
    assert child.id.parent_id == parent.id
    assert child.grid.values.size != parent.grid.values.size
    assert child.lineage[0].statement.startswith("Child grid recorded")


def test_projection_feeds_the_existing_export_gate():
    created_at = datetime(2026, 9, 24, tzinfo=timezone.utc)
    message = "Recorded difference in the unit text."
    finding = FindingRecord(
        id=finding_key(2, "mnemonic_left_unchanged", ()),
        stage=2,
        check="mnemonic_left_unchanged",
        identity=(),
        block_type=BlockType.INFORMATIONAL,
        subject_refs=(),
        evidence=Evidence(BlockType.INFORMATIONAL.value),
        message=message,
        created_at=created_at,
    )
    open_finding = project_for_export(finding, None)
    assert open_finding.disposition.issue_disposition is IssueDisposition.UNDISPOSITIONED
    blocked = export_permission((open_finding,), ())
    assert blocked.allowed is False
    assert finding.id in blocked.reasons[0]
    decision = DecisionRecord(
        "dec-1",
        finding.id,
        Disposition(IssueDisposition.ACCEPTED),
        UserAction.ACKNOWLEDGE,
        None,
        DecisionScope.THIS_ITEM,
        DecisionActor.USER,
        created_at,
        "0.0.0",
        (("families", "1"),),
        FileHash("a" * 64),
        None,
        evidence_digest(finding),
    )
    accepted = project_for_export(finding, decision)
    assert export_permission((accepted,), ()).allowed is True


def test_undeclared_evidence_key_is_refused():
    with pytest.raises(ValueError, match="not declared"):
        Evidence(BlockType.INFORMATIONAL.value, (("guess", "value"),))


def test_decision_for_another_finding_is_refused():
    created_at = datetime(2026, 9, 24, tzinfo=timezone.utc)
    finding = FindingRecord(
        id=finding_key(0, "depth_no_curve", ()),
        stage=0,
        check="depth_no_curve",
        identity=(),
        block_type=BlockType.STRUCTURAL_DATA_FAILURE,
        subject_refs=(),
        evidence=Evidence(BlockType.STRUCTURAL_DATA_FAILURE.value),
        message="Depth is not monotonic.",
        created_at=created_at,
    )
    other = FindingRecord(
        id=finding_key(1, "depth_token_not_decimal", ()),
        stage=1,
        check="depth_token_not_decimal",
        identity=(),
        block_type=BlockType.STRUCTURAL_DATA_FAILURE,
        subject_refs=(),
        evidence=Evidence(BlockType.STRUCTURAL_DATA_FAILURE.value),
        message="A depth token is not a finite decimal.",
        created_at=created_at,
    )
    decision = DecisionRecord(
        "dec-9",
        other.id,
        Disposition(IssueDisposition.UNDISPOSITIONED),
        None,
        None,
        DecisionScope.THIS_ITEM,
        DecisionActor.USER,
        created_at,
        "0.0.0",
        (("families", "1"),),
        FileHash("b" * 64),
        None,
        evidence_digest(other),
    )
    with pytest.raises(ValueError, match="does not match"):
        project_for_export(finding, decision)
