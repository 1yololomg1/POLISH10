"""The runner enforces stage contracts. Contracts section 6, D24, C5."""

from pathlib import Path

import pytest

from polish.model.ids import SampleRef
from polish.model.overlay import Overlay
from polish.model.states import SampleStatus, StageStatus
from polish.stages.runner import RegisteredStage, Runner, StageContractViolation, StageOutcome
from tests.arch.builders import noop_stage, source_dataset, stage_contract


def _dataset(tmp_path: Path):
    path = tmp_path / "in.las"
    path.write_bytes(b"~V\n")
    return source_dataset(path)


def test_stages_0_to_4_cannot_declare_that_they_alter_data():
    for number in range(5):
        with pytest.raises(ValueError, match="D24"):
            stage_contract(number, alters_data=True)


def test_read_only_stage_that_changes_the_overlay_fails(tmp_path: Path):
    dataset = _dataset(tmp_path)
    curve = dataset.curves[0]

    def run(context):
        context.overlay.put(SampleRef(curve.ref, 0), 4.0, SampleStatus.CHANGED, "prov-1")
        return StageOutcome(StageStatus.COMPLETE, ())

    def unmet(_context):
        return ()

    stage = RegisteredStage(stage_contract(0), run, unmet)
    with pytest.raises(StageContractViolation, match="alters_data=False"):
        Runner((stage,)).run(dataset)


def test_any_stage_that_writes_a_source_array_fails(tmp_path: Path):
    dataset = _dataset(tmp_path)

    def run(context):
        values = context.dataset.curves[0].values
        values.setflags(write=True)
        values[0] = 123.0
        return StageOutcome(StageStatus.COMPLETE, ())

    def unmet(_context):
        return ()

    stage = RegisteredStage(stage_contract(5, alters_data=True), run, unmet)
    with pytest.raises(StageContractViolation, match="C5"):
        Runner((stage,)).run(dataset)


def test_unmet_requirement_does_not_run_the_stage_and_reports_why(tmp_path: Path):
    dataset = _dataset(tmp_path)
    called = {"run": False}

    def run(_context):
        called["run"] = True
        return StageOutcome(StageStatus.COMPLETE, ())

    def unmet(_context):
        return ("units: curve 0 has no declared unit",)

    stage = RegisteredStage(stage_contract(2, name="units"), run, unmet)
    results = Runner((stage,)).run(dataset)
    assert called["run"] is False
    assert len(results) == 1
    assert results[0].status is StageStatus.NOT_RUN
    assert results[0].findings == ()
    assert "units: curve 0 has no declared unit" in results[0].detail
    assert type(results[0].status) is StageStatus


def test_a_stage_depending_on_a_stage_that_did_not_complete_does_not_run(tmp_path: Path):
    dataset = _dataset(tmp_path)
    called = {"run": False}

    def blocked(_context):
        return ("preconditions: depth index is not identified",)

    def run(_context):
        called["run"] = True
        return StageOutcome(StageStatus.COMPLETE, ())

    first = RegisteredStage(stage_contract(0, name="depth"), run, blocked)
    # The first stage's run is the unmet path's run, which must not be called either.
    # Use a separate run for stage 1.
    second = RegisteredStage(stage_contract(1, depends_on=(0,)), run, lambda _context: ())
    results = Runner((second, first)).run(dataset)
    assert called["run"] is False
    by_number = {result.stage_number: result for result in results}
    assert by_number[0].status is StageStatus.NOT_RUN
    assert "depth index is not identified" in by_number[0].detail
    assert by_number[1].status is StageStatus.NOT_RUN
    assert "stage 0" in by_number[1].detail
    assert "NOT_RUN" in by_number[1].detail


def test_stale_upstream_marks_the_stage_and_its_dependent(tmp_path: Path):
    dataset = _dataset(tmp_path)
    called = {"n": 0}

    def run(_context):
        called["n"] += 1
        return StageOutcome(StageStatus.COMPLETE, ())

    def unmet(_context):
        return ()

    stages = (
        RegisteredStage(stage_contract(0), run, unmet),
        RegisteredStage(stage_contract(1, depends_on=(0,)), run, unmet),
    )
    results = Runner(stages).run(dataset, upstream_stale=frozenset({0}))
    assert called["n"] == 0
    assert results[0].status is StageStatus.STALE
    assert results[1].status is StageStatus.STALE
    assert "STALE" in results[1].detail


def test_altering_stage_may_add_an_overlay_entry(tmp_path: Path):
    dataset = _dataset(tmp_path)
    curve = dataset.curves[0]
    before = bytes(curve.values.tobytes())

    def run(context):
        context.overlay.put(SampleRef(curve.ref, 0), 4.0, SampleStatus.CREATED, "prov-2")
        return StageOutcome(StageStatus.COMPLETE, ())

    def unmet(_context):
        return ()

    overlay = Overlay()
    stage = RegisteredStage(stage_contract(5, alters_data=True), run, unmet)
    results = Runner((stage,)).run(dataset, overlay)
    assert results[0].status is StageStatus.COMPLETE
    assert curve.values.tobytes() == before
    assert overlay.read_with_overlay(curve)[0] == 4.0


def test_noop_pass_completes(tmp_path: Path):
    dataset = _dataset(tmp_path)
    results = Runner((noop_stage(),)).run(dataset)
    assert results[0].status is StageStatus.COMPLETE
    assert results[0].detail == ""
