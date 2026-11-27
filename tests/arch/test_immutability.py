"""Source arrays and the input file stay untouched. C5, D17, D43."""

import builtins
from pathlib import Path

import numpy
import pytest

from polish.model.ids import FileHash, SampleRef
from polish.model.overlay import Overlay
from polish.model.states import SampleStatus
from tests.arch.builders import child_dataset, run_noop, source_dataset

_FIXTURES = sorted(Path(__file__).resolve().parents[1].joinpath("fixtures").glob("*.las"))


def _opens_for_write(mode: object) -> bool:
    if not isinstance(mode, str) or mode == "":
        return True
    return mode[0] in "wax" or "+" in mode


def test_writing_curve_values_raises(tmp_path: Path):
    path = tmp_path / "in.las"
    path.write_bytes(b"~V\n")
    dataset = source_dataset(path)
    for curve in dataset.curves:
        with pytest.raises(ValueError, match="read-only"):
            curve.values[0] = 0.0


def test_overlay_returns_a_new_array_and_leaves_the_source(tmp_path: Path):
    path = tmp_path / "in.las"
    path.write_bytes(b"~V\n")
    dataset = source_dataset(path)
    curve = dataset.curves[0]
    before = numpy.array(curve.values, copy=True)
    overlay = Overlay()
    assert overlay.read_original(curve) is curve.values
    overlay.put(SampleRef(curve.ref, 1), 99.0, SampleStatus.CHANGED, "prov-1")
    overlaid = overlay.read_with_overlay(curve)
    overlaid[1] = 5.0
    assert overlaid is not curve.values
    assert numpy.array_equal(curve.values, before)
    assert curve.values.flags.writeable is False
    assert overlaid[1] == 5.0
    fresh = overlay.read_with_overlay(curve)
    assert fresh[1] == 99.0
    assert numpy.array_equal(curve.values, before)


def test_child_dataset_does_not_mutate_the_parent(tmp_path: Path):
    path = tmp_path / "in.las"
    path.write_bytes(b"~V\n")
    parent = source_dataset(path)
    before = tuple(numpy.array(curve.values, copy=True) for curve in parent.curves)
    grid_before = numpy.array(parent.grid.values, copy=True)
    child = child_dataset(parent)
    for curve, snapshot in zip(parent.curves, before, strict=True):
        assert numpy.array_equal(curve.values, snapshot)
        assert curve.values.flags.writeable is False
    assert numpy.array_equal(parent.grid.values, grid_before)
    assert child.grid is not parent.grid
    assert child.grid.values is not parent.grid.values
    assert child.curves[0].values is not parent.curves[0].values
    assert child.parent_id == parent.id
    assert child.lineage[0].parent_id == parent.id


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda path: path.name)
def test_fixture_hash_is_unchanged_by_a_runner_pass(fixture: Path, monkeypatch: pytest.MonkeyPatch):
    before = FileHash.from_path(fixture)
    dataset = source_dataset(fixture)
    modes: list[tuple[object, object]] = []
    real_open = builtins.open
    real_path_open = Path.open

    def tracking_open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        modes.append((file, mode))
        assert not _opens_for_write(mode), mode
        return real_open(file, mode, *args, **kwargs)

    def tracking_path_open(self: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
        modes.append((self, mode))
        assert not _opens_for_write(mode), mode
        return real_path_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    monkeypatch.setattr(Path, "open", tracking_path_open)
    run_noop(dataset)
    after = FileHash.from_path(fixture)
    assert before == after
    assert dataset.source_file.file_hash == before
    for _target, mode in modes:
        assert not _opens_for_write(mode)
