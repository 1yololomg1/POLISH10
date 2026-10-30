"""polish/ does not import the UI or the legacy packages. D14, C5."""

import subprocess
import sys
from pathlib import Path

from tests.arch.import_walk import imported_names

_REPO = Path(__file__).resolve().parents[2]
_LAYER_PREFIXES = (
    "tkinter",
    "PySide6",
    "matplotlib",
    "pyqtgraph",
    "polish_ui",
    "advanced_preprocessing_system10",
    "core",
    "ui",
    "petrophysics",
)


def _banned(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _LAYER_PREFIXES)


def test_stages_model_decisions_and_tables_do_not_import_pipeline():
    """Stages, the model, decisions, and tables do not import the run. D98."""

    offenders: list[str] = []
    for folder in ("stages", "model", "decisions", "tables"):
        for path, line, name in imported_names(_REPO / "polish" / folder):
            if name == "polish.pipeline" or name.startswith("polish.pipeline."):
                offenders.append(f"{path.relative_to(_REPO)}:{line} imports {name}")
    assert offenders == []


def test_stages_model_and_decisions_do_not_import_provenance():
    """Stages, the model, and decisions do not import the describe report. D95."""

    offenders: list[str] = []
    for folder in ("stages", "model", "decisions"):
        for path, line, name in imported_names(_REPO / "polish" / folder):
            if name == "polish.provenance" or name.startswith("polish.provenance."):
                offenders.append(f"{path.relative_to(_REPO)}:{line} imports {name}")
    assert offenders == []


def test_polish_does_not_import_jsonschema():
    """jsonschema is a test dependency. The application does not import it. D95."""

    offenders: list[str] = []
    for path, line, name in imported_names(_REPO / "polish"):
        if name == "jsonschema" or name.startswith("jsonschema."):
            offenders.append(f"{path.relative_to(_REPO)}:{line} imports {name}")
    assert offenders == []


def test_polish_sources_do_not_import_ui_or_legacy_modules():
    offenders: list[str] = []
    for path, line, name in imported_names(_REPO / "polish"):
        if _banned(name):
            offenders.append(f"{path.relative_to(_REPO)}:{line} imports {name}")
    assert offenders == []


def test_import_polish_and_a_runner_pass_leave_ui_toolkits_unloaded():
    script = r"""
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy

import polish
from polish.model.dataset import (
    Curve,
    Dataset,
    DepthDirection,
    DepthGrid,
    DepthRegularity,
    SourceFile,
)
from polish.model.ids import CurveRef, DatasetId
from polish.model.states import StageStatus
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, Runner, StageOutcome

path = Path(tempfile.mkdtemp()) / "in.las"
path.write_bytes(b"~V\n")
source = SourceFile.read(path, datetime(2026, 9, 24, tzinfo=timezone.utc))
dataset_id = DatasetId.for_source(source.file_hash)
depths = numpy.array([1.0, 2.0], dtype=numpy.float64)
grid = DepthGrid(depths, DepthDirection.ASCENDING, DepthRegularity.REGULAR, 1.0, "M")
curve = Curve(CurveRef(dataset_id, 0), "GR", "GAPI", "gamma", depths)
dataset = Dataset(dataset_id, None, source, grid, (curve,), ())
contract = StageContract(
    0,
    "noop",
    StageRequires("any if declared", "may see absent samples", ()),
    StageGuarantees("unchanged", "unchanged", "mask unchanged"),
    (),
    (),
    False,
)

def run(_context):
    return StageOutcome(StageStatus.COMPLETE, ())

def unmet(_context):
    return ()

Runner((RegisteredStage(contract, run, unmet),)).run(dataset)
banned = ("tkinter", "matplotlib", "PySide6", "pyqtgraph")
found = [
    name
    for name in banned
    if any(key == name or key.startswith(name + ".") for key in sys.modules)
]
if found:
    raise SystemExit("imported " + ",".join(found))
print("ok", polish.__version__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
