"""Interface shell over Stages 0-4. D71, D81, D83, D86, UXC2, UXC5, C6.

Qt runs on the offscreen platform. The tests are skipped where PySide6 or
pyqtgraph cannot be imported. Loads are synchronous so no thread is involved.
App-data and local tables are temporary directories.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets

from polish.decisions import session_path
from polish.model.states import UserAction
from polish_ui import wording
from polish_ui.loading import load_well
from polish_ui.queue_panel import CHOOSE_FAMILY
from polish_ui.window import LoadWorker, MainWindow


def _las(
    *,
    curves: str = "DEPT.M : DEPTH\nGR.GAPI : GAMMA\nDT.US/FT : SONIC\n",
    data: str = "100.0 50.0 80.0\n100.5 51.0 81.0\n101.0 -999.25 82.0\n101.5 53.0 83.0\n102.0 54.0 84.0\n",
    strt: str = "100.0",
    stop: str = "102.0",
    step: str = "0.5",
    null_line: str = "NULL. -999.25 : NULL\n",
) -> str:
    return (
        "~Version\nVERS. 2.0 : VERSION\nWRAP. NO : WRAP\n~Well\n"
        f"STRT.M {strt} : START\nSTOP.M {stop} : STOP\nSTEP.M {step} : STEP\n{null_line}"
        f"~Curve\n{curves}~ASCII\n{data}"
    )


def _write(directory: Path, body: str, name: str = "well.las") -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, tmp_path: Path):
    shell = MainWindow(app_data=tmp_path / "app", local_tables_directory=tmp_path / "tables", synchronous=True)
    yield shell
    shell.close()


def _finding(window: MainWindow, check: str):
    found = [item for item in window.queue.findings_shown() if item.check == check]
    assert len(found) == 1, f"{check}: {len(found)} findings"
    return found[0]


def _choose(window: MainWindow, check: str, action: UserAction) -> None:
    assert window.queue.select_finding(_finding(window, check).id)
    index = window.queue.action_box.findData(action)
    assert index >= 0, f"{action.value} is not offered for {check}"
    window.queue.action_box.setCurrentIndex(index)


def _rail(window: MainWindow) -> list[str]:
    return [window.rail.item(row).text() for row in range(window.rail.count())]


def test_opening_a_file_shows_the_depth_and_curve_results_and_writes_nothing(window, tmp_path: Path):
    path = _write(tmp_path, _las())
    before = sorted(item.name for item in tmp_path.iterdir())
    window.open_path(path)
    rail = _rail(window)
    assert [text.split(": ")[1] for text in rail[:5]] == ["COMPLETE"] * 5
    assert all(text.endswith("NOT_RUN (not built)") for text in rail[5:])
    assert "Regularity REGULAR" in window.depth_summary.text()
    assert "Depth unit M" in window.depth_summary.text()
    assert window.table.rowCount() == 3
    assert window.table.item(1, 4).text() == "GAMMA_RAY_TOTAL"
    assert window.table.item(1, 10).text() == "Validated against rule set v1"
    assert "No value has been altered" in window.headline.text()
    assert sorted(item.name for item in tmp_path.iterdir()) == before


def test_a_structural_refusal_is_a_result_in_the_rail_with_no_disposition_controls(window, tmp_path: Path):
    body = _las(
        strt="102.0",
        stop="100.0",
        step="-0.5",
        data="102.0 54.0 84.0\n101.5 53.0 83.0\n101.0 52.0 82.0\n",
    )
    window.open_path(_write(tmp_path, body))
    rail = _rail(window)
    assert rail[1] == "1  Depth Axis: This file cannot proceed past Depth Axis"
    assert rail[2] == "2  Identification: NOT_RUN"
    assert window.headline.text().startswith("This file cannot proceed past Depth Axis.")
    assert window.queue.select_finding(_finding(window, "step_negative").id)
    assert window.queue.offered_actions() == []
    assert window.queue.form.isHidden()
    assert wording.VIEW_ONLY in window.queue.detail.toPlainText()


def test_only_actions_the_engine_can_carry_out_are_offered(window, tmp_path: Path):
    body = _las(curves="DEPT.M : DEPTH\nAPPLE.GAPI : FRUIT\nDT.US/FT : SONIC\n", null_line="")
    window.open_path(_write(tmp_path, body))
    assert window.queue.select_finding(_finding(window, "null_undeclared").id)
    assert window.queue.offered_actions() == ["REJECT_PROPOSAL", "DEFER"]
    assert wording.NO_OPTIONS in window.queue.detail.toPlainText()
    assert window.queue.select_finding(_finding(window, "mnemonic_unrecognised").id)
    assert window.queue.offered_actions() == ["MAP_EXPLICITLY", "ACKNOWLEDGE", "DEFER"]
    assert window.queue.select_finding(_finding(window, "vers_parsed_as_2").id)
    assert window.queue.offered_actions() == ["ACKNOWLEDGE", "DEFER"]
    detail = window.queue.detail.toPlainText()
    for heading in ("WHAT POLISH OBSERVED", "WHAT IT MEANS", "WHAT YOU MAY DO", "WHAT HAPPENS NEXT"):
        assert heading in detail


def test_an_acknowledgement_is_saved_beside_the_well_and_shown_after_the_rerun(window, tmp_path: Path):
    path = _write(tmp_path, _las())
    window.open_path(path)
    finding = _finding(window, "vers_parsed_as_2")
    _choose(window, "vers_parsed_as_2", UserAction.ACKNOWLEDGE)
    assert not session_path(path).exists()
    window.queue.record_button.click()
    assert session_path(path).is_file()
    assert not (tmp_path / "app").exists()
    assert window.queue.select_finding(finding.id)
    assert window.queue.tree.currentItem().text(0).startswith("[ACCEPTED]")


def test_a_deferral_without_a_reason_is_refused_inline_and_nothing_is_saved(window, tmp_path: Path):
    path = _write(tmp_path, _las())
    window.open_path(path)
    _choose(window, "vers_parsed_as_2", UserAction.DEFER)
    window.queue.record_button.click()
    assert "requires a reason" in window.queue.message.text()
    assert not session_path(path).exists()


def test_mapping_a_curve_goes_through_the_engine_and_is_labelled_user_mapped(window, tmp_path: Path):
    body = _las(curves="DEPT.M : DEPTH\nAPPLE.GAPI : FRUIT\nDT.US/FT : SONIC\n")
    path = _write(tmp_path, body)
    window.open_path(path)
    assert window.table.item(1, 4).text() == ""
    assert window.table.item(1, 10).text() == "Not automatically validated"
    _choose(window, "mnemonic_unrecognised", UserAction.MAP_EXPLICITLY)
    window.queue.family_box.setCurrentText("GAMMA_RAY_TOTAL")
    window.queue.reason_edit.setText("behaves as gamma ray")
    window.queue.record_button.click()
    assert window.queue.message.text() == ""
    assert session_path(path).is_file()
    assert window.table.item(1, 2).text() == "APPLE"
    assert window.table.item(1, 4).text() == "GAMMA_RAY_TOTAL"
    assert "User-mapped — not identified by POLISH" in window.table.item(1, 6).text()
    assert window.table.item(1, 10).text() == "Validated against rule set v1"


def test_tracks_follow_the_ticked_curves_and_hover_reads_an_actual_sample(window, tmp_path: Path):
    window.open_path(_write(tmp_path, _las()))
    assert window.selected_columns() == [1, 2]
    assert [column for _plot, column in window._plots] == [1, 2]
    between = window.readout_text(1, 100.3)
    assert between.startswith("Sample 1 · depth 100.5 M · GR 51 GAPI · MEASURED")
    absent = window.readout_text(1, 101.02)
    assert "Sample 2" in absent and "GR no value" in absent and "ABSENT" in absent
    window.table.item(2, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert [column for _plot, column in window._plots] == [1]


def test_presentation_actions_call_no_engine_function(window, app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    window.open_path(_write(tmp_path, _las()))
    window.resize(1500, 900)
    window.show()
    app.processEvents()
    calls: list[str] = []

    def _spy(name: str):
        def called(*_args: object, **_kwargs: object) -> None:
            calls.append(name)

        return called

    # The names loading.py imported, not the window's wrappers (P18.5, UXC5).
    monkeypatch.setattr("polish_ui.loading.run_well", _spy("run_well"))
    monkeypatch.setattr("polish_ui.loading.apply_decision", _spy("apply_decision"))
    monkeypatch.setattr("polish_ui.loading.save_session", _spy("save_session"))
    monkeypatch.setattr("polish_ui.loading.choose_changed_file", _spy("choose_changed_file"))
    window.set_queue_visible(False)
    window.set_queue_visible(True)
    window.table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    window.table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Checked)
    window.queue.select_finding(_finding(window, "vers_parsed_as_2").id)
    plot, _column = window._plots[0]
    center = plot.sceneBoundingRect().center()
    move = QtWidgets.QGraphicsSceneMouseEvent(QtCore.QEvent.Type.GraphicsSceneMouseMove)
    move.setScenePos(center)
    window.tracks.scene().mouseMoveEvent(move)
    app.processEvents()
    assert window.readout.text().startswith("Sample ")
    box = plot.getViewBox()
    before = box.viewRange()
    box.scaleBy((2.0, 2.0))
    zoomed = box.viewRange()
    assert zoomed != before
    box.translateBy(x=5, y=10)
    assert box.viewRange() != zoomed
    assert calls == []


def test_hiding_the_queue_leaves_the_depth_view_geometry_identical(window, app, tmp_path: Path):
    window.open_path(_write(tmp_path, _las()))
    window.show()
    app.processEvents()
    before = window.tracks.geometry()
    width = window.queue_region.width()
    window.set_queue_visible(False)
    app.processEvents()
    assert window.tracks.geometry() == before
    assert window.queue_region.width() == width
    window.set_queue_visible(True)
    app.processEvents()
    assert window.tracks.geometry() == before


def test_opening_a_second_file_carries_nothing_over(window, tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    window.open_path(_write(first, _las(null_line="")))
    assert any(item.check == "null_undeclared" for item in window.queue.findings_shown())
    body = _las(curves="DEPT.M : DEPTH\nGR.GAPI : GAMMA\n", data="100.0 50.0\n100.5 51.0\n101.0 52.0\n", stop="101.0")
    window.open_path(_write(second, body))
    assert window.table.rowCount() == 2
    assert all(item.check != "null_undeclared" for item in window.queue.findings_shown())
    assert [column for _plot, column in window._plots] == [1]
    assert not session_path(first / "well.las").exists()
    assert not session_path(second / "well.las").exists()


def _pen_colours(window) -> dict[int, str]:
    return {
        column: plot.listDataItems()[0].opts["pen"].color().name().upper()
        for plot, column in window._plots
    }


def test_each_curve_is_drawn_in_its_family_colour(window, tmp_path: Path):
    body = _las(
        curves="DEPT.M : DEPTH\nGR.GAPI : GAMMA\nDT.US/FT : SONIC\nZZQ.M : UNKNOWN\n",
        data="100.0 50.0 80.0 1.0\n100.5 51.0 81.0 2.0\n101.0 52.0 82.0 3.0\n",
        stop="101.0",
    )
    window.open_path(_write(tmp_path, body))
    assert window.table.item(1, 4).text() == "GAMMA_RAY_TOTAL"
    assert window.table.item(2, 4).text() == "SONIC_COMPRESSIONAL"
    assert window.table.item(3, 4).text() == ""
    assert _pen_colours(window) == {1: "#008000", 2: "#800080", 3: "#808080"}
    assert window.tracks.backgroundBrush().color().name().upper() == "#FFFFFF"


def test_every_shipped_family_has_a_colour():
    from polish.tables.loader import load_shipped_tables
    from polish_ui.curve_colours import FAMILY_COLOURS

    shipped = {family.id for family in load_shipped_tables().families}
    assert shipped == set(FAMILY_COLOURS)


def test_track_axes_show_values_as_written(window, tmp_path: Path):
    body = _las(
        curves="DEPT.M : DEPTH\nNPHI.V/V : NEUTRON\n",
        data="100.0 0.21\n100.5 0.35\n101.0 0.52\n",
        stop="101.0",
    )
    window.open_path(_write(tmp_path, body))
    plot, _column = window._plots[0]
    for side in ("left", "bottom"):
        assert plot.getAxis(side).autoSIPrefix is False
        assert plot.getAxis(side).autoSIPrefixScale == 1.0


def test_identified_curves_are_ticked_first_in_the_expert_order(window, tmp_path: Path):
    # Unidentified and caliper curves come first in the file; the default tracks
    # are still sonic, density, porosity and gamma ray (build plan P20 item 1).
    body = _las(
        curves=(
            "DEPT.M : DEPTH\nZZQ.M : UNKNOWN\nCALI.IN : CALIPER\nGR.GAPI : GAMMA\n"
            "RHOB.G/CM3 : DENSITY\nNPHI.V/V : NEUTRON\nDT.US/FT : SONIC\n"
        ),
        data=(
            "100.0 1.0 8.5 50.0 2.30 0.25 80.0\n"
            "100.5 2.0 8.6 51.0 2.31 0.26 81.0\n"
            "101.0 3.0 8.7 52.0 2.32 0.27 82.0\n"
        ),
        stop="101.0",
    )
    window.open_path(_write(tmp_path, body))
    families = {row: window.table.item(row, 4).text() for row in range(window.table.rowCount())}
    assert families[3] == "GAMMA_RAY_TOTAL" and families[4] == "BULK_DENSITY"
    assert families[5] == "NEUTRON_POROSITY" and families[6] == "SONIC_COMPRESSIONAL"
    assert window.selected_columns() == [3, 4, 5, 6]
    assert set(_pen_colours(window).values()) == {"#008000", "#FF0000", "#0000FF", "#800080"}


def test_a_file_the_engine_cannot_handle_is_reported_not_hung(window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A real case from 13 Nov 2026: a web server error page saved as .las. Stage 0
    # let lasio's KeyError escape, and the window waited on "Running" forever.
    page = (
        '<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">\n<html><head>\n'
        "<title>502 Proxy Error</title>\n</head><body>\n<h1>Proxy Error</h1>\n"
        "</body></html>\n"
    )
    path = _write(tmp_path, page, name="download.las")
    result = load_well(path, app_data=tmp_path / "app", local_tables_directory=tmp_path / "tables")
    assert result.run is None and result.error is not None

    def engine_fault(*_args, **_kwargs):
        raise KeyError("No ~ sections found. Is this a LAS file?")

    monkeypatch.setattr("polish_ui.loading.run_well", engine_fault)
    window.open_path(path)
    assert window.headline.text().startswith("Not read.")
    assert "Unexpected error while reading, KeyError" in window.headline.text()
    assert window.table.rowCount() == 0
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["download.las"]


def _wait_until(app, predicate, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


def _idle(window: MainWindow) -> bool:
    worker = window._worker
    running = worker is not None and worker.isRunning()
    return not running and window._queued_path is None and not window._loading


class _LoadGate:
    """Block load_well on the worker thread so a test can act during a load."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = 0
        self.block_after = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = load_well

        def gated(path: Path, **kwargs: object):
            self.calls += 1
            if self.calls > self.block_after:
                self.entered.set()
                assert self.release.wait(30)
            return real(path, **kwargs)

        monkeypatch.setattr("polish_ui.window.load_well", gated)


def _async_window(tmp_path: Path) -> MainWindow:
    return MainWindow(app_data=tmp_path / "app", local_tables_directory=tmp_path / "tables", synchronous=False)


def test_a_load_shows_running_and_the_queue_form_takes_no_input(app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    window = _async_window(tmp_path)
    gate = _LoadGate()
    gate.block_after = 1
    gate.install(monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    window.open_path(_write(first, _las()))
    _wait_until(app, lambda: _idle(window) and window.table.rowCount() == 3)
    assert window.queue.findings_shown()
    other = _write(second, _las(curves="DEPT.M : DEPTH\nGR.GAPI : GAMMA\n", data="100.0 50.0\n101.0 51.0\n", stop="101.0"))
    window.open_path(other)
    assert gate.entered.wait(30)
    rail = _rail(window)
    assert [text.split(": ", 1)[1] for text in rail[:5]] == ["RUNNING"] * 5
    assert all(text.endswith("NOT_RUN (not built)") for text in rail[5:])
    assert window.table.rowCount() == 0
    assert window.queue.findings_shown() == []
    assert other.name in window.headline.text()
    assert window.queue.form.isEnabled() is False
    assert window.queue.message.text() == "A file is still being read."
    assert "\n" not in window.queue.message.text()
    gate.release.set()
    _wait_until(app, lambda: _idle(window) and window.table.rowCount() == 2)
    assert window.queue.form.isEnabled() is True
    window.close()


def test_two_decisions_recorded_back_to_back_both_remain_after_the_reload(
    app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    window = _async_window(tmp_path)
    gate = _LoadGate()
    gate.block_after = 1
    gate.install(monkeypatch)
    path = _write(tmp_path, _las())
    window.open_path(path)
    _wait_until(app, lambda: _idle(window) and "No value has been altered" in window.headline.text())
    finding = _finding(window, "vers_parsed_as_2")
    window._record(finding, UserAction.ACKNOWLEDGE, {})
    assert gate.entered.wait(30)
    assert window.queue.form.isEnabled() is False
    window._record(finding, UserAction.DEFER, {"reason": "review later"})
    gate.release.set()
    _wait_until(app, lambda: _idle(window))
    payload = json.loads(session_path(path).read_text(encoding="utf-8"))
    assert [item["action"] for item in payload["decisions"]] == ["ACKNOWLEDGE", "DEFER"]
    assert window.queue.form.isEnabled() is True
    window.close()


def test_an_empty_ascii_section_is_refused_at_ingest(window, tmp_path: Path):
    body = (
        "~Version\nVERS. 2.0 : VERSION\nWRAP. NO : WRAP\n"
        "~Well\nSTRT.M 0.0 : START\nSTOP.M 1.0 : STOP\nSTEP.M 1.0 : STEP\nNULL. -999.25 : NULL\n"
        "~Curve\nDEPT.M : DEPTH\nGR.GAPI : GAMMA\n"
        "~ASCII\n"
    )
    window.open_path(_write(tmp_path, body))
    assert "past Ingest" in window.headline.text()
    assert _rail(window)[0] == "0  Ingest: This file cannot proceed past Ingest"


def test_a_decreasing_depth_file_draws_no_tracks(window, tmp_path: Path):
    body = _las(
        strt="102.0",
        stop="100.0",
        step="-0.5",
        data="102.0 54.0 84.0\n101.5 53.0 83.0\n101.0 52.0 82.0\n",
    )
    window.open_path(_write(tmp_path, body))
    assert window._plots == []
    assert window.tracks_note.isHidden() is False
    assert window.tracks_note.text() == "No tracks: the depth axis did not pass."
    assert window.readout.text() == "Move the pointer over a track to read the nearest actual sample."


def test_the_mapping_form_starts_empty_for_the_next_finding_and_the_next_well(window, tmp_path: Path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    body = _las(
        curves="DEPT.M : DEPTH\nAPPLE.GAPI : FRUIT\nPEAR.GAPI : FRUIT\nDT.US/FT : SONIC\n",
        data="100.0 50.0 40.0 80.0\n100.5 51.0 41.0 81.0\n101.0 52.0 42.0 82.0\n",
        stop="101.0",
    )
    path = _write(first, body, "one.las")
    window.open_path(path)
    apple = _mnemonic_finding(window, "APPLE")
    assert window.queue.select_finding(apple.id)
    index = window.queue.action_box.findData(UserAction.MAP_EXPLICITLY)
    window.queue.action_box.setCurrentIndex(index)
    assert window.queue.family_box.currentText() == CHOOSE_FAMILY
    window.queue.record_button.click()
    assert window.queue.message.text() == "Choose a family before recording."
    assert not session_path(path).exists()
    window.queue.family_box.setCurrentText("GAMMA_RAY_TOTAL")
    window.queue.reason_edit.setText("behaves as gamma ray")
    window.queue.record_button.click()
    assert session_path(path).is_file()
    pear = _mnemonic_finding(window, "PEAR")
    assert window.queue.select_finding(pear.id)
    assert window.queue.family_box.currentText() == CHOOSE_FAMILY
    assert window.queue.role_box.currentText() == ""
    assert window.queue.label_edit.text() == ""
    assert window.queue.reason_edit.text() == ""
    other = _write(
        second,
        _las(curves="DEPT.M : DEPTH\nORANGE.GAPI : FRUIT\n", data="100.0 50.0\n101.0 51.0\n", stop="101.0"),
        "two.las",
    )
    window.open_path(other)
    orange = _mnemonic_finding(window, "ORANGE")
    assert window.queue.select_finding(orange.id)
    assert window.queue.family_box.currentText() == CHOOSE_FAMILY
    assert window.queue.label_edit.text() == ""
    assert window.queue.reason_edit.text() == ""


def _mnemonic_finding(window: MainWindow, mnemonic: str):
    found = [
        item
        for item in window.queue.findings_shown()
        if item.check == "mnemonic_unrecognised" and item.evidence.as_dict().get("mnemonic") == mnemonic
    ]
    assert len(found) == 1, mnemonic
    return found[0]


def test_evidence_names_curves_and_does_not_print_a_source_hash(window, tmp_path: Path):
    source = Path(__file__).resolve().parents[1] / "fixtures" / "159-19A_LFP.las"
    folder = tmp_path / "lfp"
    folder.mkdir()
    dest = folder / source.name
    shutil.copy(source, dest)
    window.open_path(dest)
    hex64 = re.compile(r"[0-9a-fA-F]{64}")
    prefix = None
    for finding in window.queue.findings_shown():
        assert window.queue.select_finding(finding.id)
        detail = window.queue.detail.toPlainText()
        assert hex64.search(detail) is None, finding.check
        if finding.check == "prefix_group":
            prefix = detail
            assert len(detail) < 5000
    assert prefix is not None
    assert "col " in prefix


def test_filling_a_reserved_region_does_not_move_the_tracks(window, app, tmp_path: Path):
    window.open_path(_write(tmp_path, _las()))
    window.resize(1500, 900)
    window.show()
    app.processEvents()
    before = window.tracks.geometry()
    for region in (window.coverage, window.detail_record, window.export_review):
        size = region.size()
        layout = region.layout()
        taken = layout.takeAt(0)
        assert taken is not None and taken.widget() is not None
        taken.widget().setParent(None)
        filler = QtWidgets.QWidget()
        filler.setFixedSize(size)
        layout.addWidget(filler)
        app.processEvents()
        assert window.tracks.geometry() == before


def test_finished_load_workers_are_released(app, tmp_path: Path):
    window = _async_window(tmp_path)
    path = _write(tmp_path, _las())
    for _ in range(3):
        window.open_path(path)
        _wait_until(app, lambda: _idle(window) and "No value has been altered" in window.headline.text())
    for _ in range(5):
        app.processEvents()
    assert len(window.findChildren(LoadWorker)) <= 1
    window.close()


def test_only_loading_imports_the_engine_entry_points():
    root = Path(__file__).resolve().parents[2] / "polish_ui"
    banned = {"run_well", "apply_decision", "save_session", "choose_changed_file"}
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "loading.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            else:
                continue
            for name in imported:
                if name in banned:
                    offenders.append(f"{path.name} imports {name}")
    assert offenders == []
