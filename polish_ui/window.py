"""Main window: fixed regions around one depth view. D71, D81, D83, UXC5, C6.

Regions, each with its space reserved from the start: stage rail; coverage
track (reserved, empty until OD-25 is decided); depth-axis and curve results;
log tracks; hover readout; findings queue; detail/record pane (reserved);
export review (reserved, empty). Showing or hiding the queue does not change
the geometry of the depth view. Filling a reserved region later does not
either: each one has a fixed size and a one-line note until it is built.

Opening a file runs Stages 0-4 and alters nothing (C6). Track selection,
zoom, pan, and panel visibility are presentation only and call no engine
function that changes analytical state (UXC5). No visual language for
provenance states is invented here: a track is a plain line with gaps where
samples are absent (C7.4), and the hover readout states the sample status in
words.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from polish.decisions import Session
from polish.model.states import BlockType, StageStatus, UserAction
from polish.render import annotate, hit_test
from polish_ui import wording
from polish_ui.curve_colours import curve_colour
from polish_ui.loading import LoadResult, SessionNotWritten, load_well, record_decision
from polish_ui.queue_panel import QueuePanel
from polish_ui.theme import (
    AXIS_COLOUR,
    COVERAGE_HEIGHT,
    CURVE_WIDTH,
    DETAIL_HEIGHT,
    EXPORT_HEIGHT,
    QUEUE_WIDTH,
    RAIL_WIDTH,
    TABLE_HEIGHT,
    TITLE_CHARS,
    TRACK_WIDTH,
)

STAGES = tuple(enumerate(wording.STAGE_NAMES))
_BUILT_THROUGH = 4
MAX_TRACKS = 6
_DEFAULT_TRACKS = 4
# Build plan P20 item 1 (domain expert, 14 Sept). Sw has no family yet.
_PREFERRED_FAMILIES = (
    "SONIC_COMPRESSIONAL",
    "SONIC_SHEAR",
    "BULK_DENSITY",
    "NEUTRON_POROSITY",
    "GAMMA_RAY_TOTAL",
    "RESISTIVITY_DEEP",
)
_COLUMNS = (
    "Show",
    "Col",
    "Mnemonic",
    "Unit as written",
    "Family",
    "Role",
    "Evidence tier",
    "Processing eligibility",
    "Unit interpretation",
    "Canonical unit",
    "Validation",
)


class LoadWorker(QtCore.QThread):
    """Runs Stages 0-4 off the interface thread. No dialog is shown while it runs."""

    loaded = QtCore.Signal(object)

    def __init__(self, path: Path, app_data: Path | None, tables: Path | None, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._path = path
        self._app_data = app_data
        self._tables = tables

    def run(self) -> None:
        """Thread body."""

        self.loaded.emit(load_well(self._path, app_data=self._app_data, local_tables_directory=self._tables))


def _default_tracks(run: object, columns: list[int]) -> set[int]:
    """The curve columns ticked when a file opens, at most _DEFAULT_TRACKS.

    Identified curves come first, in the domain expert's track order (build plan
    P20 item 1, 14 Sept: compressional sonic, shear sonic, density, porosity, Sw,
    GR, deep resistivity; matched by family, never by mnemonic), then any other
    curve with a family in column order, then curves with no family. The index
    column (0) is never a track. Presentation only: nothing is run or saved."""

    identification = getattr(run, "identification", None)

    def family(column: int) -> str | None:
        if identification is None:
            return None
        return identification.curves[column].family_id

    candidates = [column for column in columns if column != 0]
    preferred = [
        column
        for wanted in _PREFERRED_FAMILIES
        for column in candidates
        if family(column) == wanted
    ]
    other_families = [c for c in candidates if family(c) is not None and c not in preferred]
    no_family = [c for c in candidates if family(c) is None]
    return set((preferred + other_families + no_family)[:_DEFAULT_TRACKS])


_READOUT_DEFAULT = "Move the pointer over a track to read the nearest actual sample."
_NO_TRACKS = "No tracks: the depth axis did not pass."
_LOAD_REASON = "A file is still being read."


def _reserved(text: str, height: int) -> QtWidgets.QWidget:
    """A fixed region. The note is the only contents until that region is built. P18.1."""

    region = QtWidgets.QWidget()
    region.setFixedHeight(height)
    layout = QtWidgets.QVBoxLayout(region)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    label = QtWidgets.QLabel(text)
    label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
    layout.addWidget(label)
    return region


def _quantity(value: object) -> str:
    if value is None:
        return "not recorded"
    unit = f" {value.unit}" if value.unit else ""
    return f"{value.text}{unit}"


class MainWindow(QtWidgets.QMainWindow):
    """The shell. ``synchronous`` loads on the calling thread, for tests."""

    def __init__(
        self,
        *,
        app_data: Path | None = None,
        local_tables_directory: Path | None = None,
        synchronous: bool = False,
    ) -> None:
        super().__init__()
        self._app_data = app_data
        self._tables = local_tables_directory
        self._synchronous = synchronous
        self._result: LoadResult | None = None
        self._worker: LoadWorker | None = None
        self._saved: Session | None = None
        self._queued_path: Path | None = None
        self._epoch = 0
        self._loading = False
        self._plots: list[tuple[object, int]] = []
        self.setWindowTitle("POLISH")
        self.resize(1500, 900)

        self.headline = QtWidgets.QLabel("No file is open. File → Open reads a LAS file and alters nothing.")
        self.headline.setWordWrap(True)
        self.headline.setFixedHeight(44)

        self.rail = QtWidgets.QListWidget()
        self.rail.setFixedWidth(RAIL_WIDTH)
        self.rail.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        self.coverage = _reserved("Coverage track: not built (P19 waits on OD-25).", COVERAGE_HEIGHT)
        self.depth_summary = QtWidgets.QLabel("")
        self.depth_summary.setWordWrap(True)
        self.depth_summary.setFixedHeight(64)
        self.table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFixedHeight(TABLE_HEIGHT)
        self.table.itemChanged.connect(self._track_selection_changed)
        self.tracks = pyqtgraph.GraphicsLayoutWidget()
        # White, like log paper: some family colours are black (caliper, depth,
        # and any curve with no family), which a black background would hide.
        self.tracks.setBackground("w")
        self.tracks.scene().sigMouseMoved.connect(self._hover)
        self.tracks_note = QtWidgets.QLabel(_NO_TRACKS, self.tracks)
        self.tracks_note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.tracks_note.hide()
        self.tracks.installEventFilter(self)
        self.readout = QtWidgets.QLabel(_READOUT_DEFAULT)
        self.readout.setFixedHeight(40)
        self.readout.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.export_review = _reserved("Export review: not built (P22).", EXPORT_HEIGHT)
        self.detail_record = _reserved("Detail and record: not built.", DETAIL_HEIGHT)

        self.queue = QueuePanel()
        self.queue.decision_requested.connect(self._record)
        self.queue_region = QtWidgets.QWidget()
        self.queue_region.setFixedWidth(QUEUE_WIDTH)
        region = QtWidgets.QVBoxLayout(self.queue_region)
        region.setContentsMargins(0, 0, 0, 0)
        region.addWidget(self.queue, 1)
        region.addWidget(self.detail_record)

        middle = QtWidgets.QVBoxLayout()
        middle.addWidget(self.headline)
        middle.addWidget(self.coverage)
        middle.addWidget(self.depth_summary)
        middle.addWidget(self.table)
        middle.addWidget(self.tracks, 1)
        middle.addWidget(self.readout)
        middle.addWidget(self.export_review)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self.rail)
        layout.addLayout(middle, 1)
        layout.addWidget(self.queue_region)
        self.setCentralWidget(central)

        open_action = QtGui.QAction("Open…", self)
        open_action.triggered.connect(self._choose_file)
        self.queue_action = QtGui.QAction("Show findings queue", self)
        self.queue_action.setCheckable(True)
        self.queue_action.setChecked(True)
        self.queue_action.toggled.connect(self.set_queue_visible)
        self.menuBar().addMenu("File").addAction(open_action)
        self.menuBar().addMenu("View").addAction(self.queue_action)
        self.statusBar().showMessage("")
        self._show_rail({number: StageStatus.NOT_RUN for number, _name in STAGES})

    # ----- presentation only (UXC5) -----

    def set_queue_visible(self, visible: bool) -> None:
        """Show or hide the queue contents. The region keeps its width."""

        self.queue.setVisible(visible)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Keep the track-area note inside the track widget when it resizes."""

        if watched is self.tracks and event.type() == QtCore.QEvent.Type.Resize:
            self.tracks_note.setGeometry(self.tracks.rect())
        return super().eventFilter(watched, event)

    # ----- loading -----

    def _choose_file(self) -> None:
        name, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open a LAS file", "", "LAS files (*.las);;All files (*)")
        if name:
            self.open_path(Path(name))

    def open_path(self, path: Path) -> None:
        """Run Stages 0-4 on ``path`` and show the result. Nothing is altered or written.

        A load started while another is still running is queued. It is not
        dropped (P21.7). The queue form takes no input until the load that is
        showing has finished.
        """

        path = Path(path)
        if self._synchronous:
            self._clear_for_load(path)
            self._saved = None
            self.show_result(load_well(path, app_data=self._app_data, local_tables_directory=self._tables))
            return
        if self._saved is not None and Path(self._saved.input_path) != path:
            self._saved = None
        # A result from a load that started before this request is stale.
        self._epoch += 1
        if self._worker is not None and self._worker.isRunning():
            self._queued_path = path
            self._clear_for_load(path)
            return
        self._queued_path = None
        self._start_worker(path)

    def _start_worker(self, path: Path) -> None:
        """Start one background load at the current epoch. The worker is deleted when it finishes."""

        self._clear_for_load(path)
        epoch = self._epoch
        worker = LoadWorker(path, self._app_data, self._tables, self)
        worker.loaded.connect(lambda result, epoch=epoch: self._accept_load(result, epoch))
        worker.finished.connect(lambda worker=worker: self._retire(worker))
        self._worker = worker
        worker.start()

    def _accept_load(self, result: LoadResult, epoch: int) -> None:
        """Show ``result`` only when no later save or open has superseded this load."""

        if epoch != self._epoch:
            return
        self._saved = None
        self.show_result(result)

    def _retire(self, worker: LoadWorker) -> None:
        """Drop a finished worker and start the load that was waiting for it. P21.7."""

        if self._worker is worker:
            self._worker = None
        worker.deleteLater()
        if self._queued_path is None:
            return
        path = self._queued_path
        self._queued_path = None
        self._start_worker(path)

    def _clear_for_load(self, path: Path) -> None:
        """Show that this file is being read, and nothing from the previous file. P18.2, P18.3."""

        self._loading = True
        self.headline.setText(f"Running Stages 0–4 on {path.name}. Nothing in the file is altered.")
        running = {
            number: StageStatus.RUNNING if number <= _BUILT_THROUGH else StageStatus.NOT_RUN
            for number, _name in STAGES
        }
        self._show_rail(running)
        self.table.setRowCount(0)
        self.tracks.clear()
        self._plots = []
        self.tracks_note.hide()
        self.readout.setText(_READOUT_DEFAULT)
        self.depth_summary.setText("Depth axis: not run.")
        self.queue.set_items((), None, ())
        self.queue.set_form_enabled(False, _LOAD_REASON)

    def show_result(self, result: LoadResult) -> None:
        """Fill every region from one load. No state is carried over from the previous file."""

        self._loading = False
        self._saved = None
        self._result = result
        self._show_rail(self._statuses(result))
        self.headline.setText(self._headline(result))
        self._show_depth(result)
        self._show_curves(result)
        curves = self._loaded_curves(result)
        if result.run is not None:
            session = result.run.session_load.session
            self.queue.set_items(session.findings, session, curves)
        elif result.partial is not None:
            self.queue.set_items(result.partial.findings, None, curves)
        else:
            self.queue.set_items((), None, curves)
        self._draw_tracks()

    def _statuses(self, result: LoadResult) -> dict[int, StageStatus]:
        statuses = {number: StageStatus.NOT_RUN for number, _name in STAGES}
        if result.run is not None:
            run = result.run
            statuses[0] = run.ingest.status
            for number, part in ((1, run.depth), (2, run.identification), (3, run.units), (4, run.validity)):
                if part is not None:
                    statuses[number] = part.status
        elif result.partial is not None:
            statuses[0] = result.partial.status
        return statuses

    def _loaded_curves(self, result: LoadResult) -> tuple:
        """Curves of the dataset this load produced, or none."""

        run = result.run
        if run is not None and run.ingest.dataset is not None:
            return run.ingest.dataset.curves
        if result.partial is not None and result.partial.dataset is not None:
            return result.partial.dataset.curves
        return ()

    def _show_rail(self, statuses: dict[int, StageStatus]) -> None:
        self.rail.clear()
        for number, name in STAGES:
            suffix = "" if number <= _BUILT_THROUGH else " (not built)"
            status = statuses[number]
            # Glossary line 193: the rail shows the structural sentence, not the token. UXC2.
            shown = wording.structural_sentence(name) if status is StageStatus.REFUSED_STRUCTURAL else status.value
            self.rail.addItem(f"{number}  {name}: {shown}{suffix}")

    def _refused_stage(self, result: LoadResult) -> int | None:
        """The stage whose status is REFUSED_STRUCTURAL, if one refused."""

        if result.run is not None:
            run = result.run
            parts = (
                (0, run.ingest),
                (1, run.depth),
                (2, run.identification),
                (3, run.units),
                (4, run.validity),
            )
            for number, part in parts:
                if part is not None and part.status is StageStatus.REFUSED_STRUCTURAL:
                    return number
            return None
        if result.partial is not None and result.partial.status is StageStatus.REFUSED_STRUCTURAL:
            return 0
        return None

    def _headline(self, result: LoadResult) -> str:
        if result.error is not None:
            return f"Not read. {result.error}"
        if result.unclassified is not None:
            return f"Unclassified at Stage 0. {result.unclassified}"
        run = result.run
        findings = run.session_load.session.findings
        refused = self._refused_stage(result)
        if refused is not None:
            structural = [item for item in findings if item.block_type is BlockType.STRUCTURAL_DATA_FAILURE]
            extra = f" {structural[0].message}" if structural else ""
            return f"{wording.structural_sentence(wording.STAGE_NAMES[refused])}.{extra}"
        curves = 0 if run.ingest.dataset is None else len(run.ingest.dataset.curves)
        return f"Read {result.path.name}: {curves} curves, {len(findings)} findings. No value has been altered."

    def _show_depth(self, result: LoadResult) -> None:
        depth = None if result.run is None else result.run.depth
        if depth is None:
            self.depth_summary.setText("Depth axis: not run.")
            return
        direction = "not recorded" if depth.direction is None else depth.direction.value
        regularity = "not recorded" if depth.regularity is None else depth.regularity.value
        unit = depth.depth_unit if depth.depth_unit else "not recorded"
        self.depth_summary.setText(
            f"Depth axis: {depth.status.value}. Direction {direction}. Regularity {regularity}. Depth unit {unit}.\n"
            f"Declared step {_quantity(depth.declared_step)}. Observed step {_quantity(depth.observed_step)}. "
            f"Corrected step {_quantity(depth.corrected_step)}. Displacement {_quantity(depth.displacement)}.\n"
            f"Recorded STRT {_quantity(depth.recorded_strt)}. Recorded STOP {_quantity(depth.recorded_stop)}."
        )

    def _show_curves(self, result: LoadResult) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        run = result.run
        if run is not None:
            rows = [(curve.column_index, curve.mnemonic, curve.unit) for curve in run.ingest.audit.curves]
        elif result.partial is not None:
            rows = [(curve.column_index, curve.mnemonic, curve.unit) for curve in result.partial.audit.curves]
        else:
            rows = []
        has_data = run is not None and run.ingest.dataset is not None
        ticked = _default_tracks(run, [column for column, _mnemonic, _unit in rows]) if has_data else set()
        for column, mnemonic, unit in rows:
            identified = None if run is None or run.identification is None else run.identification.curves[column]
            interpreted = None if run is None or run.units is None else run.units.curves[column]
            assessed = None if run is None or run.validity is None else run.validity.curves[column]
            tier = "" if identified is None else identified.evidence_tier.value
            if identified is not None and identified.label:
                tier = f"{tier} ({identified.label})"
            cells = [
                "",
                str(column),
                mnemonic,
                unit,
                "" if identified is None or identified.family_id is None else identified.family_id,
                "" if identified is None else identified.role.value,
                tier,
                "" if interpreted is None else interpreted.processing_eligibility.value,
                "" if interpreted is None else interpreted.outcome.value,
                "" if interpreted is None or interpreted.canonical_unit is None else interpreted.canonical_unit,
                "" if assessed is None else wording.validation_wording(assessed.outcome, assessed.rule_set_version),
            ]
            row = self.table.rowCount()
            self.table.insertRow(row)
            for position, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if position == 0 and has_data and column != 0:
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                    checked = column in ticked
                    item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
                self.table.setItem(row, position, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    # ----- tracks and hover (presentation only) -----

    def selected_columns(self) -> list[int]:
        """Curve columns ticked for display, at most MAX_TRACKS."""

        columns: list[int] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
                columns.append(int(self.table.item(row, 1).text()))
        return columns[:MAX_TRACKS]

    def _track_selection_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() == 0:
            self._draw_tracks()

    def _draw_tracks(self) -> None:
        self.tracks.clear()
        self._plots = []
        self.tracks_note.hide()
        run = None if self._result is None else self._result.run
        # Tracks are a Stage 1 view. A file that stopped earlier is not drawn,
        # including a decreasing depth axis, which would not be strictly
        # ascending (D3, P17, P20).
        if run is None or run.depth is None or run.depth.status is not StageStatus.COMPLETE:
            if self._result is not None:
                self.tracks_note.setGeometry(self.tracks.rect())
                self.tracks_note.show()
            return
        if run.ingest.dataset is None:
            return
        dataset = run.ingest.dataset
        depths = dataset.grid.values
        first = None
        for position, column in enumerate(self.selected_columns()):
            curve = dataset.curves[column]
            plot = self.tracks.addPlot(row=0, col=position)
            unit = curve.original_unit if curve.original_unit else "no unit"
            name = curve.original_mnemonic
            # A long mnemonic is shortened in the track title only, so every
            # track keeps the same width. The table row carries it in full.
            short = name if len(name) <= TITLE_CHARS else name[: TITLE_CHARS - 1] + "…"
            family = self._family_of(column)
            colour = curve_colour(family)
            plot.setTitle(short, size="9pt", color=colour)
            what = family if family is not None else "no family: drawn grey"
            plot.titleLabel.setToolTip(f"{name} ({unit}) · {what}")
            for side in ("left", "bottom"):
                plot.getAxis(side).setPen(AXIS_COLOUR)
                plot.getAxis(side).setTextPen(AXIS_COLOUR)
                # Show values as written: no automatic rescaling (porosity
                # 0.2-0.6 must not read as 200-600 "x0.001").
                plot.getAxis(side).enableAutoSIPrefix(False)
            plot.setLabel("bottom", unit, color=AXIS_COLOUR)
            plot.invertY(True)
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.plot(
                numpy.asarray(curve.values),
                numpy.asarray(depths),
                connect="finite",
                pen=pyqtgraph.mkPen(colour, width=CURVE_WIDTH),
            )
            if first is None:
                first = plot
                depth_unit = dataset.grid.unit if dataset.grid.unit else "no unit"
                plot.setLabel("left", f"Depth ({depth_unit})", color=AXIS_COLOUR)
            else:
                plot.setYLink(first)
                plot.getAxis("left").setStyle(showValues=False)
            self.tracks.ci.layout.setColumnStretchFactor(position, 1)
            self.tracks.ci.layout.setColumnPreferredWidth(position, TRACK_WIDTH)
            self._plots.append((plot, column))

    def _family_of(self, column: int) -> str | None:
        """Stage 2's family for ``column``, or None when Stage 2 gave none."""

        run = None if self._result is None else self._result.run
        if run is None or run.identification is None:
            return None
        return run.identification.curves[column].family_id

    def _hover(self, position: object) -> None:
        for plot, column in self._plots:
            if plot.sceneBoundingRect().contains(position):
                depth = plot.vb.mapSceneToView(position).y()
                self.readout.setText(self.readout_text(column, float(depth)))
                return

    def readout_text(self, column: int, depth: float) -> str:
        """The hover sentence for the actual sample of ``column`` nearest ``depth``."""

        run = self._result.run
        dataset = run.ingest.dataset
        curve = dataset.curves[column]
        index = hit_test(dataset.grid.values, depth)
        mark = None if run.validity is None else run.validity.curves[column].samples[index]
        note = annotate(
            dataset.grid.values,
            curve.values,
            index,
            depth_unit=dataset.grid.unit,
            unit=curve.original_unit,
            status=None if mark is None else mark.status,
            output_inclusion=None if mark is None else mark.output_inclusion,
            invalid_rules=() if mark is None else mark.invalid_rules,
        )
        value = "no value" if note.value is None else f"{note.value:g} {note.unit}".strip()
        status = "sample status not assessed" if note.status is None else note.status.value
        parts = [
            f"Sample {note.sample_index}",
            f"depth {note.depth:g} {note.depth_unit}".strip(),
            f"{curve.original_mnemonic} {value}",
            status,
        ]
        if note.output_inclusion is not None:
            parts.append(f"output inclusion {note.output_inclusion.value}")
        if note.invalid_rules:
            parts.append("marked by " + ", ".join(note.invalid_rules))
        return " · ".join(parts)

    # ----- decisions -----

    def _record(self, finding: object, action: UserAction, fields: dict) -> None:
        """Record against the session last saved, never against an older one. D47, P21.7.

        The saved session stays current until the reload of that save is the
        result on screen. A reload requested while a load is running is queued
        by ``open_path``; it is not dropped.
        """

        session = self._saved
        if session is None:
            if self._loading or self._result is None or self._result.run is None:
                self.queue.set_form_enabled(False, _LOAD_REASON)
                return
            session = self._result.run.session_load.session
        try:
            recorded = record_decision(session, finding, action, app_data=self._app_data, **fields)
        except SessionNotWritten as exc:
            self.queue.show_message(exc.finding.message)
            return
        except ValueError as exc:
            self.queue.show_message(f"Not recorded. {exc}")
            return
        self._saved = recorded.session
        self.statusBar().showMessage(f"Decision recorded. Session saved to {recorded.path}")
        self.open_path(self._result.path)
