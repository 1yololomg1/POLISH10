"""Smoke check for every popup visualization route in POLISH.

PURPOSE
-------
Drives all thirteen popup visualization types in AdvancedPreprocessingApplication
in-process, without manual clicking, and reports a pass/fail table.

WHY THIS EXISTS
---------------
_create_popup_visualization catches its own exceptions and reports them through
tkinter's messagebox. A failing plot helper therefore produces a modal dialog
rather than a traceback, which means a manual click-through can silently look
like it "worked" while a helper actually errored. This harness intercepts those
dialogs, so a swallowed failure becomes an assertable result.

It verifies two independent kinds of invariant.

The first is CRUDE: drive every route and assert only that it does not raise.
This exists because the orientation assertion below inspects axes that already
exist, so a route that throws before creating an axis contributes no axis and is
silently skipped. That is how a dead plot_log_display, which raised NameError on
every invocation, sat inside a 19-of-19 green report. Each route is driven both
before and after the processing pipeline state is present, because frame-mixing
defects only appear once processing_results exists on a different grid to
current_data.

The popup helpers are called directly on a bare Figure for this check rather
than through _create_popup_visualization, which converts any exception into a
messagebox and would hide the raise. The set of helpers driven is cross-checked
against the dispatch table in the application source, so a route added there
without a harness case is reported rather than going quietly uncovered.

The second kind is SPECIFIC, and covers the three properties that manual
clicking cannot show:

  1. Cleanup. The promoted Toplevel implementation registers each window in
     self.popup_windows / self.popup_figures and drains them in its on_close
     callback. The harness invokes the real close handler and asserts both
     registries return to their prior length.

  2. No pyplot leak. The whole point of promoting the Toplevel implementation
     was to stop using plt.figure(), which retains figures in pyplot's global
     registry forever. The harness asserts plt.get_fignums() stays empty, which
     is direct evidence the leak is gone rather than an eyeball check of Task
     Manager.

  3. Depth axis orientation. Wireline display requires depth increasing
     downward, which matplotlib expresses as ylim bottom > top. Several plot
     helpers used to call set_ylim(depth_max, depth_min) and then
     invert_yaxis(), a double flip that silently rendered depth upward. The
     orientation is invisible in a smoke test that only asks "did a window
     open", so every depth-labelled axis is asserted explicitly.

     Four of the affected helpers draw into the embedded canvas rather than a
     popup, so the popup routes alone do not reach them. They are driven
     directly by check_embedded_depth_plots().

It deliberately does NOT judge whether a plot looks scientifically correct. That
still needs a human eye on a few representative plots.

STRUCTURE
---------
build_synthetic_well()      Deterministic wireline dataset (DataFrame + curve_info).
build_processing_results()  Per-curve original/final arrays the helpers read.
resample_frame()            Puts a frame on a different grid, as resampling does.
DialogCapture               Context manager intercepting messagebox calls.
CheckResult                 Outcome record for a single visualization route.
RaiseCheck                  Outcome record for the crude does-not-raise invariant.
find_toolbar()              Recursively locates a matplotlib navigation toolbar.
stray_toplevels()           Detects orphan windows left by a failed route.
is_colorbar_axes()          Distinguishes colorbar axes from data axes.
depth_axis_violations()     Asserts depth-labelled axes are inverted.
check_popup()               Opens, validates and closes one popup.
embedded_depth_cases()      Builds the direct-call cases for embedded helpers.
check_embedded_depth_plots() Drives embedded helpers and checks orientation.
popup_helper_cases()        Direct-call cases for every popup dispatch branch.
dispatched_viz_types()      Reads the dispatch table out of the app source.
coverage_gaps()             Reports dispatch branches with no harness case.
check_routes_raise()        Drives every route and records only raises.
prepare_application()       Instantiates the app in the pre-processing state.
apply_processed_state()     Adds the state the processing pipeline leaves behind.
main()                      Orchestration, reporting, process exit code.

USAGE
-----
    python popup_smoke_check.py
    python popup_smoke_check.py --mismatched-grid

With --mismatched-grid the post-processing state places processed_data on a
coarser grid than current_data, reproducing the row-count change that
resample_to_standard_spacing introduces in a real session.

Requires a desktop session because it creates real Tk windows. Windows are
created and destroyed quickly; the main application window stays hidden.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# The depth curve is excluded from the curve lists offered to the plot helpers
# because it is the plotting ordinate, not a plottable measurement.
DEPTH_MNEMONIC = "DEPT"

# Each entry is (viz_type, curve). A curve of None means the route derives its
# own subject, either from the Tk selection variables or from the curve listbox.
# Any axis whose Y label names depth is asserted to be inverted. Matching on the
# label rather than a hardcoded route list means a depth axis added to a new plot
# is covered automatically, and non-depth plots (histogram, crossplot,
# correlation matrix) are skipped without needing to be enumerated.
DEPTH_LABEL_HINT = "depth"

VIZ_ROUTES: List[Tuple[str, Optional[str]]] = [
    ("single_curve", "GR"),
    ("single_curve_comparison", None),
    ("comparison", "GR"),
    ("multi_curve", None),
    ("log_display", None),
    ("quality_overview", None),
    ("unprocessed_curves", None),
    ("correlation_matrix", None),
    ("scatter_plot", "GR"),
    ("3d_visualization", "GR"),
    ("quality_metrics", None),
    ("uncertainty", "GR"),
    ("histogram", "GR"),
]


def build_synthetic_well(n_samples: int = 1200,
                         seed: int = 20260802) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Generate a deterministic, physically plausible wireline dataset.

    A fixed seed keeps failures reproducible. Values sit inside the ranges the
    mnemonic database expects, so range validation does not strip the data and
    mask a genuine plotting failure behind an all-NaN curve.

    Returns the DataFrame and the matching curve_info mapping. curve_info entries
    carry only 'curve_type' and 'unit', which is all the popup helpers read.
    """
    rng = np.random.default_rng(seed)

    # 0.1524 m is the standard six-inch wireline sampling increment.
    depth = 1000.0 + np.arange(n_samples) * 0.1524

    def drift(scale: float, smooth: int = 40) -> np.ndarray:
        """Random walk smoothed into a geologically plausible trend."""
        walk = np.cumsum(rng.normal(0.0, scale, n_samples))
        kernel = np.ones(smooth) / smooth
        return np.convolve(walk, kernel, mode="same")

    gamma = np.clip(75.0 + drift(1.2), 5.0, 150.0)
    # Density and neutron are anti-correlated in clean sands, which also gives
    # the crossover-shading code in the log display something real to render.
    density = np.clip(2.45 + drift(0.012), 1.90, 2.95)
    neutron = np.clip(0.22 - drift(0.004), 0.00, 0.45)
    resistivity = np.clip(np.exp(1.2 + drift(0.05)), 0.2, 2000.0)
    sonic = np.clip(90.0 - drift(0.9), 40.0, 140.0)
    caliper = np.clip(8.5 + drift(0.05), 6.0, 16.0)
    spontaneous = np.clip(-20.0 + drift(0.9), -160.0, 100.0)

    frame = pd.DataFrame({
        DEPTH_MNEMONIC: depth,
        "GR": gamma,
        "RHOB": density,
        "NPHI": neutron,
        "RT": resistivity,
        "DT": sonic,
        "CALI": caliper,
        "SP": spontaneous,
    })

    # Real logs contain washouts and tool drop-outs. Including gaps exercises the
    # NaN handling in the plot helpers, which is where popup code most often
    # breaks. The depth channel is deliberately left complete.
    for mnemonic, start, length in (("RHOB", 300, 40), ("NPHI", 305, 35), ("DT", 700, 25)):
        frame.loc[start:start + length, mnemonic] = np.nan

    # These are the fully qualified curve type names the application's own
    # display code keys off: _setup_log_display_figure looks up
    # 'DEPTH_MEASURED', and the four tracks look up 'GAMMA_RAY_TOTAL',
    # 'SPONTANEOUS_POTENTIAL', 'CALIPER_SINGLE', 'RESISTIVITY_DEEP',
    # 'NEUTRON_POROSITY', 'BULK_DENSITY' and 'SONIC_COMPRESSIONAL'. Shorter
    # aliases such as 'GAMMA_RAY' match nothing there, which would leave every
    # track empty and stop the log display from exercising its own curve
    # selection, crossover shading and QC-indicator code at all.
    curve_info: Dict[str, Dict[str, str]] = {
        DEPTH_MNEMONIC: {"curve_type": "DEPTH_MEASURED", "unit": "M"},
        "GR": {"curve_type": "GAMMA_RAY_TOTAL", "unit": "GAPI"},
        "RHOB": {"curve_type": "BULK_DENSITY", "unit": "G/C3"},
        "NPHI": {"curve_type": "NEUTRON_POROSITY", "unit": "V/V"},
        "RT": {"curve_type": "RESISTIVITY_DEEP", "unit": "OHMM"},
        "DT": {"curve_type": "SONIC_COMPRESSIONAL", "unit": "US/F"},
        "CALI": {"curve_type": "CALIPER_SINGLE", "unit": "IN"},
        "SP": {"curve_type": "SPONTANEOUS_POTENTIAL", "unit": "MV"},
    }
    return frame, curve_info


def build_processing_results(frame: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """Build the processing_results mapping the comparison helpers read.

    Only 'original_data' and 'final_data' are consumed by the popup helpers, so
    only those are produced. 'final_data' is a lightly smoothed copy so that
    comparison plots show two visibly distinct traces rather than one on top of
    the other, which would hide an axis or legend bug.
    """
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for mnemonic in frame.columns:
        if mnemonic == DEPTH_MNEMONIC:
            continue
        original = frame[mnemonic].to_numpy(dtype=float)
        series = pd.Series(original)
        smoothed = series.rolling(window=9, center=True, min_periods=1).mean().to_numpy()
        results[mnemonic] = {"original_data": original, "final_data": smoothed}
    return results


@dataclass
class DialogCapture:
    """Intercepts messagebox calls so error dialogs cannot block the run.

    The application reports popup failures through messagebox rather than by
    raising, and a modal dialog with no one to dismiss it would hang the
    harness. Captured messages are the primary failure signal.
    """

    module: Any
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    _saved: Dict[str, Callable] = field(default_factory=dict)

    def __enter__(self) -> "DialogCapture":
        box = self.module.messagebox
        for name in ("showerror", "showwarning", "showinfo"):
            self._saved[name] = getattr(box, name)

        def record(sink: List[str]) -> Callable:
            def handler(title: str = "", message: str = "", *args, **kwargs) -> str:
                sink.append("{}: {}".format(title, message))
                return "ok"
            return handler

        box.showerror = record(self.errors)
        box.showwarning = record(self.warnings)
        box.showinfo = record([])
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        box = self.module.messagebox
        for name, original in self._saved.items():
            setattr(box, name, original)

    def reset(self) -> None:
        self.errors.clear()
        self.warnings.clear()


@dataclass
class CheckResult:
    """Outcome of exercising a single visualization route."""

    viz_type: str
    curve: Optional[str]
    opened: bool = False
    has_toolbar: bool = False
    closed_cleanly: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.opened and self.has_toolbar and self.closed_cleanly and not self.notes

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class RaiseCheck:
    """Outcome of driving one route purely to see whether it raises.

    Deliberately records nothing about what was drawn. The value of this check is
    that it cannot be satisfied by a route that produces no axes, which is the
    blind spot in every other assertion in this harness.
    """

    route: str
    phase: str
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def find_toolbar(widget: Any) -> bool:
    """Recursively search a window for a matplotlib navigation toolbar.

    The toolbar is packed into an intermediate frame rather than directly into
    the Toplevel, so a shallow child scan would miss it.
    """
    try:
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    except ImportError:
        return False

    try:
        children = widget.winfo_children()
    except Exception:
        return False

    for child in children:
        if isinstance(child, NavigationToolbar2Tk):
            return True
        if find_toolbar(child):
            return True
    return False


def stray_toplevels(root: Any, known: List[Any]) -> List[Any]:
    """Return Toplevel windows that exist but were never registered for cleanup.

    _create_popup_visualization builds its Toplevel before routing to the plot
    helper and only registers it afterwards. A helper that raises therefore
    leaves an orphan window behind. Detecting these keeps one failure from
    cascading into the following checks.
    """
    orphans = []
    import tkinter as tk
    try:
        for child in root.winfo_children():
            if isinstance(child, tk.Toplevel) and child not in known:
                orphans.append(child)
    except Exception:
        pass
    return orphans


def is_colorbar_axes(ax: Any) -> bool:
    """Report whether an Axes is a colorbar rather than a data axes.

    matplotlib tags the axes it creates for a colorbar with the label
    '<colorbar>' and attaches the Colorbar back-reference to it. Both are
    checked because the attribute is private and the label is the more stable
    of the two across versions.
    """
    if getattr(ax, "_colorbar", None) is not None:
        return True
    return (ax.get_label() or "") == "<colorbar>"


def depth_axis_violations(fig: Any, context: str,
                          require_depth_axis: bool = False) -> List[str]:
    """Return a message for every depth-labelled axis that is not inverted.

    matplotlib reports limits as (bottom, top). Depth increases downward only
    when bottom > top, so that ordering is the assertion. Twin axes created with
    twinx share the parent's Y axis and usually carry no label of their own,
    which is why an unlabelled axis is skipped rather than failed.

    Colorbar axes are excluded. A scatter coloured by depth carries the label
    'Depth (m)' on its colorbar, but a colour scale reads low at the bottom and
    is correct un-inverted, so including it would report a false failure.

    With require_depth_axis set, a figure containing no depth-labelled axis at
    all is itself reported. That guards against a silent pass caused by a
    renamed label or a helper that bailed out before plotting.
    """
    problems: List[str] = []
    found = 0

    for index, ax in enumerate(fig.axes):
        if is_colorbar_axes(ax):
            continue
        label = (ax.get_ylabel() or "").strip()
        if DEPTH_LABEL_HINT not in label.lower():
            continue
        found += 1
        bottom, top = ax.get_ylim()
        if bottom <= top:
            problems.append(
                "{}: axes[{}] labelled {!r} not inverted "
                "(ylim bottom={:.4f}, top={:.4f})".format(
                    context, index, label, bottom, top))

    if require_depth_axis and found == 0:
        problems.append("{}: no depth-labelled axis found to check".format(context))

    return problems


def embedded_depth_cases(app: Any, module: Any) -> List[Tuple[str, Callable[[], Any]]]:
    """Build the direct-call cases for depth helpers that bypass the popup routes.

    Each entry returns the Figure to inspect. These helpers draw into the
    embedded canvas, so nothing in VIZ_ROUTES exercises them; without this list
    the orientation assertion would cover only two of the six affected sites.
    """
    colors = module.PHYSICAL_CONSTANTS.LOG_COLORS
    with_depth = [DEPTH_MNEMONIC, "GR", "RHOB"]
    without_depth = ["GR", "RHOB"]

    def on_fresh_axis(plot_call: Callable[[Any], None]) -> Any:
        app.ensure_figure_exists()
        ax = app.fig.add_subplot(111)
        plot_call(ax)
        return app.fig

    def depth_based_with_depth() -> Any:
        return on_fresh_axis(
            lambda ax: app._plot_depth_based_curves(ax, with_depth, colors))

    def depth_based_index_fallback() -> Any:
        # No curve in this list carries a DEPTH curve_type, so the helper falls
        # back to the row index. This is the branch that relies on invert_yaxis
        # rather than set_ylim, and it must still come out inverted.
        return on_fresh_axis(
            lambda ax: app._plot_depth_based_curves(ax, without_depth, colors))

    def unprocessed_depth_based() -> Any:
        return on_fresh_axis(
            lambda ax: app._plot_unprocessed_depth_based_curves(ax, with_depth, colors))

    def unprocessed_on_axis() -> Any:
        depth_data = app.current_data[DEPTH_MNEMONIC].values
        return on_fresh_axis(
            lambda ax: app._plot_unprocessed_curves_on_axis(
                ax, without_depth, depth_data, "Depth (m)"))

    def comparison_embedded() -> Any:
        # Covers both set_ylim blocks in plot_comparison. The first block's axes
        # is removed and re-created on a GridSpec partway through, so only the
        # second block's axes survives to be inspected here.
        app.plot_comparison("GR")
        return app.fig

    def uncertainty_embedded() -> Any:
        app.plot_uncertainty("GR")
        return app.fig

    def log_display_embedded() -> Any:
        # The four-track display builds and manages its own figure, so it is not
        # wrapped in on_fresh_axis. Its four tracks share one Y axis, which makes
        # the orientation check here worth more than on a single-axes plot: a
        # per-track flip is a no-op only at an even track count.
        app.plot_log_display()
        return app.fig

    return [
        ("_plot_depth_based_curves", depth_based_with_depth),
        ("_plot_depth_based_curves (index fallback)", depth_based_index_fallback),
        ("_plot_unprocessed_depth_based_curves", unprocessed_depth_based),
        ("_plot_unprocessed_curves_on_axis", unprocessed_on_axis),
        ("plot_comparison (embedded)", comparison_embedded),
        ("plot_uncertainty (embedded)", uncertainty_embedded),
        ("plot_log_display (embedded)", log_display_embedded),
    ]


def popup_helper_cases(app: Any, curve: str = "GR") -> List[Tuple[str, Callable[[], Any]]]:
    """Build a direct-call case for every branch of the popup dispatch table.

    Each case builds its own bare Figure and calls the plot helper on it. Going
    through _create_popup_visualization instead would defeat the purpose: that
    method wraps the whole dispatch in try/except and turns any exception into a
    messagebox, so a raising helper would look like a captured dialog rather than
    a raise. Here the exception propagates to the caller and is recorded verbatim.

    No Toplevel is created, so these cases are also cheap enough to run in both
    the pre- and post-processing phases.
    """
    from matplotlib.figure import Figure

    def bare(plot_call: Callable[[Any], None]) -> Callable[[], Any]:
        def invoke() -> Any:
            fig = Figure(figsize=(10, 8), dpi=100)
            plot_call(fig)
            return fig
        return invoke

    return [
        ("single_curve", bare(lambda fig: app._plot_single_curve_popup(fig, curve))),
        ("single_curve_comparison", bare(app._plot_single_curve_comparison_popup)),
        ("comparison", bare(lambda fig: app._plot_comparison_popup(fig, curve))),
        ("multi_curve", bare(app._plot_multi_curve_popup)),
        ("log_display", bare(app._plot_log_display_popup)),
        ("quality_overview", bare(app._plot_quality_overview_popup)),
        ("unprocessed_curves", bare(app._plot_unprocessed_curves_popup)),
        ("correlation_matrix", bare(app._plot_correlation_matrix_popup)),
        ("scatter_plot", bare(lambda fig: app._plot_scatter_plot_popup(fig, curve))),
        ("3d_visualization", bare(lambda fig: app._plot_3d_visualization_popup(fig, curve))),
        ("quality_metrics", bare(app._plot_quality_metrics_popup)),
        ("uncertainty", bare(lambda fig: app._plot_uncertainty_popup(fig, curve))),
        ("histogram", bare(lambda fig: app._plot_histogram_popup(fig, curve))),
    ]


def dispatched_viz_types(module: Any) -> Optional[List[str]]:
    """Return the viz_type values routed by _create_popup_visualization.

    Read out of the application source rather than hardcoded here. A viz_type
    added to the dispatch table but not to popup_helper_cases would otherwise go
    uncovered without anything saying so, which is the same class of silent gap
    this whole invariant exists to close.

    Returns None when the dispatch table cannot be inspected, so callers can
    treat inspection failure as a coverage failure rather than as zero gaps.
    """
    import inspect
    import re

    try:
        source = inspect.getsource(
            module.AdvancedPreprocessingApplication._create_popup_visualization)
    except (OSError, TypeError, AttributeError) as exc:
        print("WARNING: could not read the popup dispatch table: {}".format(exc))
        return None

    return re.findall(r"""viz_type\s*==\s*["']([A-Za-z0-9_]+)["']""", source)


def coverage_gaps(module: Any, covered: List[str]) -> List[str]:
    """Return dispatch-table viz_types that no harness case drives.

    Source-inspection failure is returned as an explicit unavailable state so
    it cannot be mistaken for full coverage.
    """
    dispatched = dispatched_viz_types(module)
    if dispatched is None:
        return ["<coverage inspection unavailable>"]
    if not dispatched:
        return ["<coverage inspection found no dispatch branches>"]
    return [name for name in dispatched if name not in set(covered)]


def check_routes_raise(cases: List[Tuple[str, Callable[[], Any]]],
                       phase: str,
                       capture: Optional[DialogCapture] = None) -> List[RaiseCheck]:
    """Drive each route and record whether it raised or reported a dialog error.

    Exceptions are caught only so that one broken route does not stop the
    remaining routes from being driven; every one caught is reported as a
    failure. Routes that swallow the exception and show a messagebox are also
    failures: a captured error dialog is not a pass.
    """
    checks: List[RaiseCheck] = []

    for name, invoke in cases:
        check = RaiseCheck(route=name, phase=phase)
        if capture is not None:
            capture.reset()
        try:
            invoke()
        except Exception as exc:
            check.error = "{}: {}".format(type(exc).__name__, exc)
        if capture is not None and capture.errors and check.error is None:
            check.error = "error dialog -> " + capture.errors[0].replace("\n", " ")[:160]
        checks.append(check)

    return checks


def check_embedded_depth_plots(app: Any, capture: DialogCapture,
                               module: Any) -> List[CheckResult]:
    """Drive each embedded depth helper and assert its axis orientation.

    These helpers open no window, so the window and toolbar fields of
    CheckResult do not apply and are marked satisfied. The orientation check is
    the whole point of the case.
    """
    results: List[CheckResult] = []

    for name, invoke in embedded_depth_cases(app, module):
        capture.reset()
        result = CheckResult(viz_type=name, curve=None)
        # Not window-based routes; these two fields are not meaningful here.
        result.opened = True
        result.has_toolbar = True
        result.closed_cleanly = True

        try:
            fig = invoke()
        except Exception as exc:
            result.notes.append("raised {}: {}".format(type(exc).__name__, exc))
            results.append(result)
            continue

        if capture.errors:
            result.notes.append("error dialog -> " + capture.errors[0].replace("\n", " ")[:160])

        if fig is None:
            result.notes.append("helper produced no figure")
        else:
            result.notes.extend(depth_axis_violations(fig, name, require_depth_axis=True))

        results.append(result)

    return results


def check_popup(app: Any, capture: DialogCapture, viz_type: str,
                curve: Optional[str]) -> CheckResult:
    """Open one popup, validate its structure, then close it through the real handler."""
    import matplotlib.pyplot as plt

    result = CheckResult(viz_type=viz_type, curve=curve)
    capture.reset()

    windows_before = len(app.popup_windows)
    figures_before = len(app.popup_figures)
    fignums_before = set(plt.get_fignums())

    try:
        app._create_popup_visualization(viz_type, curve)
        app.root.update()
    except Exception as exc:
        # The method is supposed to contain its own failures, so anything
        # escaping is itself a defect worth reporting verbatim.
        result.notes.append("uncaught {}: {}".format(type(exc).__name__, exc))
        for orphan in stray_toplevels(app.root, app.popup_windows):
            try:
                orphan.destroy()
            except Exception:
                pass
        return result

    if capture.errors:
        result.notes.append("error dialog -> " + capture.errors[0].replace("\n", " ")[:160])
    if capture.warnings:
        result.notes.append("warning dialog -> " + capture.warnings[0].replace("\n", " ")[:160])

    result.opened = len(app.popup_windows) == windows_before + 1
    if not result.opened:
        result.notes.append(
            "no window registered (registry {} -> {})".format(windows_before, len(app.popup_windows)))
        for orphan in stray_toplevels(app.root, app.popup_windows):
            result.notes.append("orphan Toplevel left behind by failed route")
            try:
                orphan.destroy()
            except Exception:
                pass
        return result

    popup = app.popup_windows[-1]
    result.has_toolbar = find_toolbar(popup)
    if not result.has_toolbar:
        result.notes.append("no navigation toolbar found in window")

    # Routes without a depth axis (histogram, crossplot, correlation matrix)
    # simply contribute no labelled axis and are skipped by the check.
    if app.popup_figures:
        result.notes.extend(depth_axis_violations(app.popup_figures[-1], viz_type))

    leaked = set(plt.get_fignums()) - fignums_before
    if leaked:
        result.notes.append("pyplot figure(s) created: {}".format(sorted(leaked)))

    # Invoke the real WM_DELETE_WINDOW handler rather than destroy(), so the
    # on_close cleanup path is exercised exactly as a user closing the window
    # would exercise it.
    try:
        handler = popup.protocol("WM_DELETE_WINDOW")
        if handler:
            popup.tk.call(handler)
        else:
            result.notes.append("no close handler registered")
            popup.destroy()
        app.root.update()
    except Exception as exc:
        result.notes.append("close raised {}: {}".format(type(exc).__name__, exc))
        try:
            popup.destroy()
        except Exception:
            pass
        return result

    windows_after = len(app.popup_windows)
    figures_after = len(app.popup_figures)
    result.closed_cleanly = (windows_after == windows_before and figures_after == figures_before)
    if not result.closed_cleanly:
        result.notes.append(
            "registry not drained (windows {} -> {}, figures {} -> {})".format(
                windows_before, windows_after, figures_before, figures_after))

    return result


def resample_frame(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Return the frame interpolated onto a uniform grid of a different length.

    This mirrors what resample_to_standard_spacing does to processed_data: same
    depth interval, different sample count. It exists so the harness can cover
    the case where current_data and processed_data no longer share a grid.
    """
    depth = np.asarray(frame[DEPTH_MNEMONIC].values, dtype=float)
    new_depth = np.linspace(depth[0], depth[-1], int(len(depth) * factor))
    resampled = pd.DataFrame({DEPTH_MNEMONIC: new_depth})
    for column in frame.columns:
        if column == DEPTH_MNEMONIC:
            continue
        values = np.asarray(frame[column].values, dtype=float)
        resampled[column] = np.interp(new_depth, depth, values)
    return resampled


def prepare_application(module: Any) -> Any:
    """Instantiate the application in the state a freshly loaded file leaves.

    The real constructor is used rather than a stub so the harness exercises the
    same widget tree, Tk variables and instance attributes the application uses
    in production. The main window is withdrawn because only the popups matter.

    processed_data and processing_results are left empty here on purpose. That is
    what the application looks like between loading a file and pressing the
    process button, and every route is reachable from the UI in that state. Call
    apply_processed_state() to move the app to the post-processing state.
    """
    import tkinter as tk

    frame, curve_info = build_synthetic_well()

    app = module.AdvancedPreprocessingApplication()
    app.root.withdraw()

    app.current_data = frame
    app.processed_data = None
    app.curve_info = curve_info
    app.processing_results = {}
    app.well_info = {"well_name": "SMOKE-TEST-1", "field": "SYNTHETIC"}

    plottable = [c for c in frame.columns if c != DEPTH_MNEMONIC]

    # The comparison route reads its two subjects from these Tk variables.
    if hasattr(app, "viz_curve_var"):
        app.viz_curve_var.set(plottable[0])
    if hasattr(app, "viz_curve2_var"):
        app.viz_curve2_var.set(plottable[1])

    # The multi-curve route reads the listbox selection and returns early
    # without plotting if nothing is selected.
    if hasattr(app, "curve_listbox"):
        try:
            app.curve_listbox.delete(0, tk.END)
            for mnemonic in plottable:
                app.curve_listbox.insert(tk.END, mnemonic)
            app.curve_listbox.selection_set(0, min(3, len(plottable) - 1))
        except Exception as exc:
            print("WARNING: could not populate curve listbox: {}".format(exc))

    app.root.update()
    return app


def apply_processed_state(app: Any, mismatched_grid: bool = False) -> None:
    """Move the application into the state the processing pipeline leaves behind.

    The pipeline itself runs on a background thread and depends on a long list of
    Tk option variables, so this injects its output rather than invoking it:
    processed_data as a second frame, and processing_results holding per-curve
    original/final arrays taken from that frame. Those two attributes are the
    entire post-processing contract the plot helpers read.

    With mismatched_grid set, processed_data is placed on a coarser grid than
    current_data. That reproduces a real session where resampling to standard
    depth spacing changes the row count, which is the condition that makes a
    helper plot one frame's values against another frame's depth axis.
    """
    processed = (resample_frame(app.current_data, 0.305)
                 if mismatched_grid else app.current_data.copy())
    app.processed_data = processed
    app.processing_results = build_processing_results(processed)


def main() -> int:
    """Run every route and print a report. Returns a process exit code."""
    # Import here rather than at module scope so an import failure is reported
    # with context instead of an opaque traceback at load time.
    try:
        import advanced_preprocessing_system10 as polish
    except Exception:
        print("FATAL: could not import advanced_preprocessing_system10")
        traceback.print_exc()
        return 2

    import matplotlib.pyplot as plt

    mismatched = "--mismatched-grid" in sys.argv

    try:
        app = prepare_application(polish)
    except Exception:
        print("FATAL: could not construct the application")
        traceback.print_exc()
        return 2

    print("POLISH popup visualization smoke check")
    print("Synthetic well: SMOKE-TEST-1, {} curves, {} samples".format(
        len(app.current_data.columns) - 1, len(app.current_data)))
    print()

    # Phase 1: crude does-not-raise invariant, driven in both processing states.
    # Every route is reachable from the UI before processing has run, so a route
    # that only survives a populated processing_results is a real defect.
    raise_checks: List[RaiseCheck] = []
    covered = [name for name, _ in popup_helper_cases(app)]
    gaps = coverage_gaps(polish, covered)

    print("Route-level does-not-raise invariant")
    print("-" * 78)
    for phase in ("pre-processing", "post-processing"):
        if phase == "post-processing":
            apply_processed_state(app, mismatched_grid=mismatched)
            print("  grid: {}".format(
                "MISMATCHED - current_data {} rows vs processed_data {} rows".format(
                    len(app.current_data), len(app.processed_data))
                if mismatched
                else "matched - both frames {} rows".format(len(app.current_data))))

        cases = popup_helper_cases(app) + embedded_depth_cases(app, polish)
        with DialogCapture(polish) as capture:
            phase_checks = check_routes_raise(cases, phase, capture)
        raise_checks.extend(phase_checks)

        failures = [c for c in phase_checks if not c.passed]
        print("  {:16s} {} of {} routes did not raise".format(
            phase, len(phase_checks) - len(failures), len(phase_checks)))
        for check in failures:
            print("      [FAIL] {:44s} {}".format(check.route, check.error))
    if gaps:
        print("  dispatch branches with no harness case: {}".format(", ".join(gaps)))
    print()

    # Phase 2: the specific invariants - window lifecycle, pyplot leak and depth
    # axis orientation - which need a rendered figure to inspect.
    print("Exercising {} visualization routes".format(len(VIZ_ROUTES)))
    print()

    results: List[CheckResult] = []
    embedded_results: List[CheckResult] = []
    with DialogCapture(polish) as capture:
        for viz_type, curve in VIZ_ROUTES:
            outcome = check_popup(app, capture, viz_type, curve)
            results.append(outcome)
            print("  [{}] {:26s} {}".format(
                outcome.status, viz_type, "; ".join(outcome.notes) if outcome.notes else ""))

        print()
        print("Exercising {} embedded depth-axis helpers".format(
            len(embedded_depth_cases(app, polish))))
        print()
        embedded_results = check_embedded_depth_plots(app, capture, polish)
        for outcome in embedded_results:
            print("  [{}] {:42s} {}".format(
                outcome.status, outcome.viz_type,
                "; ".join(outcome.notes) if outcome.notes else ""))

    print()
    print("=" * 78)
    print("{:<26} {:<8} {:<8} {:<8} {}".format("ROUTE", "OPENED", "TOOLBAR", "CLOSED", "RESULT"))
    print("-" * 78)
    for outcome in results:
        print("{:<26} {:<8} {:<8} {:<8} {}".format(
            outcome.viz_type,
            "yes" if outcome.opened else "no",
            "yes" if outcome.has_toolbar else "no",
            "yes" if outcome.closed_cleanly else "no",
            outcome.status))
    print("-" * 78)
    print("{:<44} {}".format("EMBEDDED DEPTH HELPER", "RESULT"))
    print("-" * 78)
    for outcome in embedded_results:
        print("{:<44} {}".format(outcome.viz_type, outcome.status))
    print("=" * 78)

    all_results = results + embedded_results
    passed = sum(1 for r in all_results if r.passed)
    failed = [r for r in all_results if not r.passed]
    raise_failed = [c for c in raise_checks if not c.passed]

    # Global leak assertions. These are the evidence that promoting the Toplevel
    # implementation actually removed the pyplot leak, rather than an inference
    # from reading the code.
    residual_windows = len(app.popup_windows)
    residual_figures = len(app.popup_figures)
    residual_fignums = plt.get_fignums()

    print()
    print("Cleanup state after all routes:")
    print("  popup_windows remaining : {}".format(residual_windows))
    print("  popup_figures remaining : {}".format(residual_figures))
    print("  pyplot figures remaining: {} {}".format(
        len(residual_fignums), residual_fignums if residual_fignums else ""))

    leak_clean = (residual_windows == 0 and residual_figures == 0 and not residual_fignums)
    print("  leak check              : {}".format("PASS" if leak_clean else "FAIL"))

    print()
    print("{} of {} route/orientation checks passed".format(passed, len(all_results)))
    print("{} of {} does-not-raise checks passed".format(
        len(raise_checks) - len(raise_failed), len(raise_checks)))
    if failed:
        print()
        print("Failing routes:")
        for outcome in failed:
            print("  {}".format(outcome.viz_type))
            for note in outcome.notes:
                print("      {}".format(note))
    if raise_failed:
        print()
        print("Routes that raised:")
        for check in raise_failed:
            print("  {} [{}]".format(check.route, check.phase))
            print("      {}".format(check.error))
    if gaps:
        print()
        print("Dispatch branches with no harness case: {}".format(", ".join(gaps)))

    try:
        app.root.destroy()
    except Exception:
        pass

    return 0 if (not failed and not raise_failed and not gaps and leak_clean) else 1


if __name__ == "__main__":
    sys.exit(main())
