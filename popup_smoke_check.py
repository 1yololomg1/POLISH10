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

It also verifies the two properties that manual clicking cannot show:

  1. Cleanup. The promoted Toplevel implementation registers each window in
     self.popup_windows / self.popup_figures and drains them in its on_close
     callback. The harness invokes the real close handler and asserts both
     registries return to their prior length.

  2. No pyplot leak. The whole point of promoting the Toplevel implementation
     was to stop using plt.figure(), which retains figures in pyplot's global
     registry forever. The harness asserts plt.get_fignums() stays empty, which
     is direct evidence the leak is gone rather than an eyeball check of Task
     Manager.

It deliberately does NOT judge whether a plot looks scientifically correct. That
still needs a human eye on a few representative plots.

STRUCTURE
---------
build_synthetic_well()      Deterministic wireline dataset (DataFrame + curve_info).
build_processing_results()  Per-curve original/final arrays the helpers read.
DialogCapture               Context manager intercepting messagebox calls.
CheckResult                 Outcome record for a single visualization route.
find_toolbar()              Recursively locates a matplotlib navigation toolbar.
stray_toplevels()           Detects orphan windows left by a failed route.
check_popup()               Opens, validates and closes one popup.
prepare_application()       Instantiates the app and injects the synthetic state.
main()                      Orchestration, reporting, process exit code.

USAGE
-----
    python popup_smoke_check.py

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

    curve_info: Dict[str, Dict[str, str]] = {
        DEPTH_MNEMONIC: {"curve_type": "DEPTH", "unit": "M"},
        "GR": {"curve_type": "GAMMA_RAY", "unit": "GAPI"},
        "RHOB": {"curve_type": "DENSITY", "unit": "G/C3"},
        "NPHI": {"curve_type": "NEUTRON", "unit": "V/V"},
        "RT": {"curve_type": "RESISTIVITY", "unit": "OHMM"},
        "DT": {"curve_type": "SONIC", "unit": "US/F"},
        "CALI": {"curve_type": "CALIPER", "unit": "IN"},
        "SP": {"curve_type": "SP", "unit": "MV"},
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


def prepare_application(module: Any, mismatched_grid: bool = False) -> Any:
    """Instantiate the application and inject the synthetic well state.

    The real constructor is used rather than a stub so the harness exercises the
    same widget tree, Tk variables and instance attributes the application uses
    in production. The main window is withdrawn because only the popups matter.

    With mismatched_grid set, processed_data is placed on a coarser grid than
    current_data. That reproduces a real session where resampling to standard
    depth spacing changes the row count, which is the condition that made the
    popups plot one frame's values against another frame's depth axis.
    """
    import tkinter as tk

    frame, curve_info = build_synthetic_well()

    app = module.AdvancedPreprocessingApplication()
    app.root.withdraw()

    processed = resample_frame(frame, 0.305) if mismatched_grid else frame.copy()

    app.current_data = frame
    app.processed_data = processed
    app.curve_info = curve_info
    app.processing_results = build_processing_results(processed)
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
        app = prepare_application(polish, mismatched_grid=mismatched)
    except Exception:
        print("FATAL: could not construct the application")
        traceback.print_exc()
        return 2

    print("POLISH popup visualization smoke check")
    print("Synthetic well: SMOKE-TEST-1, {} curves, {} samples".format(
        len(app.current_data.columns) - 1, len(app.current_data)))
    print("Grid mode: {}".format(
        "MISMATCHED - current_data {} rows vs processed_data {} rows".format(
            len(app.current_data), len(app.processed_data))
        if mismatched else "matched - both frames {} rows".format(len(app.current_data))))
    print("Exercising {} visualization routes".format(len(VIZ_ROUTES)))
    print()

    results: List[CheckResult] = []
    with DialogCapture(polish) as capture:
        for viz_type, curve in VIZ_ROUTES:
            outcome = check_popup(app, capture, viz_type, curve)
            results.append(outcome)
            print("  [{}] {:26s} {}".format(
                outcome.status, viz_type, "; ".join(outcome.notes) if outcome.notes else ""))

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
    print("=" * 78)

    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]

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
    print("{} of {} routes passed".format(passed, len(results)))
    if failed:
        print()
        print("Failing routes:")
        for outcome in failed:
            print("  {}".format(outcome.viz_type))
            for note in outcome.notes:
                print("      {}".format(note))

    try:
        app.root.destroy()
    except Exception:
        pass

    return 0 if (not failed and leak_clean) else 1


if __name__ == "__main__":
    sys.exit(main())
