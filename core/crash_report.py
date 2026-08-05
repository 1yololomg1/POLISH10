"""
Local crash and startup diagnostics reporter.

STRUCTURE
---------
Module-level constants
    APP_NAME, APP_VERSION, MAX_RETAINED_REPORTS

Path and privacy helpers
    get_log_directory()     Resolve the per-user log directory, with fallbacks.
    scrub_text()            Strip directory paths and the OS user name from text.
    _scrub_path()           Reduce a single filesystem path to its basename.

Diagnostics collection
    collect_diagnostics()   Gather technical environment facts only.
    _module_versions()      Report versions of the scientific stack.

Report writing
    write_startup_record()  Append one line per launch to the session log.
    write_crash_report()    Write a full crash report file and return its path.
    prune_old_reports()     Keep the report directory bounded.

Handler installation
    install_global_handlers()   Route uncaught exceptions from the main thread,
                                worker threads, and the Tk callback loop into
                                write_crash_report().

DESIGN NOTES
------------
This module performs NO network activity of any kind, by deliberate design.
Wireline data is commercially sensitive: well names, UWIs, field names and even
directory names can disclose an operator's activity. Reports are written to the
local machine only, and it is the user's decision whether to send one to
support.

For the same reason every string that reaches a report is passed through
scrub_text(), which reduces absolute paths to bare filenames and removes the
logged-in account name. Tracebacks are rendered without local variables (the
standard traceback module behaviour), so curve arrays, headers and file
contents can never reach the report. Only the technical facts needed to
diagnose a failure are recorded: interpreter version, frozen/packaged state,
OS build, dependency versions and the exception chain.

Failure of the reporter itself must never mask the original error, so every
public entry point swallows its own exceptions and degrades to doing nothing.
"""

import os
import re
import sys
import platform
import tempfile
import traceback
from datetime import datetime

APP_NAME = "POLISH"
APP_VERSION = "10.0"

# Reports are small (a few KB). Retaining a bounded history gives support a
# chance to see a pattern of repeated failures without growing without limit.
MAX_RETAINED_REPORTS = 20

# Dependencies whose versions materially change behaviour. Version skew in this
# list is the single most common cause of "works here, fails there" reports,
# which is exactly what a packaged build hides from the user.
_TRACKED_MODULES = (
    "numpy", "pandas", "matplotlib", "scipy", "sklearn",
    "lasio", "seaborn", "pywt", "dlisio",
)

# Matches Windows ("C:\a\b\c.las", "\\server\share\x") and POSIX ("/home/a/b")
# absolute paths so the directory portion can be discarded before writing.
# \s already covers \r and \n, so they are not repeated inside the class.
_PATH_SEGMENT = r"[^\s\"'<>|:*?]"
_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/]|/)"
    r"(?:" + _PATH_SEGMENT + r"+[\\/])*" + _PATH_SEGMENT + r"*"
)


# Data-file extensions whose *names* can encode operator, well, UWI, lease or
# field identity. For these, the filename stem is dropped and only the type is
# kept. Source and library filenames (.py, .dll, .pyd, ...) are retained,
# because in a traceback they are the diagnostic payload and disclose nothing
# about the user's data.
_DATA_EXTENSIONS = frozenset({
    ".las", ".dlis", ".dat", ".csv", ".tsv", ".xlsx", ".xls",
    ".txt", ".json", ".segy", ".sgy", ".asc", ".xml",
})


def _scrub_path(match):
    """Reduce a matched absolute path so it cannot disclose the user's data.

    Directory structure is one leak vector: a path such as
    C:\\Projects\\OperatorName\\Block42\\WELL-A1.las discloses the client and
    the prospect, so the directory portion is always discarded.

    The filename is a second leak vector specific to this domain, because well
    log files are routinely named after the well, UWI, lease or field. When the
    file is a data file (see _DATA_EXTENSIONS) its stem is therefore removed and
    only the file type is kept, e.g. "<data.las>". Non-data filenames such as
    source modules are preserved, since a traceback needs them and they reveal
    nothing about the user's data.
    """
    text = match.group(0)
    base = os.path.basename(text.rstrip("\\/"))
    if not base:
        return "<path>"
    ext = os.path.splitext(base)[1].lower()
    if ext in _DATA_EXTENSIONS:
        return "<data{}>".format(ext)
    return base


def scrub_text(text):
    """Remove absolute paths and the OS account name from a string.

    Applied to every message and traceback before it is written. Returns the
    input unchanged if it is not a string, and never raises.
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        cleaned = _PATH_PATTERN.sub(_scrub_path, text)
        user = os.environ.get("USERNAME") or os.environ.get("USER")
        # The account name frequently equals the employee's real name, so it is
        # removed even when it appears outside of a path.
        if user and len(user) > 2:
            cleaned = re.sub(re.escape(user), "<user>", cleaned, flags=re.IGNORECASE)
        return cleaned
    except Exception:
        return "<unprintable>"


def get_log_directory():
    """Return the directory used for logs, creating it if necessary.

    Preference order is the per-user application data directory, then the
    system temp directory. The install directory is deliberately not used:
    under Program Files it is read-only for standard accounts, which is the
    usual reason logging silently fails on a customer machine.

    Returns None if no writable location can be established.
    """
    candidates = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(os.path.join(local_appdata, APP_NAME, "logs"))
    home = os.path.expanduser("~")
    if home and home != "~":
        candidates.append(os.path.join(home, ".{}".format(APP_NAME.lower()), "logs"))
    candidates.append(os.path.join(tempfile.gettempdir(), "{}-logs".format(APP_NAME)))

    for directory in candidates:
        try:
            os.makedirs(directory, exist_ok=True)
            # Confirm writability now rather than discovering it during a crash.
            probe = os.path.join(directory, ".write_test")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe)
            return directory
        except Exception:
            continue
    return None


def _module_versions():
    """Return versions of already-imported tracked dependencies.

    Modules are read from sys.modules rather than imported, so calling this
    during a failed startup cannot trigger further imports and cannot deepen
    an in-progress failure.
    """
    versions = {}
    for name in _TRACKED_MODULES:
        module = sys.modules.get(name)
        if module is None:
            versions[name] = "not loaded"
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _detect_bundle_mode(frozen):
    """Return 'source', 'one-folder' or 'one-file'.

    PyInstaller sets sys._MEIPASS for both packaged layouts, so its presence
    alone cannot distinguish them. The difference is where it points: a
    one-folder build resolves it to the _internal directory sitting beside the
    executable, whereas a one-file build resolves it to an extraction
    directory under the system temp folder. Comparing the parent directories
    separates the two reliably.
    """
    if not frozen:
        return "source"
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return "one-folder"
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        bundle_parent = os.path.dirname(os.path.abspath(meipass))
        return "one-folder" if os.path.normcase(exe_dir) == os.path.normcase(bundle_parent) else "one-file"
    except Exception:
        return "packaged"


def collect_diagnostics():
    """Collect technical environment facts. Contains no user or well data."""
    frozen = bool(getattr(sys, "frozen", False))
    diagnostics = {
        "app": "{} {}".format(APP_NAME, APP_VERSION),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "packaged_build": frozen,
        # Distinguishes a one-folder build from a one-file build, which changes
        # both startup cost and where dependencies are loaded from.
        "bundle_mode": _detect_bundle_mode(frozen),
        "python": platform.python_version(),
        # Level 2 strips docstrings, which some scientific libraries rely on at
        # import time. Recording it makes that class of packaging fault obvious.
        "optimize_level": sys.flags.optimize,
        "os": "{} {}".format(platform.system(), platform.release()),
        "os_build": platform.version(),
        "architecture": platform.machine(),
        "dependencies": _module_versions(),
    }
    return diagnostics


def _format_diagnostics(diagnostics):
    """Render the diagnostics mapping as an aligned plain-text block."""
    lines = []
    for key, value in diagnostics.items():
        if isinstance(value, dict):
            lines.append("{}:".format(key))
            for sub_key, sub_value in value.items():
                lines.append("    {:<14} {}".format(sub_key, sub_value))
        else:
            lines.append("{:<18} {}".format(key + ":", value))
    return "\n".join(lines)


def write_startup_record():
    """Append a single line recording this launch to the session log.

    A launch history distinguishes "the application never started" from "the
    application started and was closed", which is otherwise impossible to tell
    apart from a user's description of the problem.

    Returns the session log path, or None if logging is unavailable.
    """
    try:
        directory = get_log_directory()
        if not directory:
            return None
        diagnostics = collect_diagnostics()
        path = os.path.join(directory, "session.log")
        # Dependency versions are deliberately omitted here. This record is
        # written before the scientific stack is imported, so that a launch is
        # logged even when importing that stack is what fails. At this point
        # every version would read "not loaded", which is noise. Versions are
        # captured accurately in the crash report instead.
        line = "{} | start | {} | {} | python {} | {} {}".format(
            diagnostics["timestamp"],
            APP_VERSION,
            diagnostics["bundle_mode"],
            diagnostics["python"],
            diagnostics["os"],
            diagnostics["architecture"],
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        _trim_session_log(path)
        return path
    except Exception:
        return None


def _trim_session_log(path, max_lines=500):
    """Cap the session log so it cannot grow unbounded on a long-lived install."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        if len(lines) > max_lines:
            with open(path, "w", encoding="utf-8") as handle:
                handle.writelines(lines[-max_lines:])
    except Exception:
        pass


def prune_old_reports(directory, keep=MAX_RETAINED_REPORTS):
    """Delete the oldest crash reports beyond the retention limit."""
    try:
        reports = [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.startswith("crash-") and name.endswith(".log")
        ]
        if len(reports) <= keep:
            return
        reports.sort(key=lambda p: os.path.getmtime(p))
        for stale in reports[:-keep]:
            try:
                os.remove(stale)
            except Exception:
                continue
    except Exception:
        pass


def write_crash_report(exc_type, exc_value, exc_traceback, context=None):
    """Write a scrubbed crash report and return its path, or None on failure.

    The traceback is rendered with the standard formatter, which emits source
    lines but never local variable values, so no curve data, header content or
    file content can be captured. The rendered text is then scrubbed to remove
    directory paths and the account name.

    context is an optional short label describing what the application was
    doing (for example "startup"); it is scrubbed like everything else.
    """
    try:
        directory = get_log_directory()
        if not directory:
            return None

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(directory, "crash-{}.log".format(stamp))

        rendered = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        sections = [
            "{} crash report".format(APP_NAME),
            "=" * 60,
            "",
            "This file contains technical diagnostics only. It records no well",
            "data, no curve values, no file contents and no directory paths,",
            "and it is never transmitted anywhere. Send it to support only if",
            "you choose to.",
            "",
            "-" * 60,
            "ENVIRONMENT",
            "-" * 60,
            _format_diagnostics(collect_diagnostics()),
            "",
            "-" * 60,
            "CONTEXT",
            "-" * 60,
            scrub_text(context) if context else "not specified",
            "",
            "-" * 60,
            "ERROR",
            "-" * 60,
            "{}: {}".format(getattr(exc_type, "__name__", "Error"),
                            scrub_text(exc_value)),
            "",
            "-" * 60,
            "TRACEBACK",
            "-" * 60,
            scrub_text(rendered),
        ]

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(sections))

        prune_old_reports(directory)
        return path
    except Exception:
        # A reporter failure must never replace or hide the original exception.
        return None


def install_global_handlers(tk_root=None):
    """Route uncaught exceptions from every execution context into a report.

    Three separate channels can drop an exception on the floor in a Tkinter
    application, and all three are covered here:

    1. sys.excepthook          - the main thread outside the event loop.
    2. threading.excepthook    - background processing threads (Python 3.8+).
    3. Tk report_callback_exception - exceptions raised inside a widget
       callback, which Tk otherwise prints to a console that a windowed
       build does not have. This is why packaged GUI failures so often
       appear to the user as nothing happening at all.

    Passing tk_root is optional; the Tk channel is skipped when it is None.
    Returns the log directory in use, or None when logging is unavailable.
    """
    try:
        previous_hook = sys.excepthook

        def _handle_main(exc_type, exc_value, exc_traceback):
            write_crash_report(exc_type, exc_value, exc_traceback, context="main thread")
            previous_hook(exc_type, exc_value, exc_traceback)

        sys.excepthook = _handle_main

        import threading

        # threading.excepthook exists on Python 3.8+; sys.unraisablehook is unrelated.
        if hasattr(threading, "excepthook"):
            def _handle_thread(args):
                write_crash_report(args.exc_type, args.exc_value, args.exc_traceback,
                                   context="worker thread")

            threading.excepthook = _handle_thread

        if tk_root is not None:
            def _handle_tk(exc_type, exc_value, exc_traceback):
                write_crash_report(exc_type, exc_value, exc_traceback,
                                   context="user interface callback")

            tk_root.report_callback_exception = _handle_tk

        return get_log_directory()
    except Exception:
        return None
