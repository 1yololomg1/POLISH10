"""Grab readable progress screenshots on the normal Windows platform.

The offscreen Qt platform draws no letters, so this tool refuses to run
when QT_QPA_PLATFORM is offscreen. Synthetic copies of the well files are
opened from a temporary folder. The grab is the window itself: nothing is
pasted or drawn onto it.

Structure
---------
Shot
    One output name, the well file to open, and the finding check to select.
    A finding of "-" selects nothing.
grab
    Open each shot in the shell and write docs/review/progress/<prompt>_<name>.png.
main
    Command line. Exit 2 when the platform is offscreen.

Usage
-----
python tools/progress_screenshots.py PROMPT --shot NAME PATH FINDING

FINDING is "-" when the shot should not select a finding. Repeat --shot
for each picture. Example:

python tools/progress_screenshots.py P13k --shot 159 tests/fixtures/159-19A_LFP.las - --shot null_question C:\\temp\\null.las null_undeclared
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The repository root is the parent of tools/, so polish_ui imports resolve
# when this file is run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_OUTPUT = _ROOT / "docs" / "review" / "progress"


@dataclass(frozen=True)
class Shot:
    """One picture: a name, a well file, and the finding check to select."""

    name: str
    path: Path
    finding: str


def _refuse_offscreen() -> str | None:
    """The message when this process would draw no letters, else None."""

    platform = os.environ.get("QT_QPA_PLATFORM", "")
    if platform.lower() == "offscreen":
        return (
            "Refusing to grab screenshots: QT_QPA_PLATFORM is offscreen, "
            "which draws no letters. Run this on the normal Windows platform."
        )
    return None


def _parse(argv: list[str]) -> tuple[str, tuple[Shot, ...]]:
    parser = argparse.ArgumentParser(description="Grab readable POLISH progress screenshots.")
    parser.add_argument("prompt", help="Prompt name, used as the file prefix.")
    parser.add_argument(
        "--shot",
        nargs=3,
        action="append",
        metavar=("NAME", "PATH", "FINDING"),
        required=True,
        help="Output name, well file, and finding check. Use - to select nothing.",
    )
    args = parser.parse_args(argv)
    shots = tuple(Shot(name, Path(path), finding) for name, path, finding in args.shot)
    return args.prompt, shots


def grab(prompt: str, shots: tuple[Shot, ...], output_dir: Path = _OUTPUT) -> None:
    """Open each shot and write one PNG. Refuses the offscreen platform.

    Each well file is copied into a temporary folder and opened from there.
    The window is shown on the normal platform, then closed.
    """

    refusal = _refuse_offscreen()
    if refusal is not None:
        raise RuntimeError(refusal)
    from PySide6 import QtWidgets

    from polish_ui.window import MainWindow

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    output_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        if not shot.path.is_file():
            raise FileNotFoundError(f"{shot.path} is not a file.")
        folder = Path(tempfile.mkdtemp())
        try:
            copy = folder / shot.path.name
            shutil.copyfile(shot.path, copy)
            window = MainWindow(
                app_data=folder / "app",
                local_tables_directory=folder / "tables",
                synchronous=True,
            )
            window.resize(1500, 900)
            window.show()
            application.processEvents()
            window.open_path(copy)
            application.processEvents()
            if shot.finding != "-":
                matches = [item for item in window.queue.findings_shown() if item.check == shot.finding]
                if len(matches) != 1:
                    shown = sorted({item.check for item in window.queue.findings_shown()})
                    raise RuntimeError(
                        f"{shot.name}: {shot.finding} matched {len(matches)} findings. Shown: {shown}."
                    )
                if not window.queue.select_finding(matches[0].id):
                    raise RuntimeError(f"{shot.name}: {shot.finding} could not be selected.")
                application.processEvents()
            pixmap = window.grab()
            if pixmap.isNull() or pixmap.width() < 1:
                raise RuntimeError(f"{shot.name}: the window grab is empty.")
            destination = output_dir / f"{prompt}_{shot.name}.png"
            if not pixmap.save(str(destination), "PNG"):
                raise RuntimeError(f"{shot.name}: the grab was not written to {destination}.")
            window.close()
            application.processEvents()
        finally:
            shutil.rmtree(folder, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and grab the shots. Returns a process status."""

    refusal = _refuse_offscreen()
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2
    try:
        prompt, shots = _parse(sys.argv[1:] if argv is None else argv)
        grab(prompt, shots)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
