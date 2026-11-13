"""python -m polish_ui [FILE]. Opens the shell. Reading a file alters nothing. C6."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtWidgets

from polish_ui.window import MainWindow


def main(argv: list[str] | None = None) -> int:
    """Start the interface. An optional first argument is a LAS file to open."""

    arguments = sys.argv if argv is None else argv
    app = QtWidgets.QApplication(arguments)
    window = MainWindow()
    window.show()
    if len(arguments) > 1:
        window.open_path(Path(arguments[1]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
