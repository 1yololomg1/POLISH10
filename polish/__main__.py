"""python -m polish. Dispatches to polish.cli. Writes nothing by itself."""

from __future__ import annotations

import sys

from polish.cli import main

if __name__ == "__main__":
    sys.exit(main())
