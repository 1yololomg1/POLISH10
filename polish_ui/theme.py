"""Plain style constants for the shell. P18.7, UXC5.

Colour encodes curve identity only, and only through curve_colours.py (D72).
Nothing here is a visual language for provenance states (OD-25 is open).
Region heights are a layout choice (P18.1): the coverage track and the export
review grew by the same amount the curve table shrank, so at 1500x900 the
log tracks stay the height they had when those two regions were one line each.
The detail/record pane is in the queue column, so it does not take track height.
"""

from __future__ import annotations

# Track title and axis. A long mnemonic is shortened in the title only.
TITLE_CHARS = 18
TRACK_WIDTH = 160
CURVE_WIDTH = 1.5
AXIS_COLOUR = "#000000"

# Fixed regions. Widths keep the depth view from moving when the queue is hidden.
RAIL_WIDTH = 210
QUEUE_WIDTH = 440

# Heights at which later contents can be swapped without moving the tracks.
COVERAGE_HEIGHT = 72
EXPORT_HEIGHT = 72
DETAIL_HEIGHT = 200
# 230 minus the coverage growth (72 - 26) and the export growth (72 - 26).
TABLE_HEIGHT = 138
