"""POLISH interface shell. It imports polish/; polish/ never imports it. D14, D71, D81.

This is the P18 shell over Stages 0-4, with a findings queue that records
single this-well decisions through polish.decisions. The coverage track
(P19), family-ordered tracks (P20), batch review and staleness (P21), and
export review (P22) are not built. The keymap (Appendix B) is not approved
and is not wired.
"""

import os

# pyqtgraph picks the first Qt binding it finds and tries PyQt6 before
# PySide6. This package is written against PySide6, so name it before
# pyqtgraph is imported anywhere.
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
