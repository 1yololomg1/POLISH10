# P04 toolkit measurements (for OD-07)

Throwaway spike evidence lives on `spike/toolkit` and must not be merged. Figures
below were read from `spike/toolkit:spikes/toolkit/results/summary.json` and
`spike/toolkit:spikes/toolkit/results/pyinstaller.json` on 16 Sept 2026
(commit `2b05cb9`). They are not a toolkit recommendation.

## Interactivity (both measured wells)

Hover: 200 synthetic mouse-move events, `numpy` Generator seed 4. Zoom times
are full-well → 10 ft window and back. Peak memory is Windows `peak_wset` from
psutil.

| Prototype | Well | Hover median (ms) | Hover p95 (ms) | Zoom in (ms) | Zoom out (ms) | Peak WSet (MB) |
|---|---|---:|---:|---:|---:|---:|
| A PySide6 + matplotlib QtAgg | 159-19A_LFP | 2.663 | 15.719 | 581 | 611 | 180.5 |
| B PySide6 + pyqtgraph | 159-19A_LFP | 0.676 | 2.544 | 38 | 29 | 158.0 |
| C Tkinter + matplotlib TkAgg | 159-19A_LFP | 2.727 | 6.952 | 419 | 370 | 137.2 |
| A | WLC_PETRO…15_9-F-1B | 2.357 | 4.421 | 659 | 584 | 180.0 |
| B | WLC_PETRO…15_9-F-1B | 0.469 | 1.039 | 89 | 94 | 169.6 |
| C | WLC_PETRO…15_9-F-1B | 2.523 | 4.043 | 420 | 425 | 135.9 |

Across both wells:

- Prototype B hover median is 0.469–0.676 ms; zoom 29–94 ms.
- Prototype A hover median is 2.357–2.663 ms (p95 up to 15.719 ms on 159-19A_LFP); zoom 581–659 ms.
- Prototype C hover and zoom are comparable to A (hover median 2.523–2.727 ms; zoom 370–425 ms). The rebuild architecture keeps `polish/` free of any UI toolkit; Tkinter is the current monolith's toolkit and is not a candidate for `polish_ui/` under that rule. This does not fill OD-07's plot-layer choice between matplotlib and pyqtgraph.
- Memory is in the same band for A and B (158–180.5 MB peak WSet). C is lower (135.9–137.2 MB). The three are not identical.

## Packaging size (PyInstaller one-folder)

| Prototype | Success | Folder size (MB) |
|---|---|---:|
| A | yes (second attempt) | 1478.9 |
| B | yes (second attempt) | 1474.5 |
| C | yes (second attempt) | 843.6 |

Measured range: **843.6–1478.9 MB**, not the earlier 50–80 MB estimate.

This number is **not usable for OD-07 as measured.** The PyInstaller run used
`--collect-all` against the global site-packages. The captured logs name
`torch`, `yt_dlp`, `requests`, `websockets`, `Cryptodome` and other packages
with no relationship to POLISH. First attempt aborted because PyQt6 and
PySide6 were both installed; the recorded sizes are from a second attempt that
excluded PyQt6/PyQt5.

`requests`, `websockets` and `yt_dlp` are networking-capable. That is a
measurement artifact of the contaminated environment, not app code, but it
conflicts with the no-network-calls requirement (D14) if those packages were
ever shipped. Frozen binaries were not launched.

OD-07's size question needs a rerun from a clean virtual environment scoped to
`polish/`'s real imports before the figure can be trusted.
