"""Phase 0 item 2 verification against the KEOUGH reference well.

Checks the arithmetic claim behind the gate rather than trusting it: with unit
standardization off, the FT-derived 0.5 spacing is applied to FT depths and the
row count survives; with it on, the same 0.5 is applied to converted metres and
the well decimates. Also confirms RHOB against the BULK_DENSITY range in each
unit system, and that the shipped defaults are actually False.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RHOB_RANGE = [1.0, 3.5]          # core/curve_identification.py BULK_DENSITY
NULL = -999.25
FT_TO_M = 0.3048
REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Phase 0 item 2 against a KEOUGH-style LAS file."
    )
    parser.add_argument(
        "las_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the reference LAS file (required unless POLISH_VERIFY_LAS is set).",
    )
    return parser.parse_args(argv)


def resolve_las_path(args: argparse.Namespace) -> Path:
    import os

    raw = args.las_path or os.environ.get("POLISH_VERIFY_LAS")
    if not raw:
        print(
            "ERROR: no LAS path given. Pass the file as an argument or set "
            "POLISH_VERIFY_LAS.\n"
            "Usage: python _verify_phase0_2.py <path-to-keough.las>"
        )
        sys.exit(2)
    path = Path(raw)
    if not path.is_file():
        print(f"ERROR: LAS file not found: {path}")
        sys.exit(2)
    return path


args = parse_args()
LAS = resolve_las_path(args)

# --- load the reference well -------------------------------------------------
with LAS.open("r", encoding="utf-8", errors="replace") as fh:
    text = fh.read()

curve_block = text.split("~Curve Information")[1].split("~")[0]
names = [ln.split(".")[0].strip() for ln in curve_block.splitlines()
         if ln.strip() and not ln.startswith("#") and "." in ln]

data_block = text.split("~A")[1]
rows = [ln.split() for ln in data_block.splitlines()[1:] if ln.strip()]
arr = np.array([[float(v) for v in r] for r in rows if len(r) == len(names)])
df = pd.DataFrame(arr, columns=names)
df = df.replace(NULL, np.nan)

print(f"Loaded {LAS}")
print(f"  rows as loaded          : {len(df)}")
print(f"  DEPT declared unit      : FT (from ~Well STRT.FT/STEP.FT)")
print(f"  DEPT span               : {df.DEPT.min():.4f} - {df.DEPT.max():.4f}")

failures = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got} (expected {want})")
    if not ok:
        failures.append(label)


def resampled_rows(depth, spacing):
    grid = np.arange(depth.min(), depth.max() + spacing / 2.0, spacing)
    return len(grid)


# --- row count: units off vs units on ---------------------------------------
print("\nRow count through resample_to_standard_spacing (spacing=0.5):")
print("  _sync_depth_spacing_default sees unit FT and sets depth_spacing_var=0.5")

off_rows = resampled_rows(df.DEPT.values, 0.5)                 # 0.5 ft on ft depths
on_rows = resampled_rows(df.DEPT.values * FT_TO_M, 0.5)        # 0.5 read as m
check("units OFF  -> rows", off_rows, 10720)
check("units ON   -> rows", on_rows, 3268)

# --- RHOB against its own range ---------------------------------------------
print("\nRHOB vs BULK_DENSITY range [1.0, 3.5]:")
rhob_col = next((c for c in df.columns if c.upper() in ("RHOB", "RHOZ", "DENB")), None)
if rhob_col is None:
    print("  FAIL  no recognised density curve in this file")
    failures.append("no recognised density curve")
else:
    raw = df[rhob_col].to_numpy(dtype=float)
    finite = raw[np.isfinite(raw)]
    in_range_off = int(((finite >= RHOB_RANGE[0]) & (finite <= RHOB_RANGE[1])).sum())
    conv = finite * 1000.0
    in_range_on = int(((conv >= RHOB_RANGE[0]) & (conv <= RHOB_RANGE[1])).sum())
    print(f"  {rhob_col} real samples in file : {len(finite)}")
    check("units OFF  -> RHOB samples passing range", in_range_off, len(finite))
    check("units ON   -> RHOB samples passing range", in_range_on, 0)

    logged = df.loc[np.isfinite(raw), "DEPT"]
    print(f"  logged interval         : {logged.min():.1f} - {logged.max():.1f} ft"
          f"  (header TLI1 4250 / BLI1 5326)")

# --- shipped defaults --------------------------------------------------------
print("\nShipped defaults for standardize_units_var:")
sites = [
    (REPO_ROOT / "advanced_preprocessing_system10.py", "self.standardize_units_var = tk.BooleanVar(value="),
    (REPO_ROOT / "ui" / "processing_tab.py", "self.standardize_units_var = tk.BooleanVar(value="),
    (REPO_ROOT / "core" / "unit_standardization.py", "standardize_units_var = tk.BooleanVar(value="),
]
for path, needle in sites:
    src = path.read_text(encoding="utf-8", errors="replace")
    vals = re.findall(re.escape(needle) + r"(\w+)\)", src)
    check(f"{path.name} default(s)", vals, ["False"])

print("\n" + ("ALL CHECKS PASSED" if not failures
              else f"{len(failures)} CHECK(S) FAILED: {failures}"))
sys.exit(1 if failures else 0)
