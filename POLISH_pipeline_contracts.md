# POLISH Pipeline — Contract Specification

**Derived from:** KEOUGH #12-34 (`1052970244.las`) stage-by-stage diagnostic, August 2026
**Purpose:** Define the contracts the processing pipeline must honour. Change the pipes, not the tank.
**Status:** Draft for review. Not yet implemented.

---

## 1. Why this exists — the evidence

A single well was traced stage by stage. Three curves, three different outcomes:

| Curve | In | Out | Killed by | Root cause |
|---|---|---|---|---|
| RHOB | 2169 real values, G/CC | **0 real values** | `apply_range_validation` | `uniformize_curves` converted G/CC → KG/M3 (×1000). Validator compared 2180–3247 against a g/cc range of (0.5, 7.0). |
| TBHV | 9418 real values, FT3 | **8 real values** | `apply_range_validation` | Mnemonic unrecognised. `create_comprehensive_curve_info` returned the dictionary fallback `[0.0, 1.0]`. Curve type UNKNOWN, confidence 0.00 — and it validated anyway. |
| GR | 10631 real values, GAPI | 3268 values, peak clipped | `detect_outliers_iqr` | Symmetric IQR fences on a right-skewed distribution. Upper bound 121.46 GAPI cut a 563 GAPI response. |
| DEPT | 10720 rows @ 0.5 FT | 3268 rows @ 0.5 M | `resample_to_standard_spacing` | `depth_spacing` resolved as 0.5 from unit FT, then `uniformize_curves` converted depths to metres. Stale parameter applied post-conversion. 3.28× decimation. |

Additionally, and most dangerously:

> `resample_to_standard_spacing` calls `interpolate(method='index', limit_direction='both')` with **no `limit`**. All curves enter with NaN and leave with zero. RHOB left resampling with **3268 fabricated density values** across 3600 ft where no tool ran. Range validation was accidentally masking this.

**Not one of these is a mathematical error.** GR passed cleanly through load, depth standardization, uniformization, resampling and viability. The computation works. The contracts between stages do not exist.

---

## 2. The six contracts

### C1 — Units travel with data

Every curve array carries its unit as a first-class attribute, not as loose metadata that a stage may or may not consult.

- Any stage that transforms values **must** update the declared unit in the same operation.
- Any stage that compares against a reference (range table, threshold, chartbook) **must** assert that the declared unit matches the reference's unit before comparing.
- A unit mismatch is a hard error, not a warning, not a coercion.

*Would have caught:* RHOB. The validator would have thrown on `KG/M3` vs `G/CC` at the first well ever processed.

### C2 — Missing-data provenance survives

"No tool ran at this depth" and "value absent for another reason" are different facts and must remain distinguishable end to end.

- A validity mask travels alongside every curve.
- Interpolation and gap filling **must** record which samples they synthesised.
- No stage may silently convert absent → present. Filling is allowed; filling without recording is not.
- Interpolation limits are mandatory and explicit. Extrapolation beyond the measured interval is forbidden by default.

*Would have caught:* the 3268 fabricated RHOB values.

### C3 — Unknown stays unknown

A stage that does not recognise its input must decline, not substitute.

- `curve_data.get('range', [0.0, 1.0])` and every pattern like it is forbidden. Absence returns absence.
- Confidence 0.00 / type UNKNOWN must propagate as a first-class state, not degrade into a default.
- A stage that cannot act on an unknown skips it and records why. Skipping is a logged, visible outcome — not a silent pass.

*Would have caught:* TBHV, and every other unrecognised mnemonic in the file (ABHV, RxoRt, ITT, MEL15, MEL20, MELCAL are all candidates).

### C4 — Derived parameters invalidate when their source changes

Any parameter computed from data state carries the assumptions it was derived under.

- `depth_spacing` derived from unit FT is invalid the moment depths become metres.
- Either recompute after the state changes, or carry the source assumption and assert it still holds at point of use.

*Would have caught:* the 3.28× decimation.

### C5 — Raw is retained and visible

The original loaded frame is never overwritten.

- Raw and processed are both addressable at all times.
- The UI displays them together — side by side or overlaid — for any curve, at any point.
- A sample of the original file (header + first/last data rows) is viewable without leaving the app.

*Would have caught:* everything in this document, weeks earlier. RHOB emptying is obvious the instant you can see raw next to processed.

### C6 — Processing is explicit, never implicit

Loading a file transforms nothing.

- Load is read-only and lossless. It shows you what is in the file.
- Processing is an invoked action, not a side effect of loading or of opening a tab.
- Before processing runs, the user is shown the list of transformations that will be applied — including unit conversions, resampling target, and which curves will be validated against which ranges.
- No transformation defaults to on. `_uniformize_data` defaulting to `True` is the origin of this entire investigation.

### C7 — Every visualization shows before and after together

No plot displays processed data alone. Raw and processed appear on the same axes, always.

This is C5 made operational — retention is meaningless if comparison requires two windows and a memory. Four conditions make it work; without them the overlay actively misleads:

**C7.1 — Synthesised samples must be visually distinct.** Interpolated, extrapolated and gap-filled segments render differently from measured ones (dashed, lighter, hatched — the choice is cosmetic, the distinction is not).

> This is the condition that matters most. A naive overlay of the fabricated RHOB would have shown raw as a sparse curve over 4250–5326 ft and processed as a complete smooth curve spanning the entire well. It would have looked like an improvement. Without C7.1, the before/after plot makes fabrication *more* convincing, not less.

**C7.2 — Both series plot against actual depth, in a declared common unit — never against row index.** Raw at 10720 rows and processed at 3268 plotted by index would render as a 3.28× depth compression that looks like a real geological shift. The depth axis governs; the row count is irrelevant to position.

**C7.3 — A unit mismatch between the two series is an error surfaced in the plot, not a silent draw.** RHOB raw in G/CC against processed in KG/M3 on a shared value axis is 2.5 versus 2500 — the raw curve flattens to a line at the bottom of the frame. The plot must either convert for display with the conversion stated, or refuse and report the mismatch. **This makes the visualization an enforcement point for C1.**

**C7.4 — Absent data renders as a gap.** Never as zero, never as a sentinel excursion, never as an interpolated bridge drawn silently.

*Would have caught:* all four failures in §1, on sight, on the first well.

---

## 3. Resolution vs transformation — the critical distinction

These are different operations and only one of them is restricted.

| | Definition | Where allowed |
|---|---|---|
| **Transformation** | Changes a *known* value. G/CC → KG/M3. Feet → metres. Denoising. | Anywhere, provided the declaration is updated in the same operation (C1). |
| **Resolution** | Supplies a value for an *unknown*. Picking a range for an unrecognised mnemonic. Choosing a null convention. Assigning a unit to an undeclared curve. | **Only at declared resolution points.** Nowhere else. |

Without this split, C3 would forbid legitimate unit conversion. With it, `create_comprehensive_curve_info` cannot invent `(0.0, 1.0)` as a side effect of a dictionary lookup, because it is not a resolution point.

---

## 4. Resolution points

Resolution points are enumerable, declared, and few. Every place the system must guess is one of them.

Known candidates from this codebase:

1. **Unrecognised mnemonic** → which curve type is this, what range applies, or skip validation
2. **Ambiguous or absent unit** → what unit is this curve in
3. **Null convention conflict** → which declared NULL governs when files disagree
4. **Target unit system** → process in imperial or metric (C6: never assumed)
5. **Resample target spacing** → native increment, or an explicit override
6. **Outlier strategy per curve type** → IQR, physical bounds, or none

### The payoff

**The set of unresolved unknowns is the UI.**

The four scattered unit dialogs and the `Toplevel` popups landing all over the screen exist because resolution currently happens ad hoc, wherever a stage happens to need a value. Centralise resolution and you get one coherent "here is what I do not know — tell me" moment, with before/after preview, instead of four independent windows scattered across 13,867 lines.

The UI problem and the correctness problem have the same root and the same fix.

---

## 5. Provenance manifest

Every processed output carries a manifest. Per curve:

- Final unit, and every unit transformation applied in order
- Sample count: measured vs interpolated vs extrapolated vs filled
- Depth interval actually measured, distinct from the interval spanned
- Range applied, its source, and its declared unit
- Outlier strategy used, bounds computed, count rejected
- Every resolution point hit, what was chosen, and by whom (user or default)

This makes output self-describing. *"This RHOB is 20% measured, 80% interpolated"* is information required before anyone trusts a curve, and nothing in the current pipeline can produce it.

The manifest is also the natural verification surface: raw vs processed, per curve, per stage.

---

## 6. Stage contract template

Every stage declares:

```
Stage: <name>
Requires:
  - units:        <expected declarations, or "any if declared">
  - completeness: <may this stage see NaN? synthesised values?>
  - preconditions:<assertions that must hold on entry>
Guarantees:
  - row count:    <unchanged | may change, and how it is recorded>
  - units:        <unchanged | transformed to X, declaration updated>
  - validity:     <mask unchanged | mask updated, synthesis recorded>
Declines when:
  - <conditions under which this stage skips and logs, rather than acting>
```

A stage that cannot satisfy its own `Requires` fails loudly at that boundary. Localised failure beats a silently empty output 3000 lines later.

---

## 7. Assertions to implement

Minimum enforceable set, in priority order:

1. `assert curve.unit == reference.unit` before any range or threshold comparison
2. `assert interpolation.limit is not None` — no unbounded fill
3. `assert not (confidence == 0.0 and validation_applied)` — unknown cannot be validated
4. `assert derived_param.source_state == current_state` at point of use
5. `assert raw_frame is not processed_frame` — no in-place overwrite
6. `assert transformations_declared == transformations_applied` at end of run

Each one would have thrown on the first well ever processed.

---

## 8. Sequencing

Contract work does not start until the pipeline produces correct output once.

**Phase 0 — repair (do first, in this order)**

1. Add `limit` to the resample interpolation. **Before** fixing range validation — otherwise output goes from obviously empty to plausibly wrong.
2. Gate the unit conversion. Highest-leverage single change: an already-imperial well left alone keeps DEPT in FT, makes the 0.5 spacing correct, preserves 10720 rows, and lets RHOB pass its own range. Fixes C1 and C4 symptoms in one edit.
3. Unknown mnemonic → skip validation, log warning. Never validate against a fallback range.
4. Per-curve outlier strategy. GR should not receive blanket IQR.

**Phase 1 — golden files**

Process KEOUGH. Expect 10720 rows, RHOB populated only over 4250–5326 ft. Capture output as the reference. Add two or three more wells with different unit systems and curve sets.

**Phase 2 — contracts**

Implement C1–C6 and the assertions, validating continuously against the golden files.

**Phase 3 — extraction**

Move modules to the new shell, diffing every stage against golden files. The modules are portable; the orchestration is what is being rebuilt.

**Phase 4 — toolkit and UI**

Deferred deliberately. What the interface can honestly show depends on what the pipeline can honestly promise. Resolution points (§4) define the dialogs.

C7 constrains this choice. Overlaying ~10720 raw against ~3268 processed points per curve, with zoom and pan needed to compare intervals, wants a natively embedded, interactive matplotlib canvas. That favours Tkinter (current, `FigureCanvasTkAgg`) or PySide6 (`FigureCanvasQTAgg`) over a Python-native web UI, where figures render as static images and the interactive toolbar is lost. If C7 is non-negotiable, the toolkit shortlist is effectively Tkinter or Qt.

---

## 9. Open questions

- Are metric wells processed at all? Determines whether the C6 gate is "never convert" or "convert only when source ≠ target."
- What populates the mnemonic database, and how are entries added? C3 makes unrecognised curves visible; something has to resolve them.
- GR minimum drops from 4.03 to **0** across `fill_gaps`. Zero GAPI is not physically achievable. Mechanism unidentified.
- The environmental-corrections error dialog fires on every run and is being dismissed. Contents unknown.
- TBHV collapses to 3.1e-05 in scale-aware denoising. Downstream of Phase 0 items 2 and 3 — recheck after, do not chase now.

---

## 10. What this document does not claim

Every finding in §1 is measured, from instrumented stage-by-stage output on one well. The contracts in §2 are design proposals inferred from those measurements and have not been implemented or tested. The consumer lists and function behaviours are taken from Cursor's reports on the source, not from independent reading of all 13,867 lines.

One well is one well. Phase 1 exists to widen that.
