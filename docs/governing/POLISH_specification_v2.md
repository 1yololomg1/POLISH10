# POLISH — Specification v2

**Supersedes:** `docs/governing/POLISH_pipeline_contracts.md` (v1, August 2026) and Plan of Action v2.
**Date:** 10 September 2026
**Status:** Contracts amended and still unimplemented. Work order revised.

---

## 0. How to read this document

v1 had three overlapping numbering schemes — Steps 1–6, Phases 0–4, and contracts
C1–C7 — which made it unusable as a handoff artifact for both a human and a coding
agent. This document keeps **one** scheme.

- **C1–C7** are the contracts. Numbering preserved from v1 so existing commit
  messages stay traceable.
- **Work order** in §5 replaces both "Steps" and "Phases". The mapping from the
  old numbers is in §5.1 so nothing is silently dropped.
- Amendments made on 10 September are marked **[A-10SEP]** and attributed.

Everything in §1 is measured. Everything in §2 is a design proposal, still
untested. §9 of v1 applies unchanged: one well is one well.

---

## 1. The evidence (unchanged from v1)

KEOUGH #12-34 (`1052970244.las`), traced stage by stage, August 2026.

| Curve | In | Out | Killed by | Root cause |
|---|---|---|---|---|
| RHOB | 2169 real values, G/CC | **0** | `apply_range_validation` | `uniformize_curves` converted G/CC → KG/M3 (×1000); validator compared 2180–3247 against a g/cc range of (0.5, 7.0) |
| TBHV | 9418 real values, FT3 | **8** | `apply_range_validation` | Mnemonic unrecognised; `create_comprehensive_curve_info` returned fallback `[0.0, 1.0]`; type UNKNOWN, confidence 0.00, validated anyway |
| GR | 10631 real values, GAPI | 3268, peak clipped | `detect_outliers_iqr` | Symmetric IQR on a right-skewed distribution; upper bound 121.46 GAPI cut a 563 GAPI response |
| DEPT | 10720 rows @ 0.5 FT | 3268 rows @ 0.5 M | `resample_to_standard_spacing` | `depth_spacing` resolved as 0.5 under FT, then depths converted to metres; stale parameter applied post-conversion; 3.28× decimation |

And the dangerous one: `interpolate(method='index', limit_direction='both')` with
no `limit`. RHOB left resampling with **3268 fabricated density values across
3600 ft where no tool ran.** Range validation was accidentally masking it.

None of these is a mathematical error. The computations work. The contracts
between stages do not exist.

### 1.1 Found 10 September, not in v1 **[A-10SEP]**

Read directly from source. All three are existing-contract violations, not new
categories.

| Finding | Location | Violates |
|---|---|---|
| `_uniformize_data` looks for a column literally named `DEPT`; on a well indexed on `MD` or `DEPTH`, resampling silently does not run at all | `advanced_preprocessing_system10.py` `_uniformize_data` | C3 — silent pass |
| `_validate_and_standardize_depth` catches the depth-identification failure, logs "Continuing with existing depth reference…", and proceeds on an unvalidated depth column | same file, `_validate_and_standardize_depth` | C3 — silent pass, on the most load-bearing curve in the file |
| `standardize_depth_reference` injects a `DEPTH_PRIMARY` column into the frame; nothing reads it except a blacklist entry in `normalize_processed_data`; it rides through resampling and inflates the curve count | same file, `standardize_depth_reference` | §7 assertion 6 — undeclared transformation |
| Depth spacing is logged and displayed with a hardcoded `m` in at least two places plus the UI, while `_sync_depth_spacing_default` sets 0.5 for imperial wells; the depth-aware parameter block also names its outputs `*_meters` | same file + `ui/processing_tab.py` | C4-adjacent — the unit assumption that caused the 3.28× decimation is systemic, not local |

Header `STRT`, `STOP`, `STEP` are already parsed into `well_info` by
`core/well_loaders.py`, so C6.1 below needs no new header plumbing. That same
dict defaults `depth_unit` to `'m'`.

---

## 2. The seven contracts

### C1 — Units travel with data

Every curve carries its unit as a first-class attribute, not loose metadata a
stage may or may not consult.

- A stage that transforms values **must** update the declared unit in the same
  operation.
- A stage that compares against a reference **must** assert the declared unit
  matches the reference's unit first.
- A unit mismatch is a hard error. Not a warning, not a coercion.

*Would have caught:* RHOB, on the first well ever processed.

### C2 — Missing-data provenance survives

"No tool ran here" and "value absent for another reason" are different facts and
must stay distinguishable end to end.

- A validity mask travels alongside every curve.
- Interpolation and gap filling **must** record which samples they produced.
- No stage may silently convert absent → present. Filling is allowed; filling
  without recording is not.
- Interpolation limits are mandatory and explicit. Extrapolation beyond the
  measured interval is forbidden by default.

**[A-10SEP] Three provenance states, not two.** v1 treated every produced value
as synthesised. That conflates two different acts:

| State | Meaning | Example |
|---|---|---|
| **Measured** | An instrument recorded this | RHOB inside the density-tool interval |
| **Calculated** | Deterministic function of measured data, re-evaluable at any spacing | TVD from MD via the deviation survey |
| **Synthesised** | Supplied where no measurement exists | RHOB across 3600 ft where no tool ran |

Only *synthesised* is fabrication. Re-evaluating TVD onto a new grid invents
nothing. If both render identically under C7.1, every deviated well ships
covered in warnings and the warning stops carrying information.

*Origin:* your father's instruction that TVD is resampled with everything else.
The three-state consequence is inferred from it, not stated by him.

*Would have caught:* the 3268 fabricated RHOB values.

### C3 — Unknown stays unknown

A stage that does not recognise its input declines. It does not substitute.

- `curve_data.get('range', [0.0, 1.0])` and every pattern like it is forbidden.
  Absence returns absence.
- Confidence 0.00 / type UNKNOWN propagates as a first-class state.
- A stage that cannot act on an unknown skips it and records why. **Skipping is
  a logged, visible outcome, never a silent pass.**

**[A-10SEP] Promoted from contract to product requirement.** For an internal
tool you add mnemonics as you meet them. For a tool sold to professionals
generally, the unrecognised path is the *common* path, because every buyer has
vendor naming you have never seen. One KEOUGH well already produced ABHV,
RxoRt, ITT, MEL15, MEL20 and MELCAL. C3 plus resolution point 1 in §4 is
therefore not a safety net — it is the mechanism by which the product works on a
stranger's wells, and the mechanism by which the mnemonic database grows.

*Would have caught:* TBHV and every other unrecognised mnemonic in the file.

### C4 — Derived parameters invalidate when their source changes

Any parameter computed from data state carries the assumptions it was derived
under.

- `depth_spacing` derived under FT is invalid the moment depths become metres.
- Either recompute after the state changes, or carry the source assumption and
  assert it still holds at point of use.

**[A-10SEP]** The same failure appears in *labelling*, not only in computation.
Reporting a foot-indexed well's spacing as metres is C4 leaking into the log and
the UI, and it will contaminate any reference output that captures the log.

*Would have caught:* the 3.28× decimation.

### C5 — Raw is retained and visible

The original loaded frame is never overwritten.

- Raw and processed are both addressable at all times.
- The UI displays them together for any curve, at any point.
- A sample of the original file (header + first/last data rows) is viewable
  without leaving the app.

*Would have caught:* everything in this document, weeks earlier.

### C6 — Processing is explicit, never implicit

Loading a file transforms nothing.

- Load is read-only and lossless.
- Processing is an invoked action, not a side effect of loading or of opening a
  tab.
- Before processing runs, the user is shown the list of transformations that
  will be applied — unit conversions, resampling target, and which curves will
  be validated against which ranges.

**[A-10SEP] The fourth bullet of v1 is withdrawn.** It read: *"No transformation
defaults to on. `_uniformize_data` defaulting to `True` is the origin of this
entire investigation."*

It over-generalised from its own evidence. Renaming GRD→GR did not decimate
KEOUGH. What decimated KEOUGH was resampling firing *because* rename was on,
with a spacing derived under FT applied after conversion to metres. The evidence
supports "no transformation runs undeclared, and no switch silently activates a
different transformation." It does not support "renaming must default off" —
which would ship a mnemonic-uniformisation tool that does not uniformise
mnemonics.

Replaced by:

> **C6a — No switch activates a transformation other than its own.** Rename does
> not trigger resampling. Unit conversion does not trigger resampling. Each
> transformation is gated on its own control and nothing else.
>
> **C6b — Defaults are declared, not hidden.** A transformation may default on
> if and only if it is non-destructive and appears in the pre-run summary.
> Renaming qualifies once C6c holds. Resampling and unit conversion do not, and
> default off.
>
> **C6c — Standardisation adds, it does not replace.** Where a mnemonic is
> non-standard, the original curve is retained under its original name and a
> copy is added under the standard name. Nothing the file arrived with is lost.
> *Decided by your father, 10 September, for the depth curve. Whether it extends
> to log curves is open — see §6.*

C6c is what makes C6b safe: a copy destroys nothing, so there is nothing to undo.

**C6.1 — Depth index must agree with the header. [A-10SEP]**
If the depth curve does not honour the `STRT`, `STOP` and `STEP` recorded in the
well section, POLISH refuses that well, reports why, and continues to the next
one in the batch. *Decided by your father, 10 September.* This is the first
concrete `Declines when:` clause POLISH has, and the header values are already
parsed (§1.1).

Note it does **not** resolve KEOUGH. `BLI1` is a logging-run marker, not
`STRT`/`STOP`/`STEP`, so KEOUGH passes C6.1 and the eight feet remains open.

### C7 — Every visualization shows before and after together

No plot displays processed data alone. Raw and processed on the same axes,
always. C5 made operational. Four conditions, without which the overlay actively
misleads:

**C7.1 — Produced samples are visually distinct.** Interpolated, extrapolated
and gap-filled segments render differently from measured ones.

> The condition that matters most. A naive overlay of the fabricated RHOB shows
> raw as sparse over 4250–5326 ft and processed as complete and smooth across
> the whole well. It looks like an improvement. Without C7.1 the before/after
> plot makes fabrication *more* convincing, not less.

**[A-10SEP]** Per C2, this distinguishes three states. *Calculated* curves are
marked as such but must not carry the same alarm weight as *synthesised* ones.

**C7.2 — Both series plot against actual depth in a declared common unit, never
against row index.** 10720 rows against 3268 plotted by index renders as a 3.28×
depth compression that looks geological.

**C7.3 — A unit mismatch between the two series is an error surfaced in the
plot, not a silent draw.** G/CC against KG/M3 on a shared axis is 2.5 versus
2500; the raw curve flattens to the frame bottom. Convert for display with the
conversion stated, or refuse and report. **This makes the visualization an
enforcement point for C1.**

**C7.4 — Absent data renders as a gap.** Never zero, never a sentinel
excursion, never a silently drawn bridge.

*Would have caught:* all four failures, on sight, on the first well.

---

## 3. Resolution vs transformation (unchanged)

| | Definition | Where allowed |
|---|---|---|
| **Transformation** | Changes a *known* value. G/CC → KG/M3. Feet → metres. Denoising. | Anywhere, provided the declaration updates in the same operation (C1) |
| **Resolution** | Supplies a value for an *unknown*. Picking a range for an unrecognised mnemonic. Choosing a null convention. Assigning a unit to an undeclared curve. | **Only at declared resolution points** |

Without the split, C3 would forbid legitimate unit conversion. With it,
`create_comprehensive_curve_info` cannot invent `(0.0, 1.0)` as a side effect of
a dictionary lookup, because it is not a resolution point.

---

## 4. Resolution points — and why they are the interface

Every place the system must guess is enumerable and declared.

1. **Unrecognised mnemonic** → which curve type, what range, or skip validation
2. **Ambiguous or absent unit** → what unit is this curve in
3. **Null convention conflict** → which declared NULL governs when files disagree
4. **Target unit system** → imperial or metric, never assumed
5. **Resample target spacing** → native increment, or explicit override
6. **Outlier strategy per curve type** → IQR, physical bounds, or none
7. **[A-10SEP] Non-standard depth mnemonic** → confirm the standard name added
   alongside it (C6c)

### The payoff

**The set of unresolved unknowns is the UI.**

The four scattered unit dialogs and the `Toplevel` popups landing all over the
screen exist because resolution happens ad hoc, wherever a stage needs a value.
Centralise resolution and you get one coherent *"here is what I do not know —
tell me"* moment with before/after preview, instead of four windows across
13,867 lines.

The UI problem and the correctness problem have the same root and the same fix.

**[A-10SEP]** The pre-run summary required by C6 is the first half of this and
is currently scheduled nowhere. It is the same data as the §5 manifest, shown
before instead of written after. Build it once, use it twice.

---

## 5. Provenance manifest

Every processed output carries a manifest. Per curve:

- Final unit, and every unit transformation applied in order
- Sample count: **measured / calculated / synthesised** **[A-10SEP]**, and within
  synthesised, interpolated vs extrapolated vs filled
- Depth interval actually measured, distinct from the interval spanned
- Range applied, its source, and its declared unit
- Outlier strategy used, bounds computed, count rejected
- Every resolution point hit, what was chosen, and by whom (user or default)
- **[A-10SEP]** Every standard-name copy added under C6c, and the original it
  came from

*"This RHOB is 20% measured, 80% synthesised"* is information required before
anyone trusts a curve, and nothing in the current pipeline can produce it.

**[A-10SEP] The manifest is a product surface, not a debugging aid.** Processed
LAS files leave the building and go to professionals who did not run the
processing. The manifest is what makes a delivered curve defensible, and it is
the difference between a script and something that can be sold.

---

## 6. Stage contract template (unchanged)

```
Stage: <name>
Requires:
  units:         <expected declarations, or "any if declared">
  completeness:  <may this stage see NaN? produced values?>
  preconditions: <assertions that must hold on entry>
Guarantees:
  row count:     <unchanged | may change, and how it is recorded>
  units:         <unchanged | transformed to X, declaration updated>
  validity:      <mask unchanged | mask updated, production recorded>
Declines when:
  <conditions under which this stage skips and logs, rather than acting>
```

A stage that cannot satisfy its own `Requires` fails loudly at that boundary.
Localised failure beats a silently empty output 3000 lines later.

---

## 7. Assertions to implement

1. `assert curve.unit == reference.unit` before any range or threshold comparison
2. `assert interpolation.limit is not None` — no unbounded fill
3. `assert not (confidence == 0.0 and validation_applied)`
4. `assert derived_param.source_state == current_state` at point of use
5. `assert raw_frame is not processed_frame`
6. `assert transformations_declared == transformations_applied` at end of run
7. **[A-10SEP]** `assert depth_column is validated_depth_column` — no stage
   reaches for a hardcoded depth mnemonic (§1.1, finding 1)

Each would have thrown on the first well ever processed.

---

## 8. Work order

Replaces Steps 1–6 and Phases 0–4. One list.

### 8.1 Mapping from the old numbering

| v1 | Here |
|---|---|
| Phase 0 items 1–2, Step 1, Step 1b | **Complete** — see 8.2 |
| Step 3 (decouple switches) | **Stage A**, scope expanded |
| Phase 0 items 3–4 | **Stage C** |
| Step 2 / Phase 1 (golden files) | **Stage B**, restructured |
| Step 4 / Phase 2 (contracts) | **Stage D** |
| Step 5 / Phase 3 (extraction) | **Stage E** |
| Step 6 / Phase 4 (UI) | **Stage F** |

### 8.2 Complete

Interpolation no longer extrapolates past the logged interval (`limit_area='inside'`,
`c126a62`). Unit conversion defaults off at all three definition sites (`3e0372a`).
Internal null is NaN, sentinels only at export. Unknown curves skip range
validation; the `[0.0, 1.0]` fallback is gone. GR outliers use physical bounds.
Headless load uses the new file's NULL. GR in CPS is skipped rather than treated
as GAPI. Ten tests in `tests/test_quality_gates.py`, `pyproject.toml`,
pre-commit, and CI with xvfb and coverage.

PR #6 remains unmerged, deliberately.

### Stage A — depth identity and switch separation **(next)**

Everything here is decided and unblocked. All of it is in one region of one file.

1. Write the standard-named depth copy under a real mnemonic (`DEPT`) instead of
   `DEPTH_PRIMARY`, retaining the original. Exports unchanged for now (§9).
2. Ensure the copy is made before any resample call, so the existing `DEPT`
   lookup becomes correct rather than hardcoded-and-lucky.
3. Give resampling its own control, default off. Rename stays default on. Unit
   conversion stays default off. (C6a, C6b)
4. Halt and report when the depth column cannot be identified at all, instead of
   continuing quietly. (C3)
5. Log and display the actual depth unit everywhere spacing is reported. (C4)

Not in Stage A, pending decisions in §9: the C6.1 header check, and any change
to interpolation limits.

### Stage B — reference wells

Restructured. **[A-10SEP]** KEOUGH is no longer the reference file.

- **B1 — KEOUGH as a regression fixture, not a frozen output.** Assert its four
  known failures do not return: RHOB retains ~2169 real values, TBHV ~9418, GR
  peak reaches 563 GAPI, DEPT holds 10720 rows at 0.5 ft. Eight feet at half-foot
  spacing is sixteen samples, so tolerances absorb the open question and KEOUGH
  becomes useful before it is resolved.
- **B2 — a frozen reference file from an unremarkable well.** Standard
  mnemonics, one unit system, header agreeing with data, no meaningful gaps.
  Something where "is this correct" has an obvious answer.
- **B3 — coverage wells**, in priority order: metric; non-standard depth
  mnemonic; genuine interior gap (required before an interpolation limit can be
  chosen sensibly); unrecognised vendor mnemonics; deviated with TVD; old file
  with blank or zero `STEP`.
- **B4 — two reference runs per well**, not one: *all transformations off*
  (immutable) and *shipped defaults* (moves when the mnemonic database moves,
  and its diff is reviewed rather than ignored). One file cannot do both jobs.

While here, note the environmental-corrections dialog and the GR-to-zero
gap fill (§9). Do not chase them.

### Stage C — remaining Phase 0 repair

Per-curve outlier strategy beyond GR. Interior-gap interpolation limit, which
needs B3.

### Stage D — implement the contracts

C1–C7 and the assertions, validated continuously against Stage B. Includes the
pre-run summary and the manifest, which share an implementation.

### Stage E — extraction

Move modules to the new shell, diffing every stage against reference files. The
modules are portable; the orchestration is what is being rebuilt. Extraction
without Stage B has already dropped curves in transit.

### Stage F — toolkit and UI

Deferred deliberately. What the interface can honestly show depends on what the
pipeline can honestly promise. Resolution points (§4) define the dialogs.

C7 constrains the choice: overlaying ~10720 raw against ~3268 processed points
per curve, with zoom and pan, wants a natively embedded interactive matplotlib
canvas. That favours Tkinter (`FigureCanvasTkAgg`) or PySide6
(`FigureCanvasQTAgg`) over a Python-native web UI, where figures render as
static images and the interactive toolbar is lost. If C7 is non-negotiable, the
shortlist is Tkinter or Qt.

---

## 9. Open questions

**Blocking Stage B2:**

- **The eight feet.** `BLI1` says 5326 ft; the density tool reads to 5334 ft.
  Which governs a curve's extent — the header entry, or where the data actually
  is? *With your father.* Confirm also which header section `BLI1` sits in;
  this document assumes the parameter section, from the mnemonic, not from the
  file.

**Blocking C6.1:**

- **Tolerance, and blank `STEP`.** How close counts as matching header to data.
  `STEP: 0` for irregular sampling is legal LAS 2.0 and common in older files. A
  strict check rejects real wells, which for a sold tool is worse than the
  problem it fixes.

**Blocking nothing yet, but decide before Stage D:**

- **Does a delivered LAS carry two depth curves?** C6c adds a standard-named
  copy. Two depth curves in a delivered file is legal but will confuse some
  receiving software, and under C1 both copies must convert together or two
  columns holding the same measurement disagree. Alternative: keep the copy
  internal and record the mapping in the manifest. Stage A preserves current
  export behaviour so this is not decided by accident.
- **Does C6c extend to log curves?** If a gamma arrives as SGRC, is SGRC
  retained with GR added alongside, or renamed? Changes the scope of the whole
  standardisation design. *With your father.*
- **Can client logs go in the repo?** Currently one synthetic LAS is committed
  (`tests/fixtures/new_file_null_minus_999.las`); nothing real. Real headers
  carry well name, operator, API and location, and the KEOUGH filename is itself
  an API number. Either strip identifying header fields — no test needs them —
  or use publicly released logs, which is strictly better for a product you
  intend to ship, demo and run in public CI.
- **What populates the mnemonic database, and how are entries added?** Per C3
  this is now a product question, not a housekeeping one. Known trap:
  `BULK_DENSITY` lists both `G/CC` and `KG/M3` as valid units but carries a
  single `'range': [1.0, 3.5]` (`curve_identification.py`).

**Parked investigations** — do not chase before Stage B:

- GR minimum drops from 4.03 to 0 across `fill_gaps`. Zero GAPI is not
  physically achievable. Mechanism unidentified.
- The environmental-corrections error dialog fires every run and is dismissed
  unread.
- TBHV collapses to 3.1e-05 in scale-aware denoising.

**Housekeeping:** merge PR #6 once Stage A lands; delete stale `origin/cursor/*`
branches; Semgrep is not in CI;
`apply_comprehensive_data_quality_validation` is tested but never called from
production; two pre-existing undefined `wid` names are silenced in ruff rather
than fixed.

---

## 10. What this document does not claim

§1 is measured, from instrumented stage-by-stage output on one well. §1.1 is read
directly from source on 10 September. The contracts in §2 are design proposals
inferred from those measurements and have not been implemented or tested. The
three-state provenance model in C2, the two-reference-run scheme in B4, and the
promotion of C3 to a product requirement are reasoning from decisions, not
decisions themselves — they are the most likely places this document is wrong.

One well is one well. Stage B exists to widen that.
