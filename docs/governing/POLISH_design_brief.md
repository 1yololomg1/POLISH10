# POLISH — Design Brief

**Handoff document. Prepared 11 September 2026.**

You are being asked to design the interface for POLISH, a petrophysical well log
preprocessor. This document gives you the product context, the settled
constraints, and the open questions. It deliberately does not contain visual
direction — that is your job.

Read the "What is already decided" section as fixed. Those decisions came from
specific failures in a previous version and from the product's core claim. If
something in there seems wrong, say so explicitly rather than designing around
it.

---

## 1. What POLISH is

A desktop application that takes well log files (LAS format), describes their
actual condition, and optionally repairs them — standardising curve
identification, classifying gaps, filling or declining to fill them, and
exporting a processed file.

**User:** a petrophysicist. Domain expert, reads composite logs by reflex, deeply
sceptical of automated processing, and correct to be. Not a casual user. Will
abandon a tool that wastes their time and will never trust one that has surprised
them once.

**Market context:** competing against Techlog, Interactive Petrophysics, Geolog —
mature, expensive, and all weak in one specific way (see §3).

**Deployment:** fully isolated desktop program. No network calls of any kind. No
cloud storage. No account. No telemetry. The developer holds none of the
customer's data, ever, by design — well logs are commercially sensitive and often
under statutory confidentiality. Anything the app writes goes to the user's own
disk and stays there.

---

## 2. Why the previous version is being rewritten

Two reasons, and they turned out to be the same reason.

**The interface was a control panel.** Checkboxes for fill-gaps, denoise,
resample; sliders for thresholds; a run button. The user had to configure the
pipeline *before* being told anything about the well — which is backwards,
because finding out what is wrong with the file is why they opened the tool.

**It shipped a fabrication.** The pipeline invented bulk density values across an
interval where the density tool had never been run. Thousands of samples of
plausible-looking, entirely fictional rock.

Those connect. The fabrication was invisible **because nothing in the interface
distinguished a measured sample from an invented one.** The code even classified
gaps correctly — it labelled the interval as a logging-tool absence — and then
ignored its own label, because no screen ever displayed it.

The conclusion that governs this design: **where honesty is the product, the
interface is not a layer over the method. It is the verification mechanism, and
the only one the user will ever run.**

---

## 3. The product claim

POLISH's claim is per-sample provenance that survives into the exported file, and
a report that itemises refusals as prominently as actions. It tells you which
samples were processed, by what method, from what inputs, and what it declined
to do and why.

The single most valuable interaction in the product is hovering an empty stretch
of a curve and reading:

> *Not filled — density tool absent above 4,250 ft.*

If the design makes that moment feel authoritative, the product works.

---

## 4. What is already decided

### 4.1 The original is immutable

The raw data is loaded once and never altered. **OD-03 split:** resampling
produces a **child dataset** (a new grid, with a recorded mapping to the parent).
Fills, derivations and smoothing remain **overlays** on the active dataset,
keyed to sample index. The original input is never written (D85).

Consequence for design: the unmodified curve is **always renderable**. Not a
history feature, not an undo — a permanently available layer. After resampling,
that layer lives on the parent dataset; the child is a separate object.

### 4.2 Sample status (D44), and the three-phrase Refused vocabulary (OD-04)

D44 sample statuses — these are not interchangeable:

- **Measured** — an instrument recorded this.
- **Created** — no measurement existed at this depth. Gap fill, derived curve.
  Toggling created off leaves a visible hole, which is the honest signal.
- **Changed** — a measurement existed and POLISH altered it. Smoothing,
  despiking. Two values at one depth. Unit conversion is **not** Changed
  (OD-05): stored numbers never change; the unit travels with the number.
- **Invalid** — measured, outside physical bounds, excluded from use. Must remain
  visible and marked. A sample POLISH refused to trust is information.
- **Absent** — no value at this depth (tool absence, dropout, or left out of the
  processed file).
- **Resampled** — the value exists on a child-dataset grid produced by Stage 5.

**Created and changed must look different.** Changed is the more serious claim —
contradicting a measurement rather than extending past one.

"Refused" is **not** a sample status. Per OD-04 Appendix A it splits into three
phrases, used as issue or method outcomes:

- **gap not filled**
- **declined by POLISH**
- **method not run**

### 4.3 Colour is already spoken for

GR is green or black. RHOB red. NPHI blue. Resistivity black and red on a log
scale. Every petrophysicist reads those by reflex.

**Colour encodes which curve. Provenance encodes on everything else.** Rendering
a filled GR in orange breaks curve identity to signal something that applies to a
small minority of samples.

Provenance must therefore encode **redundantly and multi-scale**:

| Zoom level | What carries the distinction |
|---|---|
| Sample-level | Marker shape (measured = filled circle) |
| Mid | Line style (solid / dashed / dash-dot) |
| Full-well | Background interval band, hatched |

The band is load-bearing: it is the only encoding that survives a full-well
screenshot and the only one that is reliably colourblind-safe. Roughly 8% of men
have a colour vision deficiency and composite logs are printed in greyscale
constantly. **Colour-only distinction is not acceptable** — a screenshot where
the only difference is hue is a tool that produces innocent-looking
fabrications.

### 4.4 Decimation rule

At full-well scale, ~20,000 samples render into ~300 pixels. Decimation is
unavoidable. **If any sample within a rendered pixel is created or changed, that
pixel renders as created or changed.** Never average measured and synthesised
into one mark.

Otherwise a three-sample fill vanishes when zoomed out and reappears when zoomed
in — the worst possible behaviour, because the tool looks clean until someone
looks closely.

### 4.5 Provenance is visible by default

The created/changed overlay is **on** when a file opens, with markings shown.
Hiding it requires an action. Seeing the truth is never the thing you have to
switch on.

Two view states only: original-only, and original-plus-overlay. **No created-only
view** — that is the fabrication rendered as fact and has no legitimate use.

A *difference* view (plotting the delta for changed samples, so the user can see
how far smoothing moved a curve) is worth having and nobody ships it.

### 4.6 Hover gives the derivation, not just the value

Hovering a created or changed sample shows the per-sample slice of its
regeneration record: depth, value, status, method, the interpolation domain, and
the actual inputs.

For a linear fill that reads as:

> *2.31 g/cc — interpolated (linear), between 2.28 @ 4,412.5 ft and 2.34 @
> 4,415.0 ft*

The user can verify the arithmetic in their head on the spot.

**Important honesty constraint:** some methods have no meaningful uncertainty
value. Linear interpolation between two measurements has no intrinsic error bar —
any number shown would be invented, which is the exact thing the product exists
to avoid. The uncertainty field must be able to read **"not quantifiable for this
method."** An honest blank beats a plausible number.

### 4.7 Refusal is a first-class outcome

A well POLISH declines to process, or a gap it declines to fill, is a **correct
result** and must be presented as one — not as an error, not as a warning to
dismiss. There is a refusals panel and it is a selling point.

### 4.8 Disposition gating

This is the mechanism that makes "everything gets checked" structurally true
rather than a thing a user has to remember.

Every finding is classified by **block type (D86)** before any action is offered:

- informational discrepancy (dispositionable)
- ambiguous interpretation (requires explicit user choice, rationale recorded)
- rule-dependent warning (explicit labelled user mapping only)
- scientific precondition failure (method declined, e.g. log fill with a
  non-positive endpoint)
- structural data failure (workflow or file refused)
- export conformance failure (fix or do not export)

**Structural data failures and export conformance failures are not four-state
disposition items and have no override.** There is no general "force export"
button.

For findings that *are* dispositionable, every item must reach one of exactly
four terminal states before export is permitted:

- **Resolved** — user decided, decision recorded
- **Accepted** — default applied, user saw it
- **Deferred** — explicitly postponed, with a reason
- **Refused** — POLISH declined, reason recorded (the three OD-04 phrases above)

Export is gated on every *dispositionable* item having a disposition. It is
**not** gated on every item being fixed. The user is not required to do the
work; they are required to have been shown it.

Silence is not one of the four states, so nothing dispositionable can be
silently unaddressed.

### 4.9 Staleness must be visible

Real work is not linear — a user will change their mind about units after seeing
the gaps. Stages are always visible with status; any stage whose preconditions
hold can be re-entered; and changing an upstream decision marks every dependent
downstream stage **stale** rather than quietly leaving it inconsistent.

Stale is a fifth visual state and it blocks export until re-run or
re-dispositioned. This is a parametric-CAD feature tree, not a wizard.

The failure mode being designed against: adjusting one setting and having no idea
what is now internally inconsistent.

---

## 5. The pipeline

Stages 0–10 (eleven stages). The ordering principle: **describe the file
completely before altering any part of it; then alter minimally with consent,
one recorded step at a time; then report everything including refusals.**

Stages 0–4 touch no data. Stages 5–8 alter data, each independently switchable
and each **defaulting to off**. Stages 9–10 report and export.

| # | Stage | Notes for design |
|---|---|---|
| 0 | Ingest | Malformed-file handling. Produces an audit ledger of every ambiguity found while parsing. Hardest stage in the system. |
| 1 | Depth axis | Establishes the shared vertical axis. May **refuse the well** if header and data disagree. |
| 2 | Identification | Curve naming. Produces family + qualifiers per curve. Source of most resolution-queue items. |
| 3 | Units | Conversion, explicit and recorded. |
| 4 | Validity | Per-sample status mask, gap inventory, gap classification. **This is what the coverage track renders.** |
| 5 | Resampling | Off by default. Creates samples that were never measured. |
| 6 | Derivation | Whole missing curve from measured curves by a stated equation. |
| 7 | Gap filling | Off by default, consented per gap. Tool-absence gaps are never fillable. |
| 8 | Denoising | Last, because it destroys information. Produces *changed* samples. |
| 9 | Manifest | Full provenance record. Human-readable and machine-readable. |
| 10 | Export | Two files: original untouched, processed written separately. |

---

## 6. Data model facts that affect the interface

**Curve identity is column index, not name.** LAS files legitimately contain
duplicate mnemonics — two curves both named `GR`. Names are not unique keys.

**Original mnemonics are never overwritten.** `RDEP` goes in, `RDEP` comes out.
POLISH's family assignment is an annotation alongside, because renaming publishes
POLISH's interpretation as if it were the file's own fact — and if the
identification is wrong, that error becomes permanent in a file the customer
trusts.

**Identification output is family + qualifiers**, not one label:

- `GR_MAIN` → GR, qualifier: main pass
- `GR_REF` → GR, qualifier: repeat pass
- `LFP_DT_SYNT` → DT, qualifiers: vendor suite LFP, synthetic
- `AT90` → induction resistivity, qualifier: 90-inch depth of investigation

**Curve roles** — the interface must distinguish these, because they are
processed differently:

- `INDEX` — the depth curve
- `MEASURED_LOG` — real wireline measurement
- `DRILLING` — real but not wireline (`ROP`); never range-validated as a log
- `ANCILLARY` — coordinates, facies labels, QC flags; loaded and displayed,
  excluded from processing
- `SYNTHETIC_SOURCE` — vendor-computed (`LFP_DT_SYNT`); never treated as a
  measurement, never used as a derivation input
- `UNKNOWN` — declines downstream processing rather than guessing

**Match confidence has tiers** — Cited (traceable to a standard), Observed (seen
in a real file with a corroborating unit), User-defined, Proposed. A
lower-tier match does not unlock processing it has not earned, and the tier
should be legible in the interface.

**Sessions are keyed by a hash of the input file**, not by well name. A filename
match with a hash mismatch is a resolution point — *"this looks like the same
well but the raw file has changed since these decisions were made"* — never a
silent application of stale settings.

---

## 7. Layout and rhythm — the settled principles

### 7.1 Depth is the spine

The one spatial fact in this domain is that depth is vertical and shared.
Everything in the main area hangs off a single vertical axis and nothing competes
with it.

- **Coverage track** — permanent column. The map of the whole file. Rendered from
  Stage 4. Shows measured intervals, interior gaps, tool-absence shoulders, and
  refusals. This is the landing view and requires no processing and no settings.
- **Log tracks** — domain expert's default order, left to right: DTP, DTS,
  density, porosity, Sw, GR, ILD. When a default curve is not in the file, use
  the nearest equivalent and label it clearly (follow-up Q5). The list of
  acceptable ILD substitutes was asked twice and left blank both times; it is
  not invented here.
- **Resolution queue** — side panel. Never covers the depth view. Never modal.
- **Detail** — anchored in reserved space.

**Nothing that appears may move anything else.** Reflow makes software feel
unstable. Panels reveal into space already held for them. Once a well is on
screen, positions are fixed for the session.

### 7.2 Two densities, one file

The coverage track shows **all** curves — it is compact enough to. The log
display shows a **chosen subset** — roughly eight tracks is the practical limit.
These are two views of the same file at different densities, and the selection
mechanism between them is real design work.

### 7.3 Rhythm — many short beats, each with visible output

The old tempo was one long beat: configure everything, press go, wait, get a
result, unable to tell what happened.

The tempo that works:

1. **Load → description, immediately.** No settings asked first. Achievable
   because Stages 0–4 touch nothing.
2. **A scan phase with no interaction.** Reading the coverage track. This is
   where the tool earns trust and it asks nothing of the user — which is why it
   must come before anything asks them a question.
3. **Decisions, one small beat each**, worked from the queue at the user's pace.
4. **Export, deliberately.**

### 7.4 Queue items share one shape

Every queue item has the same structure — **evidence, options, consequence,
disposition** — with identical layout, identical controls, identical screen
position. Repetition builds fluency; after three items the user knows how all of
them work.

Variation in queue-item layout is what would make a large queue unbearable.

### 7.5 Friction proportional to consequence

**Reversible actions are instant and unceremonious** — toggling the overlay,
zooming, changing a disposition, switching curve selection. No confirmations, no
waiting. These are how someone explores and they must feel weightless.

**Irreversible actions get a deliberate beat** — writing a file, committing an
export. Slower, confirmed, summarised.

Getting this backwards produces the worst of both: a confirmation dialog on a
toggle, and a one-click export you cannot take back.

### 7.6 Progressive disclosure, three depths, none mandatory

- **Track** — the glance. Banded intervals, dashed segments, hatched refusals.
  State with no reading required.
- **Annotation** — the read. Hover: value, status, method.
- **Record** — the inspect. Full regeneration entry with inputs and parameters.

Nobody has to open the third layer to trust the first. That is what makes rigour
cheap to carry: the detail exists for whoever wants it and costs nothing for
whoever doesn't.

---

## 8. The stress test

A real test fixture in this project contains **171 curves**, of which 147 share a
vendor prefix. Another contains 16 curves at 22% data fill. A third is 12 MB.

**Design against the 171-curve file, not the 16-curve one.** Any interface that
prompts per curve produces 171 dialogs. Any interface that requires scrolling 171
rows to confirm nothing was missed will not be used.

Two consequences that are requirements, not refinements:

- **Batch disposition.** *"147 curves share the `LFP_` prefix — strip and
  re-match, accept all"* must be one action.
- **Keyboard traversal.** Next, previous, accept, refuse, defer, without the
  mouse. Someone working a file all afternoon settles into keys.

---

## 9. Anti-patterns

Each of these was in the previous version or is a known failure in this category:

- A checkbox/settings panel as the primary surface
- Modal dialogs during processing
- A progress bar as the only feedback — it says something is happening, not what
- Settings that persist invisibly between wells
- Any "auto" mode whose decisions cannot be inspected afterwards
- Per-curve prompting of any kind
- Colour as the sole carrier of provenance
- Renaming curves to canonical names in the output file
- A created-only view

---

## 10. Open questions for you to resolve

1. **The coverage track's visual language.** All curves, full well, measured vs
   interior gap vs tool absence vs refused, legible at a glance and in greyscale.
   This is the most important screen in the product.
2. **Curve selection** between the all-curves coverage track and the ~8-track log
   display. At 171 curves this is non-trivial.
3. **Two values at one depth** — how a changed sample and its original render
   together without looking like a plotting error.
4. **Where staleness lives.** Per-stage indicator, global banner, both?
5. **The queue's information hierarchy** within the fixed item shape.
6. **How the refusals panel earns its prominence** without reading as a list of
   failures.
7. **Batch disposition affordance** — how a user sees that 147 items are about to
   be handled identically, and what they are.
8. **Density.** This user reads dense composite logs professionally. Generous
   whitespace may read as unserious. Unclear where the line is.

## 11. What is not yours to decide

Do not redesign these; they are settled for reasons outside interface concerns:

- The ten-stage pipeline and its ordering
- Which stages default off (all data-altering ones)
- Disposition gating and the four terminal states
- Original immutability and the additive overlay model
- Preservation of original mnemonics in output
- The requirement that provenance survive into the exported file
- No network, no cloud, no account

---

## 12. Technical constraint worth flagging

The existing application is Python with a Tkinter interface. Much of what is
described above — layered plots with per-sample hit-testing, non-reflowing
panels, a fixed-shape queue with keyboard traversal, multi-scale rendering with
correct decimation — is awkward in Tkinter and natural in Qt (PySide6).

**This has not been decided.** If your design depends on capabilities Tkinter
does not have, say so explicitly and early, because the toolkit choice should be
driven by the interface requirements rather than by what already exists.

---

*End of brief.*
