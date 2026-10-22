# POLISH — Decision Register (D1–D92)

**Compiled 13 September 2026. Revision 2** — external review folded in.
Companion document to the four-pass review.

### What changed in revision 2

Two independent external reviews of revision 1 produced six findings not in the
four-pass review (F14–F19), resolved the naming conflict, and forced two
corrections to the register itself.

| Change | Where |
|---|---|
| Naming conflict resolved via three export profiles | D9, D10, D62, new D85 |
| Descending logs / negative `STEP` — assumption of ascending depth was wrong | D3, D65, new D88 |
| Co-tool gap correlation is evidence, not proof | D49 |
| Physical limits split into hard vs. plausibility | D11, new D87 |
| Trust-state vocabulary renamed to operational fields | D21 |
| No general "force export" override; block types classified | D79, new D86 |
| The \$75k figure struck — it came from a role-play, not a market | D20 |
| Competitor claims narrowed | D69, D77 |
| Several LAS claims upgraded UNVER → PARTIAL (corroborated, not verified) | D65, D68 |

**Three corrections to revision 1 itself.** The title read "D1–D69" while the
register ran to D84. D9 and D10 cited "D54 (preserve originals)" — D54 is the
regeneration record; the preserve-originals decision is **D62**. And D8 cited
"D63 disposition gating" — D63 is the input-file hash; disposition gating is
**D79**. Both external reviewers inherited the wrong cross-references rather
than catching them, which is worth knowing about how closely a register gets
read.

---

## How to read this

**The D-numbers are new.** They were created while writing the four-pass review
and are not a numbering scheme from the project's own documents. They do not map
onto the C1–C7 pipeline contracts or the Stage 0–10 pipeline numbers. Treat them
as reference handles for this review only.

**Source column:**

| Code | Meaning |
|---|---|
| DAD | Decided by the domain expert (geophysicist, 40+ yrs). Treat as authoritative on petrophysics. |
| AC | Decided by the project owner on product or business grounds. |
| REPO | Verified present in the codebase at HEAD `f815811`, branch `freeze-fix-wip`. |
| DESIGN | Proposed by Claude during design conversation and not challenged. **Lowest-authority category — most likely to contain errors.** |

**Status column:**

| Code | Meaning |
|---|---|
| — | Not flagged by the four-pass review. **Not the same as verified.** |
| REVISE | Error identified. Finding number given. |
| STRUCT | Structural conflict. Decide before scaffolding. |
| OPEN | Unresolved; needs the domain expert. |
| UNVER | Rests on an unverified claim about industry standards or the LAS spec. |
| CONFLICT | Contradicts another decision in this register. |
| PARTIAL | Corroborated by independent implementations, not verified against the primary specification. |
| RESOLVED | Was a conflict; closed in revision 2. Resolution stated in the row. |

**Note on DESIGN items:** roughly two-thirds of this register originated as
Claude's proposals in conversation rather than as decisions either party
independently arrived at. Across both review rounds, errors concentrated heavily
in DESIGN rows; **no error has yet been found in a DAD row.** Where a DAD
decision appeared to conflict with something, the surrounding framework was too
narrow to hold it. A reviewer should weight scepticism accordingly.

---

## 1. Landed and verified in the repository

| ID | Decision | Source | Status |
|---|---|---|---|
| D1 | Seaborn removed from the codebase entirely rather than added to `requirements.txt` (also resolves PyInstaller build failures) | REPO | — |
| D2 | Resampling gets its own switch, defaulting to off | REPO | — |
| D3 | Depth identification halts rather than guessing when the index curve is ambiguous | REPO | REVISE — F14: the surrounding Stage 1 monotonicity test assumes **ascending** depth. A negative STEP is valid LAS (F14), and POLISH refuses it. See the amendment below. Amendment (relayed by owner, 2 Oct 2026; not a pasted answer from the domain expert): Stage 1 accepts strictly ascending depth only. A decrease, or a negative STEP, is a Stage 1 structural failure (relayed by owner, 2 Oct 2026). Nothing is reversed or reordered; the file is left untouched and later stages do not run (relayed by owner, 2 Oct 2026). User message: "valid LAS, but POLISH supports increasing depth only" (relayed by owner, 2 Oct 2026). The halt behaviour is inferred from "left as is and notified" (derived by analysis). |
| D4 | `docs/governing/POLISH_pipeline_contracts.md` (contracts C1–C7) is the governing specification | AC | — |
| D5 | KEOUGH is a regression fixture with toleranced assertions, not a frozen reference file | AC | — |
| D6 | Handoff documents use one numbering scheme, not several overlapping ones | AC | — |

## 2. Decided by the domain expert

| ID | Decision | Source | Status |
|---|---|---|---|
| D7 | TVD resamples along with everything else | DAD | REVISE — F2: conflicts with D32 role taxonomy; TVD is a computed geometric reference, a role the taxonomy lacks. The decision is right; the taxonomy is too narrow. Also F13: linear TVD interpolation is near-exact at normal increments but degrades across large increment changes in high-build-rate sections. |
| D8 | Wells whose depth data disagrees with declared header `STRT`/`STOP`/`STEP` are refused with a message and skipped | DAD | RESOLVED — F3/F17: the apparent contradiction with disposition gating (**D79**, mis-cited as D63 in revision 1) dissolves under D86. Header/data disagreement is not one condition but several: within tolerance → record and permit; endpoint rounding → flag, allow explicit acceptance; non-monotonic, duplicate or grid-inconsistent depths → structural failure, no override; ambiguous index curve → halt. The expert's refusal stands as a declared non-overridable constraint for the last two. No single tolerance number: the matching rule is in D12. |
| D9 | Non-standard depth mnemonics handled by **copying** — original retained, standard-named copy added alongside | DAD | RESOLVED — F16: the copy rule generalises as the **interoperability export profile** (D85). It remains the default for depth curves; for log curves it applies only within that profile, additively and on explicit selection. The earlier note cited D54 in error; the relevant row is D62. |
| D10 | Rename-on-by-default is correct on product grounds | AC | RESOLVED — F16: satisfied by D85's interoperability profile, where standardised names are produced as **additive** curves. Never a silent replacement of a source mnemonic. Both external reviewers reached this independently. |
| D11 | Physical limits for GR, RHOB, NPHI, DT, PEF, resistivity, SP | DAD | OPEN + REVISE — F15: "physical limit" conflates four rule types (mathematical/domain constraint, hard physical impossibility, plausibility range, data-consistency rule). Only the second justifies marking a value invalid. **The worksheet asks for min/max and must be split.** Where unit or family confidence is low the correct outcome is *not eligible for automatic range validation*, never *invalid*. See D87. **SP resolved 18 Sept — see 1.0.1: −150/50 mV kept, log-level 20%-of-samples threshold, not per-sample.** 14 Oct 2026: shipped in P12 as rule library v1 with the 16 Sept types (DT HARD_IMPOSSIBILITY; GR, RHOB, NPHI, resistivity PLAUSIBILITY; PEF PLAUSIBILITY until classified; SP log-level, more than 20% outside). Family mapping not confirmed by the domain expert. A curve with INVALID samples is HARD_VIOLATION_EXCLUDED: those samples are not used by later stages; output inclusion is decided by the user (owner, 14 Oct 2026). |
| D12 | Refusal tolerance — how far data may drift from header before refusal | DAD | RESOLVED — CLOSED 23 Sept (owner). Part One (detection) unchanged: implied interval (STOP − START) / (NSAMPLES − 1) compared to known standard intervals in feet and metres (including 0.1524 m = 0.5 ft); treat as rounding only on a match. Part Two (header STOP): both verification boxes true. **DECISION: YES.** If a well's depth grid is corrected via step-step calculation, the exported STOP depth in the final output file header MUST be rewritten to reflect the newly calculated true depth (STRT + (n-1) * S_corrected). Original file is never altered. Eligibility to apply that correction is D89 and does not change this rule. Rounding test (derived by analysis, consistent with the domain expert's rule "determine whether differences are due to rounding errors"): a depth counts as rounding if it is within one full unit of its own last written decimal place of a grid built from STRT. Reason: a rounded STRT and a rounded depth can each be off by half a unit (derived by analysis). The file's precision is recorded in the audit (derived by analysis). Each depth is judged at its own last written decimal (derived by analysis). The comparison is inclusive: exactly one unit passes (derived by analysis). The domain expert approved a half-unit version in writing (0.00005) before this gap was found (domain expert, written). The full-unit version and the per-depth precision rule were both confirmed by him on 7 Oct 2026 (relayed by owner). Consistent but non-standard step (domain expert, written): a well with a consistent but non-standard step (34_10-19 and 30_3-5 S, step 0.152 m) is left as written, no correction, with an informational notice that the step differs from 0.1524. "Bad well" (D89) means an actually irregular well only (derived by analysis). End depth: a header STOP within one sample interval of the last depth is informational (derived by analysis (what s01_depth.py implements)). The code applies the same one-interval allowance to the start depth (STRT). His answer covered the bottom depth only. The owner decided on 7 Oct 2026 to keep it for now. It is derived by analysis, not yet confirmed by him. Beyond one full interval the code refuses (STRUCTURAL_DATA_FAILURE, "beyond one step"). That refusal is derived by analysis, not his word. Evidence is already in the build plan, lines 112-117: 0 of the 143 computable wells differ by one sample interval or more (1 not computable), and STRT matched exactly in 143 of 143. The domain expert said "let's call it good and notify the user" for 0.55 and 0.81 of a step (domain expert, written, screenshot, 2 Oct 2026). For the bottom depth (STOP), a header STOP up to one full step from the last data depth is acceptable (domain expert, 7 Oct 2026, relayed by owner). Standard steps (derived by analysis): depth_standard_steps.json and the 0.1524 m primary metre step apply to polish/ only. The legacy `_sync_depth_spacing_default` metre value of 0.1 is not a conflict in the rebuild (derived by analysis). |
| D13 | The 8-foot BLI1 discrepancy in KEOUGH — header or data governs? | DAD | RESOLVED — domain-expert follow-up Q6: skip only the density readings below 5,326 ft (5,326.5–5,334 ft); the rest of the density curve is unaffected. Quote: "Only the density readings below 5,326 ft (5,326.5 - 5,334 ft)". |

## 3. Product and commercial decisions

| ID | Decision | Source | Status |
|---|---|---|---|
| D14 | POLISH ships fully isolated: no network calls, no cloud storage, no account, no telemetry, no custody of customer data | AC | — |
| D15 | Licensing by signed offline license file; public key embedded in binary; existing Render/Firebase license server not used | AC | REVISE — F11: needs a clock-rollback guard (store last-seen timestamp, refuse when system clock precedes it). |
| D16 | Full pipeline, all ten stages, for v1 | AC | OPEN — F5 (D20 benchmark is unscoped work inside this), and an external reviewer proposes **Stages 0–4 read-only as the first sellable milestone** instead: near-zero liability, validates the trust model before shipping anything that can fabricate a value, usable as an intake/QC gate. Directly contradicts this decision. Owner's call; the most consequential open item in the register. |
| D17 | Two files on export: original never opened for writing, processed written separately | AC | — |
| D18 | The legend goes in the processed file | AC | — |
| D19 | The user names the output file (save dialog, default suggested, input path blocked) | AC | — |
| D20 | A demonstrated comparison against the existing manual workflow is a product requirement | DESIGN (from procurement-persona session) | — (unscoped; see F5). **Retained on product merit, detached from any price point.** Feasible in-house without customer data: the domain expert is a practising petrophysicist and Volve / FORCE 2020 are publicly licensed and already on disk — so the benchmark can run with zero custody, satisfying D14. |
| D20a | ~~A \$75k annual licence is the commercial target~~ | **STRUCK** | Originated in a 9 Sept role-play session where Claude played a procurement persona. Not a price the owner set, not the output of buyer research, never shown to a prospect. Both external reviews built pricing and packaging recommendations on it. Recorded here only so it is not silently reintroduced. |

## 4. Trust state

| ID | Decision | Source | Status |
|---|---|---|---|
| D21 | Three-state trust framework: validated / flagged / unvalidated | AC | REVISE — F4: must be separate from sample status (D44) and curve role (D32); currently conflated in the design brief. **Extended:** "validated" is ambiguous — it could mean curve identification, a range-check result, a user decision, or downstream freshness. Split into named operational fields: `evidence_tier`, `processing_eligibility`, `validation_outcome`, `sample_status`, `issue_disposition`, `dependency_state`. The user-facing three-state summary can remain; the data model may not. |
| D22 | The unvalidated state must surface in the UI **and** in the audit export | AC | — |

## 5. Pipeline structure and sequencing

| ID | Decision | Source | Status |
|---|---|---|---|
| D23 | Ten stages, ordered: describe completely → alter minimally with consent → report everything | DESIGN | — |
| D24 | Stages 0–4 touch no data | DESIGN | — |
| D25 | Stages 5–8 alter data; each independently switchable; all default to off | DESIGN | — |
| D26 | Stage 4 (validity assessment) is the keystone — build it before repairing gap filling | DESIGN | — (F6 adds a dependency: needs tool grouping from Stage 2 first) |
| D27 | Phase order: golden files → contracts → extraction → UI | DESIGN | — (survives all Pass 3 findings; every revision lands inside the contracts phase) |
| D28 | No stage may invent an assumption another stage owns | DESIGN | REVISE — F10: Stage 1 requires the depth-alias subset of the mnemonic table to execute D9's copy rule, which Stage 2 owns. Resolvable (positional index detection needs no table; alias lookup runs at end of Stage 1 against a small hardcoded set) but must be documented so the ordering isn't "corrected" later. |
| D29 | Refusal is a first-class outcome, reported as a correct result | DESIGN | — |

## 6. Curve identification

| ID | Decision | Source | Status |
|---|---|---|---|
| D30 | Curve identity is the column index, not the mnemonic (mnemonics can legitimately duplicate within one file) | DESIGN | REVISE — F9: index is not stable across files. Provenance and session references must key to (file hash, column index), or (dataset id, column index) after F1. |
| D31 | Identification output is family **plus qualifiers**, not a single label | DESIGN | — |
| D32 | Role taxonomy: INDEX / MEASURED_LOG / DRILLING / ANCILLARY / SYNTHETIC_SOURCE / UNKNOWN | DESIGN | REVISE — F2: incomplete. Needs a computed-geometric-reference role for TVD, which must track the index. TVD role name decided 9 Oct 2026 (owner): GEOMETRIC_REFERENCE, an internal name. TVD, TVDEPTH and TVDSS carry it in polish/tables/mnemonics.json. This closes the name only; the rest of OD-17 is not closed by it. |
| D33 | Four source tiers: Cited / Observed / User-defined / Proposed | DESIGN | — 9 Oct 2026 (owner): every legacy mnemonic migrated in P10 is CITED. Basis: the owner states the legacy names were reviewed and are correct, from first-degree sources (chartbooks); not re-verified against them. Each entry keeps its legacy source text verbatim. PROPOSED applies only to similarity suggestions and names not in the shipped tables. A similarity suggestion is never applied automatically and is never offered for the P10 item 8b do-not-map list. |
| D34 | Tier gates downstream processing — a low-tier match does not unlock range validation or gap filling | DESIGN | — |
| D35 | Vendor prefix stripping, from a table of recognised prefixes | DESIGN | — |
| D36 | Short mnemonics remain exact-match only; unrecognised trailing tokens are not silently stripped | DESIGN | — |
| D37 | The shipped mnemonic table is comprehensive on published industry names; local extension covers only the shop-specific tail | AC | — |
| D38 | Alias and qualifier tables are local files on the user's disk, never transmitted | AC | — Path decided 9 Oct 2026 (owner): %APPDATA%\POLISH\tables\. P10 only loads from it and does not create it; saving is P13. The file name mnemonics.json was chosen by Cursor in P10, not a recorded decision. |
| D39 | `_MAIN` / `_REF` are pass designators (main pass, repeat pass), not synthetic noise — dropping them in the July merge lost real information | DESIGN | — |
| D40 | Two new outcomes needed: ANCILLARY (coordinates, facies labels, QC flags — loaded, displayed, excluded from processing) and SYNTHETIC_SOURCE (`*_SYNT`, vendor-computed — never treated as measurement, never a derivation input) | DESIGN | — |
| D41 | Families need a **tool-grouping** property (RHOB/DRHO/PEF = density tool) | DESIGN (F6) | NEW — added by the review. Must land in Stage 2 before Stage 4 is built. As built in P10: tool_group is null on every family until OD-13 is answered. |
| D42 | Families need an **interpolation_domain** property, distinct from display log-scale | DESIGN (F7) | NEW — added by the review. As built in P10: interpolation_domain is LOG on the five RESISTIVITY_* families (D52) and null on all others, including FORMATION_RESISTIVITY_IMAGING, until OD-14 is answered. display_log_scale is not used to set it (F7). |

## 7. Data model

| ID | Decision | Source | Status |
|---|---|---|---|
| D43 | The original is immutable; every change is a sparse overlay keyed to sample index | DESIGN | STRUCT — F1: incompatible with resampling, which changes the index. Resampling must produce a **child dataset**, not an overlay. Naive implementation interpolates the base layer to render it, making the "unmodified original" itself synthesized. **Decide before scaffolding.** |
| D44 | Sample status values: measured / created / changed / invalid / absent / resampled | DESIGN | — Follow-up Q4: "delete" of an out-of-limit value means the sample becomes internal NaN, is written as −999.25 on export, stays visible on screen (marked), stays untouched in the original file, and the decision is recorded in the report. Quote: "Delete means mark as \"no value\" visually -999.25". Matches `petrophysics/constants.py` `NULL_VALUES` (`"STANDARD": -999.25`, `"IEEE_NAN": float('nan')`) and commit `dbba867` ("Make NaN the single internal null representation; sentinels only at export"). |
| D45 | Created (no measurement existed) and changed (a measurement was altered) are different claims and must render differently | AC | — |
| D46 | Sessions keyed to a SHA-256 of the input file, not to well identity | DESIGN | — |
| D47 | Session files (decisions only, never values) stored beside the well; preferences (no well information) in app-data | DESIGN | — |
| D48 | Filename match with hash mismatch is a resolution point, never silent application of stale settings | DESIGN | — 15 Oct 2026 (owner): when the raw file has changed, the user chooses USE_SAVED_DECISIONS or START_FRESH; the old session file is always kept beside the well, never overwritten. |

## 8. Gap filling and derivation

| ID | Decision | Source | Status |
|---|---|---|---|
| D49 | Gap classification by evidence, not by size | DESIGN | REVISE — F6: the rule as stated ("a gap running to the measured-interval boundary is tool absence") is vacuous, since the measured interval is *defined* as first-to-last non-null. It also misses **interior tool absence** (tool pulled and re-run, spliced runs). Corrected rule set: (1) outside measured interval → absent; (2) interior gap shared across curves from the same tool → tool absence; (3) interior gap shorter than tool vertical resolution → dropout; (4) else → ambiguous, and only this case needs a threshold. **Softened:** shared absence across a tool group is *evidence consistent with* tool absence, not proof — telemetry loss, post-acquisition editing, a splice between passes, a merged vendor dataset, a shared QC masking convention or depth misalignment all produce the same signature. Classification carries the evidence and a confidence; non-fillable by default either way. |
| D50 | Tool-absence gaps are never fillable — enforced structurally, not by an overridable flag | DESIGN | — |
| D51 | Depth-normalised thresholds replace point-based ones (the correct table already exists in `petrophysics/constants.py`, marked PREFERRED, and is not wired in) | DESIGN | — (note: the preferred table is feet-only; three of four fixtures are metric) |
| D52 | Log-domain interpolation for log-scaled families (resistivity spans decades; linear midpoint of 2 and 200 ohm·m is 101, geometric mean is 20) | DESIGN | REVISE — F7: two errors. The existing `log_scale` property is a *display* attribute and using it to select an interpolation domain is a category error (GR is often plotted log and is not log-distributed). Needs D42. Also requires a non-positive guard — resistivity is recorded as zero or negative in bad hole, and `log(0)` is undefined; the fill must be refused with the reason recorded. |
| D53 | `MIN_POINTS_BY_METHOD` enforced as a precondition that declines rather than falling back silently | DESIGN | REVISE — F8: the thresholds (30 linear / 50 power-law / 100 ensemble / 200 cross-well) cannot apply to bounded interpolation, which needs exactly two bracketing points. A 30-point minimum would refuse nearly every dropout fill. The table is about **fitting a model** — it applies to Stage 6 derivation and the inferential tier of Stage 7, not to interpolation. Confirm the table author's intent before wiring. |
| D54 | Every filled interval emits a regeneration record (span, depth interval, method, domain, exact inputs, parameters, per-sample uncertainty, selection criterion); the acceptance test is that a third party reproduces the values from the record alone | DESIGN | — |
| D55 | Stage 6 derivation (whole missing curve from measured curves by a stated equation) is separate from Stage 7 gap filling | DESIGN | — |
| D56 | A physical bound violation means the fill is **refused and the gap left open**, with the reason recorded — never clipped | DESIGN | — |
| D57 | The uncertainty field must be able to read "not quantifiable for this method" — linear interpolation between two measurements has no intrinsic error bar, and inventing one contradicts the product's premise | DESIGN | — |
| D58 | Derived curves may not feed further derivation without explicit consent; never derive from a SYNTHETIC_SOURCE input | DESIGN | — |

## 9. Export, header and ingest

| ID | Decision | Source | Status |
|---|---|---|---|
| D59 | The output header is **generated from final state**, never inherited and patched. Preserved-verbatim fields are a separate explicit list. | DESIGN | — Addendum 19 Sept (owner): a field's status (generated-from-final-state vs. preserved-verbatim) changes only through its own explicit, recorded decision — never by default, and never for implementation convenience. D12 is the procedural precedent: a header field changes because a specific question was asked and answered, not because regenerating was easier than preserving. D12 Part Two is RESOLVED 23 Sept: processed STOP is rewritten to STRT+(n−1)×S_corrected when the well is corrected. |
| D60 | Round-trip invariant: a file POLISH wrote must load in POLISH without triggering a refusal. Second invariant: re-importing a processed file rebuilds an identical mask. | DESIGN | — |
| D61 | Status curves with a fixed, append-only code table, versioned, written into every export | AC | — |
| D62 | Original mnemonics preserved in the output; POLISH's family assignment is an annotation | DESIGN | RESOLVED — F16: holds for the auditable-processed profile (D85). Original input is never written. The interoperability profile may add canonical names as extra curves without altering originals. |
| D63 | A SHA-256 of the input file recorded in the processed file, so pairing survives renaming and copying | DESIGN | — |
| D64 | Two-pass ingest: raw text audit, then tolerant parse, with an **audit ledger** recording every ambiguity. `ignore_header_errors` is never used to suppress silently. | DESIGN | — |
| D65 | LAS format specifics: only the first character after `~` is significant; section boundary is the last section whose first letter is A; header fields split on **first period** and **last colon**; column count must equal the `~CURVE` count | DESIGN | PARTIAL — corroborated by KGS, geolog4 and lasio documentation, i.e. by independent implementations. **Not verified against CWLS.** One reviewer marked these "supported"; a second disagreed, and the second is right — vendor docs describe what a parser tolerates, not what the format requires. **Amended:** a negative `STEP` is valid LAS but POLISH refuses it (see D3) (relayed by owner, 2 Oct 2026), and the column-count rule must account for `WRAP=YES`, comments and blank lines rather than being a binary parse failure. |
| D66 | Null handling: match declared NULL with a tolerance not exact equality; never assume a default when NULL is undeclared; scan every column for known null sentinels and flag every undeclared one | DESIGN | PARTIAL + REVISE — F19. `-999.25` is confirmed as common practice but **not** as a spec default, so never-assume stands. Sentinel scanning must remain a **warning heuristic**, never a parser rule that silently converts values — a legitimate measurement can coincide with a sentinel. **Contradiction to resolve:** this row lists `1e30` / `-1.0e+30` as observed sentinels, while a proposed strict mode would flag scientific notation as nonconforming. That rule traces to a KGS page documenting KGS's own software, not the LAS format, and would flag the null column of a large share of real wells. Do not implement it. |
| D67 | `STEP = 0` when the increment is not constant | DESIGN | RESOLVED — domain-expert follow-up Q3, reading (b): if depths are genuinely uneven, accept as irregularly spaced, do not invent a step, notify the user. Quote: "Accept it as unevenly spaced, don't invent a step, and tell me". Matches D88. |
| D68 | LAS 2.0 as the target format | DESIGN | PARTIAL — supported: USGS describes LAS 2.0 as the most-used digital well-log distribution format. A v2-targeted product must **detect and report** v3 features rather than parsing them optimistically. |
| D69 | Companion flag curves are established industry practice (evidenced by `ABDCQF01`–`04` in the Volve fixture) | DESIGN | REVISE — F18: the fixture evidence is direct; the generality is inferred and should be stated as such. |

## 10. Interface

| ID | Decision | Source | Status |
|---|---|---|---|
| D70 | Coverage track as the landing view — all curves, full well, rendered from Stage 4, requiring no processing and no settings | DESIGN | — |
| D71 | Depth is the shared vertical spine; nothing competes with it; **nothing that appears may move anything else** | DESIGN | — |
| D72 | Colour is reserved for curve identity (GR green/black, RHOB red, NPHI blue — read by reflex). Provenance encodes on marker shape, line style and background band. | DESIGN | — |
| D73 | Redundant, multi-scale encoding: marker shape at sample zoom, line style at mid, background band at full-well. Colour-only distinction is not acceptable (~8% of men have a colour vision deficiency; composite logs are printed in greyscale). | DESIGN | REVISE — F12: at full-well scale only the band survives, and four to five states in one channel using tint and hatch is too tight. Full-well should distinguish **modified from unmodified only**; finer distinctions require zooming. State this as a deliberate limitation. |
| D74 | Decimation rule: if any sample within a rendered pixel is created or changed, the pixel renders as created or changed. Never average measured and synthesised into one mark. | DESIGN | — |
| D75 | Provenance visible by default on open; hiding requires an action. Two view states only — original-only, and original-plus-overlay. **No created-only view.** | AC | — |
| D76 | Original and created are drawn on the **same track**, created layered over original, original never removed; the created layer toggles off and on | AC | — |
| D77 | A difference view (plotting the delta for changed samples) is worth having | DESIGN | REVISE — F18: "no competitor ships it" struck. Unverifiable absolute about four products never tested. Positive framing instead: POLISH makes per-sample provenance, explicit refusals and original-vs-modified comparison primary and exportable, rather than treating processing history as opaque. The same edit is needed in §3 of the design brief. |
| D78 | Hover shows the per-sample derivation, not just the value: depth, value, status, method, domain, and the actual inputs (bracketing samples and values, or source curves and parameters) | AC | — |
| D79 | Disposition gating: every raised item must reach one of four terminal states before export — resolved / accepted / deferred / refused. Export is gated on every item having a disposition, **not** on every item being fixed. Silence is not one of the states. | DESIGN | REVISE — F17: not every finding is dispositionable. There must be **no general "force export anyway"** — under deadline it becomes the button everyone presses, which destroys the differentiator. Findings classify by block type (D86); D8's refusal is then a declared non-overridable constraint rather than a contradiction. |
| D80 | Staleness is a blocking fifth state: changing an upstream decision marks dependent downstream stages stale rather than leaving them quietly inconsistent | DESIGN | — |
| D81 | Resolution points are a non-modal queue, never popups; every item has an identical shape (evidence / options / consequence / disposition) in an identical screen position | DESIGN | — |
| D82 | Batch disposition and keyboard traversal are requirements, not refinements — the stress-test fixture has 171 curves, 170 sharing one prefix | DESIGN | — |
| D83 | Friction proportional to consequence: reversible actions instant and unceremonious; irreversible actions deliberate and confirmed | DESIGN | — |
| D84 | Progressive disclosure in three optional depths: track (glance) / annotation (read) / record (inspect) | DESIGN | — |

## 11. Added by external review, 13 September

| ID | Decision | Source | Status |
|---|---|---|---|
| D85 | **Three export profiles**, not one global naming behaviour. (a) *Original input* — never written or altered. (b) *Auditable processed* — the primary POLISH result; original mnemonics and curve order preserved; POLISH's interpretation lives in annotations and the manifest. (c) *Interoperability* — optional, explicitly selected per customer; adds canonical aliases as **additive** curves with a complete mapping record, never replacing a source mnemonic. | REVIEW | NEW — resolves D9 / D10 / D62. Both external reviewers proposed this independently. |
| D86 | **Block-type taxonomy.** Every finding is classified before implementation: informational discrepancy (dispositionable) · ambiguous interpretation (requires explicit user choice, rationale recorded) · rule-dependent warning (explicit labelled user mapping only) · scientific precondition failure (method declined, e.g. log fill with a non-positive endpoint) · structural data failure (workflow or file refused) · export conformance failure (fix or do not export). **No general force-export override exists.** | REVIEW | NEW — resolves the D8 / D79 conflict. |
| D87 | **Versioned validation-rule library**, not thresholds embedded in application logic. Each rule carries: curve family · accepted and canonical units · rule type (domain constraint / hard impossibility / plausibility / consistency) · bound values · environmental qualifiers · source · source tier · library version · whether violation means invalid, warning or refusal · whether it is overridable and how the override is recorded. | REVIEW | NEW — makes D11 implementable and prevents a repeat of the GAPI-bounds-on-CPS-curves bug. |
| D88 | [Superseded in part by D3; see status.] **The depth axis may ascend or descend.** Negative `STEP` is valid and normal. Monotonicity is tested in either direction; ordering is recorded, never silently reversed. Descending and irregular-but-monotonic grids are both accepted as-is — POLISH does not pretend an irregular grid is regular. | REVIEW | NEW — F14. Closes an assumption that ran through D3, D65 and Stage 1 unnoticed. The "may ascend or descend" half is superseded by D3 (relayed by owner, 2 Oct 2026). Kept: an irregular ascending grid with a declared STEP is accepted and recorded as irregular, and a regular step is never invented (relayed by owner, 2 Oct 2026). |

## 12. Added 23 September (domain expert)

| ID | Decision | Source | Status |
|---|---|---|---|
| D89 | A well is corrected to the standard step only if max bottom displacement \|original last depth − (STRT + (n−1)×S)\| ≤ 0.5 × S; otherwise the correction is invalidated, the well is left verbatim, and the end user is notified ("bad well"). | DAD | NEW — Alvaro Sr. 23 Sept, eligibility. Does **not** modify D12. D12 Part Two still governs wells that pass. Split (domain expert, written): a well with a consistent but non-standard step (34_10-19 and 30_3-5 S, step 0.152 m) is left as written, no correction, with an informational notice that the step differs from 0.1524. "Bad well" means an actually irregular well only (derived by analysis). |

## 13. Added 12 October 2026 (owner)

| ID | Decision | Source | Status |
|---|---|---|---|
| D90 | A family may accept more than one unit dimension. Units in different dimensions of one family are never converted between; a curve in any accepted dimension is compatible with its family. | AC | NEW — Decided 12 Oct 2026 (owner). GAMMA_RAY_TOTAL accepts gamma_api and count_rate (OD-05 case 4). ACOUSTIC_IMAGING and CARBON_OXYGEN_RATIO take their dimensions from the legacy unit lists, not reviewed by the domain expert. A curve whose unit is recognised but not in its family's accepted dimensions stays eligible and carries a RULE_DEPENDENT_WARNING (owner, 12 Oct 2026). |

## 14. Added 15 October 2026 (owner)

| ID | Decision | Source | Status |
|---|---|---|---|
| D91 | A curve is mapped only by choosing the family it behaves as; the user's own name is an optional label. A saved mapping with no family is kept, not processed, and the user is told why. The domain expert's "leave unchanged" list is never mappable. A mapping saved for all wells is changed only by an explicit confirmation; one name never has two saved rows. | AC | NEW — Decided 15 Oct 2026 (owner). Unrecognised-name and no-family notices are RULE_DEPENDENT_WARNING (map explicitly); the "leave unchanged" list keeps its INFORMATIONAL notice. 15 Oct 2026 (owner): a mapped curve never has role UNKNOWN. It takes the role all of its family's shipped names state; where they differ (today only DEPTH) the user chooses one of that family's roles. |
| D92 | A value outside a suspicious limit gets a keep-or-exclude question with a required reason; either answer may be saved for all wells. KEEP: used by later stages and flagged. EXCLUDE: not used by later stages, never hidden (shown marked, original untouched, listed in the export). The sample stays MEASURED. | AC | NEW — Decided 15 Oct 2026 (owner), following the domain expert's answer (POLISH_followup_answers.md lines 89–91). SP stays a log-level review flag (same file, line 129). A saved all-wells answer is changed only by explicit confirmation. |

*(Register ends at D92. See the Count section below for verified tallies. The
"44 decisions" figure in the four-pass review was an estimate stated as a count
and is wrong.)*

---

## Count

Verified by parsing this document, not estimated.

| | Count |
|---|---|
| **Total register entries** | **92** (D1–D92, continuous, plus struck D20a) |
| Decisions made during the project week | 82 |
| Added by the internal four-pass review | 2 (D41, D42) |
| Added by external review | 4 (D85–D88) |
| Struck | 1 (D20a, the \$75k figure) |

### By source

| Source | Count |
|---|---|
| DESIGN — Claude's proposals, unchallenged | 56 |
| AC — project owner | 22 |
| DAD — domain expert | 7 |
| REVIEW — external review | 4 |
| REPO — verified in code | 3 |

### By review status

| Status | Count |
|---|---|
| — not flagged | 55 |
| REVISE | 14 |
| NEW | 10 |
| RESOLVED | 7 |
| OPEN | 2 |
| PARTIAL | 3 |
| STRUCT | 1 |
| UNVER | 0 |
| CONFLICT | 0 |

Counts parsed from the table rows, not tallied by hand. All three CONFLICT rows
from revision 1 are now RESOLVED, plus D8. D12 Part Two is RESOLVED (CLOSED 23 Sept).
D13 and D67 are RESOLVED from the domain-expert follow-up (16 Sept). D89 is NEW. D90 is NEW. D91 is NEW. D92 is NEW.
D20a is struck and excluded from these counts.

### Findings

19 findings. F1–F13 internal, F14–F19 external. They touch 21 distinct rows and
generated 6 new entries. Some findings span more than one row and some rows
carry more than one finding, so the two counts do not reconcile by addition.

**Error distribution by source:** DESIGN heavily dominant; DAD rows have
produced no errors in either round — where a DAD decision appeared to conflict,
the surrounding framework was too narrow to hold it.

---

## Findings index

Referenced above by number. Full reasoning is in the four-pass review.

| # | Finding | Severity |
|---|---|---|
| F1 | Overlay model incompatible with resampling; resampling must produce a child dataset | Structural |
| F2 | TVD's role missing from the taxonomy; DAD decision correct, framework too narrow | Revise |
| F3 | "Refuse and skip" contradicts disposition gating | Conflict |
| F4 | Three orthogonal classification systems conflated (sample status / curve role / trust state) | Revise |
| F5 | Manual-workflow benchmark is unscoped work inside v1 | Scope |
| F6 | Gap classification rule partly vacuous; misses interior tool absence; co-tool correlation is the missing deductive signal | Revise |
| F7 | Interpolation domain conflated with display log-scale; no non-positive guard | Revise |
| F8 | `MIN_POINTS_BY_METHOD` cannot apply to bounded interpolation | Revise |
| F9 | Column index not stable across files | Revise |
| F10 | Stage 1 requires part of Stage 2's table | Revise |
| F11 | Offline license needs clock-rollback guard | Revise |
| F12 | Band channel too narrow for 4–5 states at full-well scale | Revise |
| F13 | TVD linear interpolation degrades across large increment changes in high-build sections | Minor |
| F14 | Ascending-depth assumption; negative `STEP` and descending logs are normal | Revise |
| F15 | "Physical limit" conflates four distinct rule types | Revise |
| F16 | Naming conflict resolved by three export profiles | Resolution |
| F17 | No general force-export override; findings need block-type classification | Revise |
| F18 | Competitor absolutes are unverifiable and should be struck | Revise |
| F19 | Sentinel scanning must stay a heuristic; the scientific-notation rule contradicts `1e30` nulls | Revise |

**F1–F13** came from the internal four-pass review. **F14–F19** came from two
independent external reviews of revision 1.

---

## For the reviewer

Four things worth knowing before you start.

**1. Two review rounds have now run and converged.** Two independent external
reviewers, reading revision 1 separately, found substantially the same items the
internal review had already flagged — F1, F4, F6, F7, F8 and the two conflicts.
That convergence is evidence the flagged set is real. It is **no evidence at all
that the 55 un-flagged rows are sound**; neither external review worked through
the interface decisions, the ingest ledger, the session model or the phase
ordering in any depth. Everything marked "—" means un-flagged, not verified.

**2. Two-thirds of this register is DESIGN-tier** — proposed by Claude in
conversation and not independently arrived at by either party. Seven of the
thirteen findings were errors in DESIGN items. None were in DAD items. The
highest-yield review target is therefore the DESIGN rows still marked "—".

**3. The LAS claims are corroborated, not verified.** D65, D66 and D68 are now
PARTIAL — independent implementations (KGS, geolog4, lasio, USGS) agree with
them. One external reviewer marked several as "supported" on that basis; the
other disagreed. **The second is right:** vendor documentation describes what a
parser tolerates, not what the format requires, and two of those sources
document their own software. The CWLS LAS 2.0 document is still the thing to
get, and D65–D68 should become an executable conformance-test matrix before any
of it reaches code.

D67 (genuinely uneven depths) is RESOLVED from the domain-expert follow-up Q3:
accept as irregular, do not invent a step, notify — matching D88.

**4. Open domain-expert questions remaining after the 23 Sept follow-up:** D11 (limit types for GR/PEF/DT/NPHI/resistivity; worksheet min/max still to be split per D87 — SP is now settled, see 1.0.1). D9, D12, D13, and SP under D11 are RESOLVED. D12 Part Two: processed STOP is rewritten to STRT+(n−1)×S_corrected when the well is corrected. D89 is the eligibility gate for that correction.

### Where a fresh reviewer is most likely to find something

- **D43 / F1** — the child-dataset resolution is mine and only one of several
  possible answers. Worth challenging.
- **D49 / F6** — the corrected gap rule is better than what it replaces but I
  cannot prove there isn't a fifth case.
- **D30 / D9 / D10 / D62** — the identity and naming cluster. Two decisions
  directly contradict and my proposed resolution is inference.
- **D72–D74** — the visual encoding scheme has not been tested at real data
  density.
- **D23–D27** — the ten-stage decomposition is a reasonable seam-placement, not
  a standard. A reviewer with commercial-preprocessor experience may cut it
  differently.

**Open from `POLISH_followup_answers.md` (16 Sept, Part Two closed 23 Sept).** D12 Part Two is RESOLVED: Yes — processed STOP is rewritten to STRT+(n−1)×S_corrected when the well is corrected. Q5 records "nearest equivalent, labelled clearly"; the ILD substitute list was asked twice and left blank both times.

*End of register.*
