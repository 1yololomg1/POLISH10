# P03 — Contracts vs register (reconciliation)

Read (this task): `POLISH_pipeline_contracts.md`, `POLISH_decision_register.md`, `POLISH_design_brief.md`, `POLISH_UIUX_cohesion_supplement.md`, `README.md`. No governing file other than this one was edited. Nothing below is applied.

Status filter for Table 1: every register row whose status contains REVISE, NEW, RESOLVED, STRUCT, OPEN, PARTIAL or UNVER (dual tags listed once), plus adopted UXC1–UXC10. Dual readings are listed, not chosen.

---

## Table 1 — Register → contracts

| Row | Touches | Contracts reflect it? | Quoted contracts sentence (one line max) | Proposed contract text (PROPOSAL — not applied) |
|---|---|---|---|---|
| D3 REVISE | C3; §4 | Partly | "A stage that does not recognise its input must decline, not substitute." | Depth-index ambiguity is a structural halt (D3), not a resolution-point guess; monotonicity is either direction (D88). |
| D7 REVISE | C2; C5 | No | *(none on TVD or computed-geometric roles)* | TVD is a computed geometric reference that tracks the index (D7, D32/F2); resample it with the index, do not treat it as MEASURED_LOG. |
| D8 RESOLVED (tolerance still OPEN via D12) | C3; §4 | Partly | "A stage that cannot act on an unknown skips it and records why." | Header/data disagreement is several conditions (D8/D86), not one skip: in-tolerance record; endpoint rounding needs explicit acceptance; non-monotonic/duplicate/grid-inconsistent is structural, no override; ambiguous index halts. |
| D9 RESOLVED | C5 | Partly | "The original loaded frame is never overwritten." | Depth-copy is additive (original retained). For log curves, canonical names are additive only under the interoperability profile (D85), never a silent replace. |
| D10 RESOLVED | C5; C6 | Partly | "No transformation defaults to on." | Standardised names, when produced, are extra curves on explicit selection (D85); source mnemonics are never replaced. |
| D11 OPEN+REVISE | C3; §4.6; §7.1 | Partly | "Never validate against a fallback range." | Split limits into domain / hard-impossibility / plausibility / consistency (D87). Low family or unit confidence → not eligible for automatic range validation, never invalid (D11). |
| D12 OPEN | none | No | *(none; no numeric header/data tolerance)* | *(no contract text until the tolerance number and unit exist)* |
| D13 OPEN | §8 Phase 1 (KEOUGH 8 ft) | Partly | "Header `BLI1` says 5326 ft; the density tool on this file reads 4250–**5334** ft — resolve the eight-foot discrepancy before freezing the golden file." | Do not freeze KEOUGH over the disputed interval until D13 is closed; two readings: skip 8 ft, or skip the whole density curve. |
| D15 REVISE | none | No | *(none on licensing)* | *(licensing is out of C1–C7; clock-rollback is D15/F11, not a pipeline contract)* |
| D16 OPEN | §8 | Partly | "Contract work does not start until the pipeline produces correct output once." | Two readings: (a) v1 is Stages 0–10 (D16); (b) first sellable milestone is Stages 0–4 read-only (external note on D16). Contracts §8 assumes repairing the legacy pipeline first; D4 says C1–C7 govern the rebuild. |
| D21 REVISE | C3 | Partly | "Confidence 0.00 / type UNKNOWN must propagate as a first-class state, not degrade into a default." | The data model is the named operational fields in D21, not a three-state trust enum. User-facing summary may stay three-state. |
| D28 REVISE | §6 | Partly | "A stage that cannot satisfy its own `Requires` fails loudly at that boundary." | Document the Stage 1 exception: positional index needs no mnemonic table; depth-alias copy (D9) may use a small declared set at end of Stage 1 (D28/F10). |
| D30 REVISE | none | No | *(none on identity keys)* | Curve identity is (file hash, column index) or (dataset id, column index), not a bare column index (D30/F9). |
| D32 REVISE | C3 | Partly | "type UNKNOWN must propagate as a first-class state" | Role taxonomy includes a computed-geometric-reference role for TVD (D32/F2), in addition to INDEX / MEASURED_LOG / DRILLING / ANCILLARY / SYNTHETIC_SOURCE / UNKNOWN. |
| D41 NEW | C2 | No | *(tool grouping not named)* | Families carry a tool-grouping property; Stage 2 before Stage 4 (D41). Shared interior nulls across a group are evidence consistent with tool absence, not proof (D49). |
| D42 NEW | C2; §4 | No | *(interpolation domain not named; §4.6 is outlier strategy)* | Families carry interpolation_domain distinct from display log-scale (D42). Non-positive endpoints refuse log-domain fill (D52). |
| D43 STRUCT | C5; C7.2 | Partly | "The original loaded frame is never overwritten." / "never against row index" | Source arrays stay immutable. Fills/derivations/smoothing are overlays on the active dataset. Resampling produces a child dataset, not an overlay (D43/F1). Two readings of "active dataset": source-only until Stage 5, or any current child. |
| D49 REVISE | C2 | Partly | "'No tool ran at this depth' and 'value absent for another reason' are different facts and must remain distinguishable end to end." | Classify by evidence with confidence: (1) outside measured interval → absent; (2) interior shared across a tool group → evidence of tool absence; (3) shorter than tool vertical resolution → dropout; (4) else → ambiguous. Non-fillable by default in (1)–(2). |
| D52 REVISE | C2; C3 | Partly | "A stage that cannot act on an unknown skips it and records why." | Interpolation domain is D42, not display `log_scale`. Log-domain fill with a non-positive endpoint is a scientific-precondition refusal (D52, D86). |
| D53 REVISE | C3 | Partly | "Skipping is a logged, visible outcome — not a silent pass." | `MIN_POINTS_BY_METHOD` applies to model fitting (Stage 6 and inferential Stage 7), not to two-point bounded interpolation. Confirm table intent before wiring (D53/F8). |
| D62 RESOLVED | C5 | Partly | "The original loaded frame is never overwritten." | Original mnemonics and order hold for the auditable-processed profile. Interoperability may add aliases (D85). D62's word "forensic" does not appear in D85 — see Table 4. |
| D65 PARTIAL | none (ingest) | No | *(no LAS parse contract)* | Only first character after `~` is significant; A-section boundary; split on first period and last colon; column count equals `~CURVE` count after WRAP=YES, comments and blanks; negative STEP is valid (D65, D88). Corroborated, not CWLS-verified. |
| D66 PARTIAL+REVISE | C2; C3 | Partly | "No stage may silently convert absent → present." / "Absence returns absence." | Match declared NULL with a tolerance; never assume a NULL when undeclared. Sentinel scanning is a warning heuristic, never a silent conversion. Do not flag scientific notation as nonconforming (D66/F19). |
| D67 UNVER | none | No | *(none on STEP=0)* | Two readings: (a) write STEP=0 when increment is not constant (D67, UNVER); (b) accept irregular-but-monotonic grids as irregular and do not force a convention (D88). |
| D68 PARTIAL | none | No | *(none on LAS version)* | Target LAS 2.0; detect and report v3 features rather than parsing them optimistically (D68). |
| D69 REVISE | none | No | *(none on companion flags)* | Companion flag curves are evidenced in one fixture; generality is inferred, not proved (D69/F18). |
| D73 REVISE | C7.1 | Partly | "Synthesised samples must be visually distinct." | At full-well scale distinguish modified from unmodified only; finer states need zoom (D73/F12, UXC6). |
| D77 REVISE | C7 | Partly | "No plot displays processed data alone." | A difference view for changed samples is worth having. Do not claim competitors lack it (D77/F18). |
| D79 REVISE | C6 | Partly | "Processing is an invoked action, not a side effect" | Disposition gating applies only to dispositionable findings. No general force-export (D79/F17, D86). Structural and export-conformance blocks are not four-state queue items. |
| D85 NEW | C5; §5 | No | "Every processed output carries a manifest." | Three profiles: original input (never written); auditable processed (source names/order preserved, interpretation in annotations/manifest); interoperability (additive canonical aliases, explicit selection) (D85). |
| D86 NEW | C3; C6 | Partly | "A stage that cannot act on an unknown skips it and records why." | Every finding carries a block type: informational · ambiguous interpretation · rule-dependent warning · scientific precondition failure · structural data failure · export conformance failure. No general force-export override (D86). |
| D87 NEW | C1; C3; §4 | Partly | "must assert that the declared unit matches the reference's unit before comparing." | Bounds and eligibility live in a versioned rule library (family, units, rule type, source, tier, version, overridability) (D87), not in stage code. |
| D88 NEW | none | No | *(none on depth direction)* | The depth axis may ascend or descend. Negative STEP is valid. Monotonicity is either direction; ordering is recorded, never silently reversed. Irregular-but-monotonic grids stay irregular (D88). |
| UXC1 | C3 | Partly | "UNKNOWN must propagate as a first-class state" | UI labels derive from D21 operational fields plus block type; never from "validated / flagged / unknown" alone. |
| UXC2 | C3; C6 | Partly | "Skipping is a logged, visible outcome" | Every finding shows its D86 block type and only the actions that type permits. No "accept anyway" / "force export". |
| UXC3 | C4 | Partly | "Either recompute after the state changes, or carry the source assumption and assert it still holds at point of use." | Stale state is visible on the stage rail, on each affected item, and in an impact summary naming the trigger and recovery path. |
| UXC4 | C6 | Partly | "Before processing runs, the user is shown the list of transformations that will be applied" | Any decision that can propagate requires explicit scope and a preview of the affected population before commit. |
| UXC5 | C6 | Partly | "Loading a file transforms nothing." / "Processing is an invoked action" | Presentation controls (track set, zoom, overlay visibility, presets) must not alter identity, eligibility, classification, transformations, export contents, or recorded decisions. |
| UXC6 | C7.1 | Partly | "Synthesised samples must be visually distinct." | Full-well: modified vs unmodified, plus major absent/refused intervals. Created/changed/invalid/refused are not all independently claimed at that scale. |
| UXC7 | C2; C3; C7.4 | Partly | "Absent data renders as a gap." / "A stage that cannot act on an unknown skips it and records why." | A refused method or gap is a completed outcome: what was not done, why, span, whether raw remains, whether export is blocked. |
| UXC8 | C5; C7.2 | Partly | "Raw and processed are both addressable at all times." / "never against row index" | After resampling, the UI names active dataset, parent, grid relationship, and must not imply source and child share sample indices. |
| UXC9 | C3 | Partly | "Never validate against a fallback range." | Distinct outcomes: passed / failed / warning / not assessed / not eligible. Absence of a violation is not a pass when no eligible rule ran. |
| UXC10 | C5; C6; §5 | Partly | "Every processed output carries a manifest." | Export review shows profile, lineage, transformations, dispositions, non-overridable blockers, stale dependencies, and included provenance. Confirm only when exportable. |

---

## Table 2 — Contracts → register

Normative contracts language that no register row covers, or that a register row contradicts. Un-flagged rows still count as coverage. Quotes that a row covers without tension are omitted.

| Contracts quote | Coverage | Register row |
|---|---|---|
| "Any stage that transforms values **must** update the declared unit in the same operation." | **Two readings.** (a) Contradicts D24 if Stage 3 conversion rewrites stored numbers. (b) Compatible if conversion is display/check/export only and stored numbers never change. Reading (b) is in the build plan (OD-05), not in the register. | D24; D25 |
| §3: Transformation "Changes a *known* value" is allowed "**Anywhere**, provided the declaration is updated" | **Contradicts** D24 ("Stages 0–4 touch no data") if "anywhere" includes ingest through validity. Two readings: "anywhere" among transforming stages; or any stage including 0–4. | D24 |
| "A unit mismatch is a hard error, not a warning, not a coercion." vs C7.3 "The plot must either convert for display with the conversion stated, or refuse" | **Two readings** of whether a plot may convert without rewriting stored numbers. C1 calls mismatch a hard error; C7.3 offers a display conversion. | D87; C7.3 vs C1 |
| "Extrapolation beyond the measured interval is forbidden **by default**." | **Contradicted / tightened** by D49(1) and D50: outside the measured interval is absent and never fillable, not an overridable default. | D49; D50 |
| C7.1 "Interpolated, **extrapolated** and gap-filled segments render differently" | **Tension** with D50: if extrapolation never occurs, the extrapolated visual encoding has no legal samples. Two readings: keep the encoding for a refused-but-attempted case; or drop "extrapolated" from C7.1. | D50 |
| "The UI displays them together — side by side or overlaid — for any curve, at any point." | **Tension** with D75: original-only is an allowed view. C7 forbids processed-alone, not original-alone. | D75; D76 |
| "No plot displays processed data alone. Raw and processed appear on the same axes, **always**." | **Contradicts** D75 original-only unless "always" applies only when processed data is shown. Two readings. | D75 |
| "A sample of the original file (header + first/last data rows) is viewable without leaving the app." | **Uncovered.** No D-row requires an in-app raw-text sample viewer. D17 is export pairing; D75 is plot views. | none |
| "Every place the system **must guess** is one of them." | **Tension** with D3/D8: some unknowns are structural refusals, not guesses. | D3; D8; D86 |
| "Every resolution point hit, what was chosen, and by whom (**user or default**)" | **Tension** with C6 "No transformation defaults to on" and D79 "Silence is not one of the states." Two readings of "default": a recorded chosen default vs an implicit one. | C6; D79 |
| §7.2 `assert interpolation.limit is not None` | **Partly contradicts** D50/D52: some fills are refused outright; a mandatory limit is not the same as a refusal. Two readings: limit applies only when a fill is eligible; or every interpolation call carries a limit even if the stage then declines. | D50; D52 |
| "Contract work does not start until the pipeline produces correct output once." | **Contradicts** D4 (C1–C7 already govern) and the rebuild-beside decision; §8 is a legacy-repair sequence. | D4; D27 |
| §8 **Phase 0–4** (repair → golden files → contracts → extraction → UI) | **Contradicts** D6 (one numbering scheme) and D23 (Stages 0–10). D27's order "golden files → contracts → extraction → UI" matches Phase 1–4 names but not "repair the monolith first." | D6; D23; D27 |
| "Process KEOUGH. Expect 10720 rows (with units off)" | **Tension** with D5 (toleranced fixture, not frozen) and D13 (8 ft still OPEN). | D5; D13 |
| "If C7 is non-negotiable, the toolkit shortlist is effectively Tkinter or Qt." | **Uncovered.** No register row on toolkit. | none |

---

## Table 3 — Brief vs register (rev 2)

| Brief locus | Register row | Conflict |
|---|---|---|
| §3 "Every other tool in this category can tell you *that* a curve was processed. None of them can tell you *which samples*…" | D77 / F18 | Competitor absolute. F18: unverifiable; strike. Same class of claim as "nobody ships it" in §4.5. |
| §4.1 "Every change is stored as a sparse overlay keyed to sample index." | D43 STRUCT / F1 | Overlay cannot represent resampling. Resampling must be a child dataset. Brief §11 still lists "the additive overlay model" as not-to-redesign. Two readings after OD-03 (not in the register): overlay for fills/derivations/smoothing; child only for resampling. |
| §4.2 categories: measured / created / changed / invalid / **refused** | D44 | D44 is measured / created / changed / invalid / **absent** / **resampled**. No sample-status "refused". Brief "refused" is a gap POLISH declined to fill — that is issue_disposition or a method outcome (D79, D86), not sample_status (D21/F4). |
| §4.2 "Changed — … unit conversion. Two values at one depth." | D24; D44 | D24: Stages 0–4 touch no data; Stage 3 is Units. If conversion rewrites stored numbers, §4.2 contradicts D24. If conversion does not rewrite stored numbers, §4.2 is wrong to call it "changed". Register has no OD-05 row. |
| §4.8 four terminal states for **every** raised item; export gated on disposition | D86; D79 REVISE | Not every finding is dispositionable. Structural data failure and export conformance failure are not four-state items. No general force-export. Brief has no block types and no "must not override" class. |
| §5 "Ten stages" then a table numbered **0–10** | D16; D23; D6 | "Ten" vs eleven stages (0 through 10). Same wording in D16 ("all ten stages") and D23 ("Ten stages"). |
| §4.5 "A *difference* view … nobody ships it." | D77 / F18 | Same competitor absolute as §3. |
| §4.3 / band table: full-well band carries provenance states | D73 / F12 | Full-well should distinguish modified from unmodified only; four-to-five states in one band is too tight. |
| §5 Stage 1 "May **refuse the well** if header and data disagree." | D8 / D86 | Disagreement is several conditions, not one refusal. In-tolerance may record and permit. |
| §6 "Curve identity is column index, not name." | D30 REVISE / F9 | Index is not stable across files. Key is (file hash, column index) or (dataset id, column index). |
| §6 role list: INDEX … UNKNOWN, no computed-geometric-reference | D32 / F2; D7 | TVD has no role in the brief list; DAD decision is that TVD resamples with the index. |
| §6 "Original mnemonics are never overwritten." / §9 anti-pattern "Renaming curves to canonical names in the output file" | D10 / D62 / D85 | Compatible with auditable-processed. Interoperability **adds** canonical names; it does not overwrite. Brief anti-pattern over-states if read as forbidding additive aliases. |
| §11 "Disposition gating and the four terminal states" as settled | D79 REVISE; D86 | Settled only for dispositionable findings. |
| §11 "The ten-stage pipeline and its ordering" as settled | D16 OPEN | D16 is OPEN: full pipeline vs Stages 0–4 read-only first. |
| §7.1 default tracks "lithology/gamma, then resistivity, then porosity/density" | *(no register row)* | Not a register conflict. Domain-expert track order lives in the build plan, not in a D-row. Listed so it is not mistaken for silence in the register. |
| §4.7 / refusals panel as selling point | D29 | Aligns; not a conflict. |
| §4.9 stale blocks export | D80 | Aligns; not a conflict. |

---

## Table 4 — Cross-reference errors

Checked in the five documents read this task, plus D82's prefix count against `tests/fixtures/159-19A_LFP.las` (~C text, not lasio).

| Place | Citation | What the cited row actually is | Mismatch |
|---|---|---|---|
| Register D21 | "sample status **(D38)**" | D38: alias and qualifier tables are local files, never transmitted. Sample status values are **D44**. Curve role is D32 (that half of the sentence matches). | Wrong D-number. Same class of error revision 2 already corrected for D54/D63. |
| Register D62 | "forensic and auditable-processed profiles **(D85)**" | D85 names: (a) Original input; (b) Auditable processed; (c) Interoperability. No profile named "forensic". | Name does not match D85. Two readings: "forensic" means (a), or it is a fourth name for (b). |
| Register D16, D23; brief §5, §11 | "ten stages" / "all ten stages" | The same documents number stages **0–10** (eleven). | Count word vs numbering. |
| Register D82; brief §8; supplement UXC4 / Scenario 4 | "147 curves" share `LFP_` / "24 without the prefix" | This task: 171 ~C lines, **170** mnemonics begin `LFP_`, exception `DEPTH`. | Count does not match the fixture. Two readings: 147 is a different subset (e.g. re-match after strip); or the number is simply wrong. |
| Register Count "DAD 7" | source tally | DAD rows in the tables: D7, D8, D9, D11, D12, D13 = **6**. (D10 is AC.) | Tally vs rows. Ready-for-P05 already flags "source tally". |
| Register Count "AC 18" | source tally | AC rows counted from the tables this task: D4–D6, D10, D14–D19, D21–D22, D37–D38, D45, D61, D75–D76, D78 = **19**. | Tally vs rows. Two readings: one AC row is mis-tagged, or the count is off by one. |
| Register "For the reviewer" point 4 | "three questions belong to the domain expert … (D9, D11, D12, D13)" | Lists **four** IDs. D9 is RESOLVED. | "Three" vs four; D9 is not still an expert question on the naming conflict. |
| Findings index F3 | "Refuse and skip contradicts disposition gating" / Severity Conflict | D8 is RESOLVED under D86; CONFLICT count is 0. | Index not updated after resolution. |
| Supplement Part E "Phase 0–3"; contracts §8 "Phase 0–4" | Phase numbering | D6: one numbering scheme; README: no Phase numbers; Stages 0–10 only. | Different Phase 0 in two governing docs. Citation of a retired scheme, not a wrong D-number. |
| Contracts §8 / §4 | "13,867 lines" | Not a D/F/C/UXC citation. P01 measured 14,382. | Stale fact, not a handle mismatch. |
| UXC1 → D21 | D21 conflation | Matches F4/D21. | No error. |
| UXC2 → D79, D86 | block types; no force export | Matches. | No error. |
| UXC7 → D52, D86 | scientific-precondition refusal | D52 non-positive log fill; D86 scientific precondition failure. | No error. |
| UXC8 → D43/F1 | overlay vs child | Matches. | No error. |
| UXC9 → D11, D87 | eligibility vs invalid | Matches. | No error. |
| UXC10 → D85, D86, D60 | profiles, no force export, round-trip | Matches. | No error. |
| D8 text | "mis-cited as D63 in revision 1" | D63 is input-file hash; disposition is D79. | Historical error, already flagged in the row. |
| D9 text | "cited D54 in error; the relevant row is D62" | D54 is regeneration records. | Historical error, already flagged in the row. |

---

## Contract changes that would be needed before Stage 0 code is written

Stage 0 is ingest (brief §5). Ordered. Each is a needed contracts amendment, not an applied edit.

1. Retire §8 Phase 0–4; name Stages 0–10 only (D6, D23). — D6
2. Strike "contract work does not start until the legacy pipeline produces correct output once"; C1–C7 govern the new ingest (D4). — D4
3. Add an ingest contract: two-pass raw-text audit then tolerant parse; every ambiguity in an audit ledger; never silent `ignore_header_errors` (D64). — D64
4. State LAS 2.0 parse rules as PARTIAL: first character after `~`, A-section boundary, first period / last colon, WRAP=YES / comments / blanks in column counts (D65). — D65
5. Negative STEP and descending or irregular-but-monotonic depth are valid; do not reverse silently (D88, D3). — D88
6. Null: use declared NULL with a tolerance; never invent a NULL; sentinel scans warn, they do not convert (D66). — D66
7. Record the D67/D88 clash on blank or irregular STEP as unresolved; ingest must not pick one (D67 UNVER, D88). — D67 / D88
8. Detect LAS v3 features and report them; do not parse them as v2 (D68). — D68
9. Identity at ingest is (file SHA-256, column index), not a bare index (D30, D63, D46). — D30
10. Input file is never opened for writing (C5, D17). — D17
11. Declared NULL and truly absent stay distinct in the validity mask from the first parse (C2). — C2
12. Unknown mnemonic and blank/unrecognised unit stay unknown; no fallback range or guessed unit at ingest (C3, §4.1–4.2, D11). — C3
13. Parse/structural failures use D86 block types; they are not four-state disposition items and have no force-export path (D86, D79). — D86
14. Do not implement a "scientific notation is nonconforming" parse rule (D66/F19). — D66
