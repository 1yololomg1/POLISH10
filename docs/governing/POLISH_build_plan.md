# POLISH — Build Plan and Cursor Prompt Pack

**Prepared Tuesday 15 September 2026.** Also published as a Notion page tree ("POLISH — Build Plan"); this file and Notion hold the same content. **Revised the same day** with your answers to OD-03, OD-05 and OD-06, and your father's worksheet answers (worksheet dated 14 September). **Revised 16 September** with the six follow-up answers in `docs/governing/POLISH_followup_answers.md`.

Built from four documents and one inspection:

- `POLISH_design_brief.md` (11 Sept)
- `POLISH_decision_register.md`, revision 2 (13 Sept)
- `POLISH — UI/UX Cohesion Supplement` (undated; its UXC items are marked *proposed*)
- `docs/governing/POLISH_pipeline_contracts.md` (read from the repo today)
- A read-only inspection of `C:\porfolio\POLISH10` today (git state, file sizes, fixtures, agent rule files). Nothing was written to your machine.
- `POLISH_questions_worksheet` — your father's answers, 5 pages, dated 14 Sept (a scanned PDF; read visually)
- `C:\Users\achav\Downloads\LAS_AUDIT_OUTPUT\master_audit_report.csv` — read today to test his depth-tolerance rule against 144 real wells

---

## How this document is numbered

This document adds exactly two handles and nothing else:

- **OD-nn** — an open decision listed in Part 1
- **Pnn** — a Cursor prompt in Part 5

Everything else keeps the handle its source document gave it: **Stage 0–10**, **D1–D88**, **F1–F19**, **UXC1–UXC10**, **DEP-UX-nn**, **C1–C7**. No "Phase" numbers appear here — milestones are named, not numbered, because the contracts document and the supplement each already have a different "Phase 0" (see OD-08).

Slots in prompts look like **⟦OD-nn: …⟧**. A prompt with an unfilled slot is not ready to send.

---

# PART 1 — DECISIONS NOT YET MADE

## 1.0 What has been answered (15 Sept; follow-up 16 Sept)

### Decided by you

- **OD-03 — resampling makes a new copy of the well.** The original stays untouched; the copy records how it was built. Two consequences added to P23: provenance is stored **per depth interval, not per sample** (per-sample detail is worked out on hover), and P23 must include a memory-and-speed test on the largest fixture. Measured today: the largest fixtures hold about 5–6 MB of values in memory (`159-19A_LFP.las`: ~3,905 depth steps × 171 curves; `WLC_PETRO…`: 33,191 × 23), so a copy is small; drawing and provenance storage are where the real cost is.
- **OD-05 — unit conversion is not a change to the data.** Stored numbers never change and the unit always travels with the number (C1). Conversion happens whenever a value is used — on screen, in checks, in formulas, at export. Four cases:
  1. *What unit a curve is in* — read from the file; if blank or unrecognised, POLISH asks. Never guessed.
  2. *Conversions needed internally* (a rule or formula written in other units) — automatic, exact, recorded; only when both units are certain, otherwise the check or formula does not run and the queue says why.
  3. *Which units the exported file uses* — the user chooses; default is as recorded; the choice is a recorded decision.
  4. *Pairs with no fixed factor* (e.g. gamma ray in CPS vs API units, which depends on tool calibration) — refused, never invented.
- **OD-06 — build beside the old app**, in a new `polish/` folder. The old app stays untouched and runnable until replaced.

### Answered by your father (worksheet, 14 Sept; follow-up 16 Sept)

The worksheet does **not** answer OD-01. Its question 2.1 ("which one wins") is about which *depth curve* governs, not which *document* governs. OD-01 is still yours. Follow-up quotes below are from `docs/governing/POLISH_followup_answers.md`.

| Worksheet | His answer | What it settles | Follow-up needed? |
|---|---|---|---|
| 1 Limits (D11, OD-12) | GR 0–300 GAPI · RHOB 1.5–3.5 g/cc · NPHI −0.15–0.6 v/v · DT 40–300 us/ft · PEF −2–270 b/e · resistivity −0.02–20,000 ohm·m. **SP: −150 to 50 mV, flagged at the log level only** — a log is flagged for review when more than 20% of its samples fall outside that range; individual out-of-range samples are not flagged on their own. Supersedes the 16 Sept "No SP limit" answer (see 1.0.1). | Values for the rule library (P12), including SP as a log-level 20% rule | Types: see 1.0.1. PEF type left blank. "Any curve where the limit should depend on something else" left blank |
| 1.2 Out-of-limit values | Prompt whether to delete the value, for this well only or all wells. Never clip. Quote: "Delete means mark as \"no value\" visually -999.25" | Internal NaN; export sentinel −999.25; value stays visible on screen (marked) and untouched in the original file; decision recorded in the report. Scope *this well* or *all wells* | No |
| 2.1 Two depth curves (D3) | The first curve in the file | Index = first curve (P09). The second depth curve is kept as-is. | No |
| 2.2 Depth drift tolerance (D12, OD-10) | Diagnose further: compute implied sampling interval as (STOP − START) / (NSAMPLES − 1), compare against known standard intervals in feet and metres including cross-unit matches (0.1524 m = 0.5 ft), treat as rounding only on a match, then correct the depth track. Quote: "Does (stop - start)/(nsam-1) match the sampling interval? … 0.1524 m which is half fot in meters" | A matching rule, not a single tolerance number (D12) | Resolved 23 Sept — YES (D12 Part Two): a corrected well's exported STOP is rewritten to STRT + (n−1)×S_corrected; the original file is never altered. |
| 2.3 Blank STEP | Check the depth curve and fill the step if appropriate when increments are regular. If depths are genuinely uneven: Quote: "Accept it as unevenly spaced, don't invent a step, and tell me" | Matches D88. Does not refuse irregular wells | No |
| 2.4 KEOUGH 8 ft (D13, OD-11) | Skip and notify. Quote: "Only the density readings below 5,326 ft (5,326.5 - 5,334 ft)" | Skip only that density interval; the rest of the density curve is unaffected | No |
| 3 Unknown mnemonics (RM, RD, RACEHM, RACELM, RPCEHM, RPCELM, ABDCQF01–04, NBGRCFM, ROP) | Leave unchanged | They stay UNKNOWN, are preserved in the output, and are not processed (C3). | Confirm: "leave unchanged" = "don't process them", not "they're resistivity, keep the name" |
| 3.1 RM | Leave unchanged and notify | Same, plus a notice in the queue | No |
| 3.2 Main vs repeat pass | Show a selection to the user, with the option to apply to this well only or all wells | A queue decision with scope *this well* / *all wells* (P10, P13). Also answers OD-22: an explicit "all wells" choice is wanted. | No |
| 3.3 Common mnemonics | (blank) | Recognition stays near half on the Norwegian wells for now; P10's candidate list is the route to improve it | Later |
| 4.1 First three things | Check headers (depths) and consistency with the depth log · verify units · verify depth increment matches step | Confirms Stage 1 → Stage 3 → Stage 1 grid check as what the user looks at first. The describe view should lead with the depth-axis and units results. | No |
| 4.2 Default tracks, left to right | DTP, DTS, density, porosity, Sw, GR, ILD. Quote: "Use the nearest equivalent (for example RDEP in place of ILD) and label it clearly" | Default track order for P20. **Overrides brief §7.1** (see 1.0.3). Absent default curves: nearest equivalent, labelled clearly | **Yes** — the ILD substitute list was asked twice and left blank both times |
| 4.3 Smoothing | Almost never | Stage 8 (denoising) drops to lowest priority | No |
| 4.4 Gaps | Depends on how big the gap is | Consistent with D49 cases 3–4, but no numbers yet (OD-13) | Later, before Stage 7 |
| 4.5 What other tools get wrong | No QC images | **New requirement:** POLISH exports QC images. Recommended for the read-only milestone (P22). | No |
| 5.1 LAS 2.0 / 3.0 | (blank) | Stay on LAS 2.0 (D68) | No |
| 5.2 Filled-sample marking in the file (OD-21) | Only when asked for | Opt-in per export | **Conflicts with D61** — see 1.0.3; your call, not needed until P28 |
| 5.3 Filled gaps to clients | I'd ask the client | The export profile is chosen per job (D85) | No |
| 5.4 Keep from old version | (blank) | — | — |

### 1.0.1 Follow-up on the limits (answered 16 Sept)

The worksheet asked for "the absolute bound where the number cannot be right, ever". F15 splits limits into *impossible* (the value gets flagged as invalid) and *suspicious* (a warning; the value stays measured). The 16 Sept follow-up classified the worksheet numbers as follows (quotes from `POLISH_followup_answers.md`):

| Curve | Worksheet numbers | Type |
|---|---|---|
| GR | 0–300 GAPI | "Suspicious all three." |
| RHOB | 1.5–3.5 g/cc | "Suspicious" |
| NPHI | −0.15–0.6 v/v | "Suspicious" |
| DT | 40–300 us/ft | "Impossible" |
| PEF | −2–270 b/e | *(classification column left blank)* |
| Resistivity | −0.02–20,000 ohm·m | "Suspicious" |
| SP | −150 to 50 mV | **Log-level threshold, not per-sample.** A log is flagged for review when more than 20% of its samples fall outside −150–50 mV; individual out-of-range samples do not trigger a flag alone. Supersedes the 16 Sept "No SP limit" answer — the numbers were never rejected, the per-sample flagging behaviour was. Final answer, 18 Sept. |

Impossible: "Leave out of the processed file when values are impossible." Suspicious: "POLISH shows a warning and asks whether to eliminate these points. If yes, then remember this answer for all wells." "Any curve where the limit should depend on something else" was left blank.

Two of the remaining numbers would flag real data:

- **GR maximum 300 GAPI.** KEOUGH's only values above 300 are in the top 1.5 ft (563 at 0 ft), almost certainly a surface artefact. The 171-curve well does read up to 1,568 API at 3,703–3,705 m. Classified suspicious.
- **RHOB minimum 1.5 g/cc.** In `30_3-5 S.las`, four samples below 1.5 (minimum 1.332) sit at 4,126.0–4,126.5 m, where the file's own lithology column reads code 90000 (coal in the FORCE 2020 key, as I understand it). The `LFP_COAL` flag in the 171-curve file is zero everywhere, so it gives no evidence either way. Classified suspicious.
- **PEF maximum 270** matches barite's PEF (about 267), so it reads as deliberate. The minimum of −2 and the resistivity minimum of −0.02 are below what is physically possible, so they look like allowances for tool noise rather than physics. PEF type is still unanswered.

**Scan of the five test wells against his limits (15 Sept, read-only):**

| Limit | What the test wells show |
|---|---|
| GR 0–300 | `159-19A_LFP`: 11 above 300 (max 1,568) at 3,703–3,705 m · KEOUGH: 4 above 300, all at 0–1.5 ft · `WLC…F-1B`: max 297.8 |
| RHOB 1.5–3.5 | `30_3-5 S`: 4 below (coal code) · `34_10-19`: 119 below (min 1.379) at 289–915 m, lithology code 30000 |
| NPHI −0.15–0.6 | `159-19A_LFP`: 12 above, including 6.9, 8.8, 12.1, 15.7 and 61.9 in a curve labelled v/v · `34_10-19`: 45 above (max 0.69) |
| DT 40–300 | KEOUGH: 7 below 40 at 5,352–5,355 ft, falling to 0.002 |
| PEF −2–270 | `30_3-5 S`: 1 above (334.5 at 3,898.5 m) · KEOUGH's PE has no unit in the header |
| Resistivity −0.02–20,000 | `WLC…F-1B` RT: 40 above (max 97,540) · KEOUGH RILD 442 and RILM 1,394 above, reaching 100,000 |
| SP −150–50 mV, log-level 20% | `34_10-19`: 6,703 of 13,224 above 50 mV (~51%) — this log flags for review · KEOUGH: 866 above, 60 below (min −200 at 4.5 ft) |

### 1.0.2 Follow-up on the rounding rule — measured on 144 wells

From `master_audit_report.csv` (read 15 Sept):

| STOP: header vs last data depth | Wells |
|---|---|
| Identical | 25 |
| Different only in the last decimal the header was written to (true decimal rounding) | 4 |
| Differ by less than half a sample interval | 58 |
| Differ by half to one sample interval | 56 |
| One sample interval or more | **0** |
| Not computable | 1 |

STRT matched exactly in 143 of 143 computable wells. Example: `15_9-15` declares STOP 3210.0056 m; its last data row is 3209.856 m; the step is 0.152 m — short by 0.98 of a step.

The 16 Sept follow-up did not tick either of the two numbered boxes. The written rule is (quote): "Does (stop - start)/(nsam-1) match the sampling interval? Can the difference be attributed to rounding? If so, correct as follows: Consider common sampling intervals (fraction of feet when in feet, fraction of metric when in metric - sometimes sample of feet converted to metric - ie. 0.1524 m which is half fot in meters). Take determined sampling interval and correct depth track."

That is a matching rule, not a single tolerance number. It is recorded on D12. These 144 wells are all Norwegian (Volve / FORCE 2020). KEOUGH is not in this set.

The second half of the question — whether "the processed file's header states the real last depth from the data, and the change is recorded. The original file is never altered" — has both Yes/No boxes blank and the written line "See the previous question". That is not a yes or a no. Needs an explicit yes/no from Alvaro before D12's second half can be closed.

### 1.0.3 Where his answers conflict with the plan

- **4.2 vs brief §7.1.** The brief (until the Pre-P05 fix) said tracks go gamma → resistivity → porosity/density, "do not innovate". His order starts with sonic and density — a well-tie layout, which fits a geophysicist. On petrophysics and workflow, his rows outrank DESIGN rows (the register's own finding). **His order is the default** (DTP, DTS, density, porosity, Sw, GR, ILD). Absent default curves: nearest equivalent, labelled clearly. The conventional order becomes a saved view. The ILD substitute list was asked twice and left blank both times.
- **5.2 vs D61.** D61 (your decision) says status curves go into every export, and brief §11 makes "provenance survives into the exported file" non-negotiable. He wants the marking only when asked. Options: (a) opt-in columns, but every export always carries a sidecar manifest plus header lines naming it, the input hash and counts of created/changed samples; (b) default on, with an opt-out. **Not needed until P28.** Note that a read-only export has nothing filled, so the question doesn't arise before then.
- **1.2 vs my P12 draft.** I had out-of-limit values excluded automatically and recorded. He wants the user asked. "Delete" means internal NaN, −999.25 on export, still visible on screen (marked). P12 follows him.
- **2.3 vs D88.** Resolved 16 Sept: genuinely uneven depths are accepted as irregular; POLISH does not invent a step; the user is told. Matches D88.

### Decided by you — vocabulary (OD-04, 15 Sept)

- **"Refused" is split into three phrases:** "gap not filled", "declined by POLISH", "method not run". It is never a sample status.
- **The user's "refuse" key means "reject POLISH's suggestion".** It records a RESOLVED decision with the rejection as the choice.
- **Curves the user maps themselves are trusted for processing** — **and clearly labelled as user-trusted, not identified by POLISH**, everywhere they appear: hover, record, queue, report, manifest and export annotations. Wording: "User-mapped — not identified by POLISH".
- **A postponed (DEFERRED) item does not block export.** Every deferral, with its reason, is listed in the export.

### Quick confirmations — owner stated no preference (15 Sept)

You marked no preference on OD-01, OD-02, OD-08, OD-26 and the branch name. P02 therefore uses the recommendations as written below. Change any of them before sending P02 if you want something different.

- OD-01: contracts C1–C7 are the core rules; the newest register decision beats older text; the domain expert's answers beat DESIGN-tier proposals on petrophysics and workflow; anything uncovered, Cursor stops and asks.
- OD-02: adopt UXC1–UXC10, keeping their handles.
- OD-08: "Stages 0–10"; no Phase numbers.
- OD-26: commit the four fixtures with Git LFS, after you check each file's licence.
- Branch: `rebuild`, from the current HEAD of `freeze-fix-wip`.

### 1.0.4 Still needed before building starts (P02–P08)

| Item | Needed by |
|---|---|
| Licence check on the four fixtures (you) | P02 |
| P08 ledger-to-block-type mapping | P08 |
| Explicit yes/no from Alvaro on D12's second half (processed header states the real last depth) | P09 |
| PEF impossible/suspicious classification (left blank 16 Sept) | P12 |
| ILD substitute list (asked twice, left blank both times) | P20 |

### 1.0.5 What Prompt 01 found (15 Sept)

Cursor's report was checked against the repo read-only; the claims below held.

- **Clean baseline.** Branch `freeze-fix-wip`, HEAD `f815811`. `python -m pytest -q`: **56 passed**, 0 failed. `ruff check .`: clean. CI runs ruff and pytest (with coverage) on every push, Python 3.11.
- **No network code.** No `.py` file imports a network client or contains a URL.
- **Three legacy `core/` modules import tkinter:** `core/error_handler.py`, `core/gap_filling.py`, `core/unit_standardization.py`. Importing them from the new engine would drag the old UI in. **Consequence:** `polish/` never imports `core/`, `ui/`, `petrophysics/` or the monolith — logic is copied across with a comment naming its source, and P06 adds a test that importing `polish` loads no UI toolkit (rules file hard rule 7 updated).
- **Environment.** Python 3.11.9. PySide6, pyqtgraph, pytest-qt, pytest-socket are not installed. PyInstaller 6.15.0 and cryptography 46.0.3 are installed but not listed in `requirements.txt`.
- **Curve-name table.** A Python dict in `core/curve_identification.py`: 33 entries across 16 families (each entry holds several mnemonics).
- **No test covers gap classification.** P12 is building that from zero.
- **A legacy test encodes a doubtful premise.** `test_gr_physical_bounds_keep_hot_shale_peak` treats KEOUGH's 563 GAPI as a real hot-shale peak; the value sits at 0 ft (1.0.1). Leave the legacy test alone; new tests must not inherit the premise.
- **Fixtures.** Depth steps confirmed: 3,905 (wrapped 171-curve file), 28,123, 13,238, 33,191. On the two FORCE wells, the counted steps are one fewer than `(STOP−STRT)/STEP+1` — the same header-vs-data question as your father's Q1.
- **Documents.** The design brief, register, specification v2 and an **older copy of this plan** sit untracked in the repo root. The supplement is not there. `docs/` does not exist.
- **Aikido.** Installed and enabled in Cursor (two MCP servers; an Aikido API key is configured in `C:\Users\achav\.cursor\mcp.json`). `aikido_full_scan` requires full file contents as input. Its description says it scans "locally", but also that the repository name identifies the repo "in the Aikido platform". **Whether code leaves the machine: cannot determine.** **Decided 15 Sept: keep Aikido** — it has solved hard problems on this codebase and you judge the risk worth it. One guard added to the rules file: never pass `.las` files, `tests/fixtures/`, or any well data to Aikido or any external tool — source code only.
- **Old rules are active.** The report ends with a "Context improved by Giga AI" line, which `.cursorrules` instructs Cursor to add. P02 archives it.
- **Not checked by Cursor:** GitHub branch protection on the remote, live CI status, whether an Aikido scan uploads code.

### 1.0.6 Ready for Prompt 05 — checklist

Suggested by the external review (15 Sept) and adopted. Every line must be true before P05 is sent. Commit this list with P02 as `docs/governing/ready_for_P05.md`.

- [x] Unit conversion semantics approved (OD-05)
- [x] Child-dataset semantics approved (OD-03) — **resampling only**; fills, derivations and smoothing are overlays on the active dataset
- [x] State schema approved (OD-04, Appendix A)
- [x] Clean test baseline recorded (P01: 56 passed, ruff clean)
- [ ] **Precedence rule explicitly approved by you** (OD-01) — currently a default you gave no preference on
- [ ] **UXC adoption explicitly approved by you** (OD-02) — currently a default
- [ ] Backup made of the four untracked fixtures and the governing documents, outside the repo, before P02 moves anything
- [ ] Fixture licences checked; fixtures committed with LFS (P02)
- [ ] Legacy agent rules archived; only `polish-rebuild.mdc` and `aikido_rules.mdc` active (P02)
- [x] Aikido decision made (OD-09): keep it
- [ ] P03 reconciliation reviewed with Claude
- [ ] **Document fixes committed:** brief §3 (competitor claim), §4.1 (overlay vs D43), §4.2 (unit conversion is not "changed"; no "refused" sample state), §4.8 (D86 block types, no override for structural or conformance blocks), §5 ("Stages 0–10"), §7.1 (domain expert's track order); register: D21 citation, source tally, reviewer-notes count, "147" in D82, D62 "forensic"; contracts: "Stages 0–10", Phase numbering retired

### 1.0.7 External review (15 Sept) — what was taken, what wasn't

A second-opinion review of this plan agreed with the sequencing and architecture. Taken: the Ready-for-P05 checklist (1.0.6), explicit approval rather than defaults for OD-01 and OD-02, a backup before P02, and fixing the contradicting documents before code rather than relying on the precedence rule alone. Not taken as written: it describes OD-03 as "data-altering operations create child datasets" — only resampling does. It also renders `polish_ui` as `polishui` and the monolith without underscores; the names in this plan are correct.



## 1.1 At a glance

| OD | Question | Owner | What it blocks | My recommendation |
|---|---|---|---|---|
| **OD-01** | Which document wins when documents disagree? | You | Every prompt | **No preference stated 15 Sept — recommendation used** (see 1.0) |
| **OD-02** | Adopt UXC1–UXC10 into the register? | You | P05 onward | **No preference stated — recommendation used:** adopt all ten |
| **OD-03** | Resampling lineage model (D43 / F1, STRUCT) | You | P06 onward | **DECIDED 15 Sept:** new copy (child dataset) |
| **OD-04** | Canonical state schema — fields and values | You (+ my sanity pass) | P05 onward | **DECIDED 15 Sept** — Appendix A with your four answers |
| **OD-05** | Is unit conversion a data alteration? | You | P05, P11 | **DECIDED 15 Sept:** no; units travel with numbers |
| **OD-06** | Rebuild beside the legacy app, or refactor in place? | You | P05 onward | **DECIDED 15 Sept:** build beside, in `polish/` |
| **OD-07** | Toolkit: Tkinter or PySide6 (and which plot layer)? | You | P17 onward | PySide6; run P04 before choosing the plot layer |
| **OD-08** | Numbering: "ten stages" is eleven; two "Phase" schemes | You | P02 | **No preference stated — recommendation used** |
| **OD-09** | Legacy Cursor rule files and the always-on Aikido rule | You | P02 | **DECIDED 15 Sept:** archive the legacy rules; **keep Aikido** |
| **OD-10** | D12 — header/data depth tolerance | Dad | P09 | **CLOSED.** The matching rule was answered; the rounding test is recorded under D12. |
| **OD-11** | D13 — KEOUGH 8 ft discrepancy | Dad | KEOUGH golden assertions over that interval | **ANSWERED:** skip only density 5,326.5–5,334 ft |
| **OD-12** | D11 / D87 — rule values, split into four rule types | Dad | P12 (non-definitional rules) | **VALUES RECEIVED; types received except PEF blank; SP settled 18 Sept as −150/50 mV at log-level 20%** (1.0.1) |
| **OD-13** | D41 tool groupings; D49 dropout cutoff and ambiguous threshold | Dad | P12, P23, P25 | — |
| **OD-14** | D42 — interpolation domain per family | Dad | P23, P25 | — |
| **OD-15** | D51 — metric version of the preferred threshold table | Dad | P23, P25 | — |
| **OD-16** | D53 — intent of `MIN_POINTS_BY_METHOD` | Table author | P24, P25 | — |
| **OD-17** | D7 / D32 — name and behaviour of the TVD role | Dad | P10, P23 | Role name proposal in Appendix A |
| **OD-18** | Does the auditable export carry the DEPT copy? (D9 vs D85b) | Dad | P29 | Stage 1 records the alias; export materialises it per profile |
| **OD-19** | D16 — ship read-only (Stages 0–4) first? | You | Release plan only | Doesn't block starting — see note |
| **OD-20** | D20 — manual-workflow benchmark in v1? | You | Release | — |
| **OD-21** | Per-sample method detail in the LAS: default or opt-in? | You | P28, P29 | Dad: opt-in. **Conflicts with D61** — your call (1.0.3) |
| **OD-22** | Can a batch decision become a reusable local rule? | You | P13, P15 | **ANSWERED (dad 1.2, 3.2):** yes, as an explicit "all wells" choice |
| **OD-23** | D15 — licensing: when, and what an invalid licence permits | You | P30 | Not before the read-only milestone |
| **OD-24** | Obtain CWLS LAS 2.0 document | You | P08 (hardening), P29 | Get it before P29. D67 is now RESOLVED from the 16 Sept follow-up (accept irregular, no invented step) |
| **OD-25** | Visual design — brief §10 Q1, Q2, Q3, Q8 still open | You / design session | P19, P20, P27 | Don't let Cursor invent the coverage track |
| **OD-26** | Commit the four real-well fixtures? LFS? | You | P02, P07 | **No preference stated — recommendation used:** LFS after your licence check |
| **OD-27** | What does a read-only release export? | You | P22 | Report + JSON + **QC images** (dad 4.5) + session file |
| **OD-28** | Which derivation equations are in v1? | Dad | P24 | — |

**Read this before worrying about OD-19.** D16 looks like the biggest open item, and it is the biggest *commercial* one, but it doesn't block starting. Stages 0–4 and the read-only interface have to be built first under either answer (D24, D26, D27, and supplement Part E all agree). D16 only decides whether you release at the end of the read-only milestone or keep going to Stage 10 first. Everything up to P22 is the same either way.

---

## 1.2 Decisions that block scaffolding — answer before P05

### OD-01 — Document precedence

**The problem.** D4 names the contracts document (C1–C7) as the governing specification. But the contracts document predates register revision 2 and doesn't reflect it (its §8 sequencing assumes the legacy pipeline gets repaired first; it still says the monolith is 13,867 lines — it's 14,382 today (P01)). The brief (11 Sept) predates the register in several places (see §1.5). The supplement's UXC items say they should be added "only after review". And `POLISH_specification_v2.md` from 11 Sept is a fifth candidate — it is now in the repo root (P01), untracked.

**Why it matters for Cursor.** When two documents disagree, Cursor will pick one silently unless told how to choose.

**Recommendation.** Contracts C1–C7 are the governing *invariants*. Where any register rev 2 row revises or adds to older text, the register row wins. The brief governs interface intent where the register is silent. The supplement applies only for UXC items you adopt (OD-02). Anything not covered: Cursor stops and asks. Then fold the register changes into the contracts document properly (P03 produces the gap list).

### OD-02 — Adopt UXC1–UXC10?

Each UXC item cites the register rows it depends on, and none contradicts a DAD row. They're the same authority tier as DESIGN rows (proposed by a model, not independently arrived at), which is where both review rounds found most errors, so read them before adopting. If adopted, **keep the UXC handles** rather than renumbering them into D89+; a rename is one more numbering migration for Cursor to get wrong.

### OD-03 — Lineage model for resampling (D43 / F1, STRUCT)

**DECIDED 15 Sept: option (a), a new copy.** Text below kept for the record.

**The problem.** A sparse overlay keyed to sample index can't represent resampling, because resampling changes the index.

**Options.**
- **(a) Child dataset per resample.** The register's proposal; UXC8 assumes it. Source stays untouched on its own grid; the child has its own grid and a recorded mapping to the parent.
- **(b) Resample only at export time.** Keeps the overlay model pure, but the user can never inspect resampled values before writing them — breaks the load → inspect → decide → alter → inspect-again loop.
- **(c) Fractional-index overlay.** Possible, complex, and still has to answer "what is the original at a child depth".

**Recommendation: (a).** Consequence: identity keys become **(dataset_id, column_index)**, where a source dataset's id is the file SHA-256 (D30/F9, D46). Build the dataset-with-parent structure in P06 even though only source datasets exist until Stage 5 — cheap now, a migration later.

The register itself flags this as "only one of several possible answers. Worth challenging." That's still true.

### OD-04 — Canonical state schema

Five concrete problems, all verified in the documents:

1. **"Refused" means three different things.** A sample display state (brief §4.2), a disposition — "POLISH declined" (D79), and a method outcome (D86 scientific precondition failure). UXC1 warns against exactly this.
2. **Sample-status vocabularies disagree.** Brief §4.2: measured / created / changed / invalid / refused. D44: measured / created / changed / invalid / absent / resampled.
3. **`processing_eligibility` has no defined values anywhere.**
4. **D34 says a "low-tier" match doesn't unlock processing, but never says which tiers are low.** Is User-defined low? Proposed obviously is. Undefined.
5. **Brief §8's keyboard verbs include "refuse" as a user action**, while D79 defines Refused as something POLISH does. Who refuses?

**Why now.** D61 makes the status code table append-only and versioned. The first export that writes a code makes that code permanent. Renaming later is a code migration at best and a broken promise at worst.

**Recommendation.** Appendix A. My proposal resolves (1) by making "refused" a *gap fill outcome* and a *disposition*, never a sample status — a refused gap's samples are simply ABSENT, and the gap record carries the refusal.

### OD-05 — Is unit conversion a data alteration?

**DECIDED 15 Sept: option (a).** See 1.0 for the four conversion cases. Text below kept for the record.

**The contradiction (verified).**
- Brief §4.2 lists *unit conversion* as an example of **changed** samples.
- Brief §5 and D24 say **Stages 0–4 touch no data** — and Stage 3 is Units.
- UXC3's example copy says after a Stage 3 change, *"No data has been altered."*

All three can't hold.

**Options.**
- **(a)** Stage 3 decides the unit *interpretation* and target system. Conversion is an exact, invertible re-expression applied when values are read for display, computation or export, recorded in the manifest. Samples stay MEASURED.
- **(b)** Conversion writes CHANGED samples, which moves Stage 3 into the data-altering group and breaks D24.

**Recommendation: (a).** A unit conversion doesn't contradict the measurement; it restates it exactly. Marking it CHANGED would bury real alterations (smoothing, despiking) among trivial ones and dilute the one claim the brief calls "more serious". (a) is compatible with C1 — the unit becomes an attribute of the view, updated in the same operation. C4 (derived parameters invalidate) still applies in full. The brief's §4.2 example would need editing.

### OD-06 — Build strategy

**DECIDED 15 Sept: build beside, in `polish/`.** Text below kept for the record.

**Verified (P01, 15 Sept).** `advanced_preprocessing_system10.py` is **14,382 lines**; `core/` holds 12 modules (largest: `petrophysical_models.py` 1,358, `gap_filling.py` 1,226, `curve_identification.py` 1,221, `well_loaders.py` 1,105). *(An earlier version of this plan gave smaller figures; those were non-blank line counts.)* D27 says "extraction"; the brief says "rewritten".

**Recommendation.** Build a new top-level package, `polish/`, stage by stage, with the contracts built in from the first line. Harvest correct logic from `core/`, `petrophysics/` and the monolith into it behind tests. Leave the legacy app untouched and runnable until the new interface covers its read-only use, then retire it.

Two consequences worth stating:
- **This departs from contracts §8**, which says contract work waits until the legacy pipeline "produces correct output once". Under this strategy the golden expectations for Stages 0–4 come from facts in the raw files themselves (curve counts, rows, declared nulls, header depths), not from legacy output — which is the right source anyway, since legacy output is what was wrong.
- **Don't name anything in the new package `core`.** Two different things called `core` is a reliable way to make an AI coding agent edit the wrong one.

Why not refactor in place: the monolith's size is itself a source of agent error — earlier sessions had to locate code by symbol because line numbers drifted under every edit.

### OD-07 — Toolkit

**Verified today.** `requirements.txt` lists numpy, pandas, matplotlib, lasio, scipy, scikit-learn, pywavelets, psutil. No Qt. Tkinter is implicit.

**What's known.** Brief §12: layered plots with per-sample hit-testing, non-reflowing docked panels, keyboard traversal and multi-scale decimation are "awkward in Tkinter and natural in Qt". Contracts §8 already rules out a web UI (figures become static images). The 6 Aug session preferred PySide6 over PyQt6 on licensing grounds — PySide6 is available under LGPLv3. LGPL has specific obligations when bundling with PyInstaller; check the licence text before release — I'm not a lawyer.

**What's not known.** Whether matplotlib's Qt backend (reuses existing plotting code, known PyInstaller behaviour) or pyqtgraph (built for large interactive data, MIT licence) handles hover and redraw acceptably at 171 curves and 8 tracks on the 12 MB file. I'd be guessing. **P04 measures it.** Packaging size: Qt adds a lot; the "50–80 MB" figure from August was an estimate, not a measurement — P04 measures that too.

**Recommendation.** PySide6. Choose the plot layer from P04's numbers. P17 keeps decimation and hit-testing headless, so the toolkit choice doesn't leak into correctness logic.

### OD-08 — Numbering

**Verified.** The brief and the register both say "ten stages" and number them **0 through 10 — eleven stages**. The contracts document uses *Phase 0–4*; the supplement uses *Phase 0–3* with different meanings. That's the same overlapping-schemes problem you corrected on 11 Sept (D6).

**Recommendation.** Say "Stages 0–10" and stop saying "ten". Retire both Phase schemes in favour of named milestones (Part 2).

### OD-09 — The Cursor environment

**DECIDED 15 Sept: archive the legacy rules; keep Aikido active.** Guard: never pass well data or fixtures to it. Text below kept for the record.

**Verified today in the repo:**

| File | Lines | What it is |
|---|---|---|
| `.cursorrules` | 111 | Reads as an auto-generated summary of the **old** architecture ("Location: advanced_preprocessing_system10.py", "Importance: 90") |
| `.cursor/rules/` — 7 domain `.mdc` files | 72–100 each | Same auto-generated pattern ("# === USER INSTRUCTIONS ===") |
| `.cursor/rules/design.mdc` | 1,166 | Casts Cursor as *"superdesign, a senior frontend designer"* — a web-frontend persona that activates on UI prompts. Likely the source of the fabricated `.superdesign/design_iterations/` folder claim on 6 Aug |
| `.cursor/rules/aikido_rules.mdc` | 17 | `alwaysApply: true`. Tells Cursor to run `aikido_full_scan` on all generated/modified code and "always provide the full file content to the scanner" |
| `.giga/specifications.json` | — | Tracked; another agent-spec artifact |
| `.github/copilot-instructions.md` | — | Another agent instruction file |

**Why it matters.** Every prompt you send will be steered by rules describing the architecture you're replacing.

**The Aikido question.** I don't know whether that scanner runs locally or sends file contents to Aikido's service. It's your source code, not customer well data, so D14 isn't directly at stake — but you should know where your code goes. P01 asks Cursor to report what it can determine, and tells it not to run the scan.

**Recommendation.** Archive all of them (not delete) and replace them with one rules file (Part 4). Decide on Aikido after P01.

---

## 1.3 Domain-expert decisions — block specific stages

These go to your dad. None blocks scaffolding; each blocks a specific piece. While pending, the prompts implement the **slot** — the value comes from the rule library, and a missing value produces an honest "not eligible" or "not set" state, never a guessed default.

- **OD-10 (D12) — header/data depth tolerance.** **CLOSED.** The matching rule was answered; the rounding test is recorded under D12. Earlier text: **Answered 14 Sept ("verify whether the difference can be due to rounding; if so correct; otherwise skip") — meaning to confirm, see 1.0.2.** Original note: he needs the measured context in front of him: **83% of 144 audited wells** show sub-sample-interval STOP drift, so zero tolerance refuses five wells in six. Needed: a number, its unit (absolute depth, or a fraction of STEP), whether STRT and STOP differ, and what "grid-inconsistent" means for D8's non-overridable case.
- **OD-11 (D13) — KEOUGH.** **Answered: skip and notify.** Header `BLI1` says 5,326 ft; density data reads to 5,334 ft. Which governs? P07 excludes that interval from KEOUGH assertions until answered.
- **OD-12 (D11 / D87) — rule values.** **Values received 14 Sept; rule types to confirm, see 1.0.1.** F15 says "physical limit" conflates four rule types (domain constraint / hard impossibility / plausibility / consistency). **The worksheet asks for min/max and must be re-cut into those four before he answers**, or his answers will be filed under the wrong type. Until then, only definitional rules ship (a fraction can't exceed 1) — consistent with the standing "no guessed physical limits" instruction from 8 Sept.
- **OD-13 (D41, D49) — gap classification inputs.** Which curves belong to which tool; each tool's vertical resolution (D49 case 3, the dropout cutoff); the threshold for the ambiguous case (case 4); what counts as "shared" absence across a tool group (exact overlap, or some fraction). The register also asks whether there's a fifth case.
- **OD-14 (D42)** — interpolation domain per family. Resistivity is log-domain; which others?
- **OD-15 (D51)** — the preferred threshold table in `petrophysics/constants.py` is feet-only. Metric version?
- **OD-16 (D53)** — confirm the intent of `MIN_POINTS_BY_METHOD` (30 / 50 / 100 / 200). F8 says it can't apply to bounded interpolation. The register says to confirm with the table's author; I don't know who that is.
- **OD-17 (D7 / D32)** — TVD's role name, and how to handle F13 (linear TVD interpolation degrading across large increment changes in high-build sections).
- **OD-18 (D9 vs D85b)** — D9 says the DEPT copy "remains the default for depth curves". D85(b) says the auditable profile preserves original mnemonics *and curve order*. An appended copy doesn't literally break either, but neither says whether the auditable file carries it. D9 is his.
- **OD-28** — which derivation equations are in v1 (Stage 6), with citations.

---

## 1.4 Release-level decisions — don't block the read-only build

- **OD-19 (D16)** — release at the end of the read-only milestone, or build through Stage 10 first. The external reviewer's case for read-only-first: near-zero liability, validates the trust model before anything can fabricate a value, usable as an intake gate. The counter-case is yours: a read-only tool may not be the SeisTool onramp your dad needs.
- **OD-20 (D20)** — manual-workflow benchmark in v1 or after. Unscoped (F5). Feasible with zero custody using Volve / FORCE 2020.
- **OD-21** — **Dad (5.2): only when asked for. Conflicts with D61; see 1.0.3.** Per-sample method detail in the exported LAS. On the 171-curve file, one status curve per curve doubles the column count. Default on, or opt-in?
- **OD-22** — **Answered by dad (1.2, 3.2): users want an explicit "this well / all wells" choice.** Scope persistence. UXC4 offers "a reusable local rule" as a scope. The brief's anti-pattern list forbids "settings that persist invisibly between wells". A batch mapping saved to the local table (D37/D38) persists across wells *by design*. Proposal: allowed only as an explicit "create local rule" action, recorded at tier User-defined, visible wherever it fires.
- **OD-23 (D15)** — when licensing lands, and what an invalid or expired licence permits (nothing? read-only?). Needs F11's clock-rollback guard.
- **OD-24** — obtain the CWLS LAS 2.0 document. D65, D66, D68 are only PARTIAL (corroborated by other implementations); D67 is fully unverified.
- **OD-25** — visual design. Brief §10 questions still open: **Q1** coverage-track visual language (the brief calls it the most important screen), **Q2** curve selection at 171 curves, **Q3** two values at one depth, **Q8** density. If UXC items are adopted: Q4 is answered by UXC3, Q5 partly by supplement §4.1, Q6 by UXC7, Q7 by UXC4. Either run the design session the brief was written for, or I mock the coverage track here first. Either way, not Cursor.
- **OD-26** — the four real-well fixtures in `tests/fixtures/` are **untracked** (8.8 MB, 5.9 MB, 3.1 MB, 12.2 MB). `git lfs` 3.6.1 is installed. Check each file's licence permits keeping it in a private repo, and note GitHub applies LFS storage and bandwidth quotas. KEOUGH lives in `Downloads`, not the repo.
- **OD-27** — **Recommendation after dad's 4.5 ("no QC images"): a read-only export writes the describe report, its JSON, QC images, and the session file.** D17 says export writes two files. In a read-only release nothing is altered, so what's the "processed" file? Options: report + JSON manifest only; or a LAS carrying annotations and status curves with every value untouched.

---

## 1.5 Contradictions and errors found while preparing this

These aren't decisions — they're fixes. Items 1–3 were verified by parsing.

1. **"147 curves share the `LFP_` prefix" doesn't match the file on disk.** Parsing the `~C` section of `tests/fixtures/159-19A_LFP.las` today: **171 curves, 170 begin with `LFP_`**; the only exception is `DEPTH`. The brief, the register (D82) and the supplement ("147 affected … 24 without the prefix") all use 147/24. Possibly 147 was a different count (curves that re-match after stripping?) — I don't know its origin. **Tests must compute the group, never hardcode 147.**
2. **The register's own source tally is off.** It states DAD 7 / AC 18 and says it was "parsed, not tallied by hand". Parsing the rows gives **DAD 6** (D7, D8, D9, D11, D12, D13) and **AC 19**.
3. **D21 cites "sample status (D38)".** D38 is the local-alias-tables row. Sample status is **D44** — the same kind of mis-citation revision 2 corrected elsewhere.
4. Register "For the reviewer" point 4 says "three questions belong to the domain expert" and then lists four — and one of them (D9) is marked RESOLVED.
5. D62 refers to a "forensic" profile. D85 names three profiles and none is called forensic.
6. **Brief §3 still carries the competitor absolute** that D77 / F18 said must be edited ("None of them can tell you…").
7. **Brief §4.1 still describes a pure overlay** with no mention of the D43 structural conflict, and **§4.8 has no block types** (D86). Someone reading the brief alone would build a gate with a force-export path.
8. "Ten stages" numbered 0–10 (OD-08).
9. The contracts document is headed "Status: Draft for review. Not yet implemented" and predates register rev 2 (P03 lists the gaps).
10. Unit conversion as "changed" vs "Stages 0–4 touch no data" (OD-05). **Now decided — brief §4.2 must drop unit conversion from its list of "changed" examples.**
11. Brief §7.1's track order is overridden by your father's answer 4.2 (see 1.0.3). The brief needs updating.
12. `159-19A_LFP.las` is `WRAP=YES` (about 30 text lines per depth step). Any "row count" must count depth steps, not lines. P01 and P07 are updated.

---

# PART 2 — THE PLAN AHEAD

## 2.1 Where things stand (verified today unless marked)

- Branch `freeze-fix-wip`, HEAD `f815811` — matches the register. The last six commits finished the depth work: DEPT copy (`7f638ac`), DEPT required before resampling (`a2be560`), resampling's own default-off switch (`962904a`), halt on unidentifiable depth (`dc0bd5d`), real depth unit in logs and display (`f2ccf61`, `f815811`).
- Working tree: four untracked LAS fixtures; nothing else uncommitted.
- Code: monolith 14,382 lines; `core/` 12 modules; `ui/` 12 modules (Tkinter); `petrophysics/` 3 modules; `tests/` 10 test files.
- Tooling: `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, ruff at a starting gate (E9, F821, F822), pytest collected from the repo root.
- Fixtures (parsed today): `159-19A_LFP.las` 171 curves; `30_3-5 S.las` 16 curves (likely the brief's "22% fill" file — fill not measured); `34_10-19.las` 18; `WLC_PETRO_COMPUTED_INPUT_1_15_9-F-1B.las` 23 curves, 12.2 MB (likely the "12 MB" file); `new_file_null_minus_999.las` 2 curves (tracked).
- **Established by P01:** 56 tests pass, ruff is clean, no golden files exist, no network code exists. CI's live status on the last push was not checked.

## 2.2 Target architecture

```
polish/                    new engine — no UI toolkit, no network, ever
  model/
    ids.py                 FileHash, DatasetId, CurveRef(dataset_id, column_index), SampleRef
    states.py              every enum from OD-04 (Appendix A)
    dataset.py             SourceFile, Dataset (read-only arrays, parent, grid), Curve
    overlay.py             sparse overlays keyed to SampleRef
    findings.py            Finding, BlockType, evidence, Disposition, DecisionRecord
  stages/
    contract.py            Requires / Guarantees / Declines (contracts §6) as data
    runner.py              executes stages, checks contracts, records results
    s00_ingest.py … s10_export.py
  rules/                   D87 rule library loader; rule files are versioned DATA
  tables/                  mnemonic, prefix, qualifier, unit-spelling tables (DATA)
  decisions/               sessions, dispositions, scope, batch
  deps/                    dependency graph, staleness, impact summaries
  provenance/              regeneration records, manifest, JSON Schema
  render/                  headless decimation, interval summaries, hit-testing
  cli.py                   `python -m polish describe FILE --out DIR`
polish_ui/                 toolkit layer (OD-07) — imports polish, never the reverse
tests/
  arch/                    no network, no UI imports in polish/, immutability, input never written
  golden/                  raw-file facts per fixture (reviewed JSON)
  stages/                  per-stage tests
  scenarios/               supplement Part F, scenarios 1–6
legacy (untouched until retirement): advanced_preprocessing_system10.py, core/, ui/, petrophysics/
```

Five properties enforced by tests, not by convention:

1. `polish/` imports no UI toolkit and no network module.
2. Source arrays are read-only; the input file is opened read-only; its hash is identical before and after any run.
3. Every stage declares its contract as data, and the runner checks it at the boundary.
4. Rendering truth — the D74 decimation rule — lives in `polish/render/` and is tested without a GUI.
5. Rules and tables are versioned data files, never constants buried in code (D87).

## 2.3 Milestones and sequence

| Milestone | Prompts | What exists at the end | Maps to |
|---|---|---|---|
| **Groundwork** | P01–P04 | Facts about the repo; governing docs in repo; one rules file; contracts gap list; toolkit numbers | — |
| **Truthful vocabulary and skeleton** | P05–P07 | State schema in code; package skeleton with architecture guards; golden raw-file facts | Supplement Part E "Phase 0"; D27 golden files + contracts |
| **Read-only engine** | P08–P16 | Stages 0–4 headless; decisions, sessions, staleness, batch in the engine; describe report + CLI | D27 extraction; supplement "Phase 1–2" engine half |
| **Read-only trust surface** | P17–P22 | Interface over Stages 0–4; queue; refusals; stale surfaces; export review | D27 UI. **End = the D16 release candidate (OD-19)** |
| **Data-altering** | P23–P27 | Stages 5–8 with lineage, consent, refusal, provenance; altered-data rendering | Supplement "Phase 3" |
| **Export and release** | P28–P31 | Manifest, status curves, three export profiles, round-trip, licensing, packaging, scenario suite | Stages 9–10 |

Order inside the read-only engine is deliberate: decisions, staleness and batch (P13–P15) sit **in the engine before any interface lets a user change an upstream decision**, because UXC3 says staleness must exist before any editable upstream decision is exposed.

## 2.4 What gates what

| Prompt | Needs these decided first |
|---|---|
| P02 | Licence check on fixtures |
| P04 | nothing (run it if OD-07 is open) |
| P05 | Every line of the Ready-for-P05 checklist (1.0.6) |
| P06 | OD-03 |
| P07 | OD-11 (interval exclusion only), OD-26 |
| P09 | OD-10 closed (rounding test under D12), dad's 2.3 meaning |
| P10 | OD-17 (role name) |
| P12 | OD-12 rule types (1.0.1), OD-13 (definitional-only until then) |
| P13, P15 | OD-22 (answered) |
| P17 onward | OD-07 |
| P19, P20 | OD-25 |
| P22 | OD-27 |
| P23 | OD-03, OD-13, OD-14, OD-15, OD-17, OD-19 |
| P24 | OD-16, OD-28 |
| P25 | OD-13, OD-14, OD-15, OD-16 |
| P27 | OD-25 (Q3) |
| P28, P29 | OD-18, OD-21, OD-24 |
| P30 | OD-23 |

## 2.5 AI comprehension as a design constraint

Two senses, both relevant here.

**The builder is an AI agent.** Small modules with one job each, one glossary, enums instead of free strings, contracts as data, governing documents inside the repo, one rules file, no name collisions (`polish/` vs legacy `core/`). Each of these removes a specific way Cursor has gone wrong on this project before: drifting line numbers in the monolith, stale rule files, a fabricated folder.

**The output will be read by AI systems.** The manifest ships with a published JSON Schema (P28), and status codes are versioned and append-only (D61). An exported POLISH file then carries machine-checkable provenance that a downstream agent — including your own validation tooling — can verify without trusting prose. That's the product claim made legible to machines, not just to petrophysicists.

---

# PART 3 — HOW TO USE THE PROMPTS

1. **Answer the OD items each prompt needs** (table 2.4). Fill every ⟦slot⟧. If you can't fill one, don't send that prompt.
2. **Run P01 first, as-is.** It's read-only and runs under the *old* rules, which is intended: it inventories them.
3. **After P02, the rules file in Part 4 is active.** Prompts from P03 onward assume it.
4. **One prompt per fresh Cursor Agent chat.** Paste the whole block. Don't combine prompts.
5. **Bring every report back here before sending the next prompt.** I'll check its claims against the repo read-only. Cursor fabricated a folder name once on this project (6 Aug); the rules require evidence for every claim, but checking is still the step that catches it.
6. **P23–P31 are drafts.** They're thorough, but they'll need revision after the read-only milestone teaches us things. Re-check each with me before sending.
7. Cursor must never fill a slot on its own, and never "temporarily" pick a value for an OD item.

---
# PART 4 — THE STANDING RULES FILE

P02 installs this as `.cursor/rules/polish-rebuild.mdc`. It replaces every legacy rule file except `aikido_rules.mdc`, which stays active (OD-09). The precedence and UXC lines are filled with the defaults recorded 15 Sept.

````markdown
---
description: POLISH rebuild — standing rules for every task in this repository
alwaysApply: true
---

# POLISH rebuild — standing rules

## What POLISH is
A fully isolated desktop preprocessor for petrophysical well logs (LAS). It describes a
file completely before altering anything, alters only with recorded consent, and reports
refusals as prominently as actions. Honesty is the product. A value that looks measured
but was invented is the worst possible defect.

## Governing documents (in docs/governing/)
- POLISH_pipeline_contracts.md — invariants C1–C7
- POLISH_decision_register.md — decisions D1–D88, findings F1–F19
- POLISH_design_brief.md — interface intent
- POLISH_UIUX_cohesion_supplement.md — UXC items (adopted: UXC1–UXC10, all ten)
- POLISH_build_plan.md — open decisions OD-nn, prompts Pnn
Precedence when they disagree: Contracts C1–C7 are the core invariants. Where a
decision-register row (revision 2 or later) revises older text, the register row wins.
The domain expert's answers (POLISH_build_plan.md section 1.0) outrank DESIGN-tier
rows on petrophysics and workflow. Owner decisions recorded in POLISH_build_plan.md
section 1.0 apply. The design brief governs interface intent where the register is
silent.
If a conflict is not covered by that rule: STOP and report it. Do not choose.

## Numbering
Use only: Stage 0–10 (eleven stages), D-numbers, F-numbers, UXC-numbers, C1–C7,
OD-numbers, P-numbers. Do not introduce steps, phases, tasks or any other numbered scheme
in code, comments, docs or commit messages.

## Hard rules — never break these
1. No network access in shipped code. polish/ and polish_ui/ must not import socket,
   requests, urllib.request, http.client, httpx, aiohttp, ftplib, smtplib, webbrowser,
   or any Qt network module. No telemetry, no update checks, no URLs fetched.
2. polish/ never imports a UI toolkit. polish_ui/ imports polish/, never the reverse.
3. The input file is never opened for writing. Source arrays are read-only.
   Nothing overwrites original values; changes live in overlays or child datasets.
4. No guessing. If a value, unit, family, tolerance, bound, threshold or rule is not
   established by a governing document, a rule/table data file, or a recorded user
   decision, the result is an explicit "not set / not eligible / unknown" state.
   Never a default. Never a fallback range. Never `.get(key, some_default)` for
   scientific values.
5. No silent conversion of absent to present. Every created or changed sample has a
   provenance record.
6. No "force export", "accept anyway", "fill anyway" or equivalent escape hatch,
   anywhere, under any name.
7. Do not modify legacy files (advanced_preprocessing_system10.py, core/, ui/,
   petrophysics/) unless the prompt explicitly says so. Reading and harvesting logic
   from them is expected. polish/ never imports them (three core/ modules import
   tkinter): copy the logic across with a comment naming its source file and symbol.
8. Never fill an ⟦OD-…⟧ slot or decide an open decision yourself. Stop and report.
9. Proper fixes only. No workarounds, suppressed warnings, skipped tests, broad
   try/except, or TODO placeholders standing in for behaviour. Never write
   "# ... rest of code ...".
10. No new dependency without listing it in the report with its licence and why.

## Working method
- Locate code by symbol name, never by line number.
- Read before writing. Before editing a module, read it and the tests that cover it.
- Every public function in polish/ has a docstring naming the D/C/UXC rows it
  implements.
- Scientific values (bounds, tolerances, thresholds, tool groups, unit factors,
  mnemonic entries) live in versioned data files under polish/rules/ or polish/tables/,
  each entry carrying its source and source tier.
- Tests accompany every change. Parametrise across the fixtures in tests/fixtures/.
  Never hardcode a count that can be computed from a fixture.
- Before committing: run `python -m pytest -q` and `ruff check .`. Both must pass.
- Commit in small, single-purpose commits. Commit messages are plain sentences
  describing the change (house style: "Halt processing when depth cannot be identified,
  and show the reason."). Reference D/C/UXC rows in the body, not the subject.
- Aikido: .cursor/rules/aikido_rules.mdc stays active by owner decision (OD-09,
  15 Sept); follow it. Never pass .las files, anything under tests/fixtures/, or any
  well data to Aikido or any other external tool — source code only.

## Evidence rules for your report
- Never claim a file, folder, function, test or result exists unless you listed,
  read or ran it in this task. Paste the command and its output (tail is fine).
- Distinguish: VERIFIED (you ran/read it), INFERRED (reasoned from code),
  NOT CHECKED.
- If something fails, report the failure. Do not paper over it.

## Report format (end of every task)
1. Summary — two to four sentences.
2. Files changed — full relative paths, one line each on what changed.
3. Commands run — with output tails, including pytest and ruff counts before and after.
4. Decisions I needed but did not make — each with the OD/D number if known.
5. Deviations from the prompt — and why.
6. Not verified — anything you could not check.
7. Suggested next step — one line.
````

---

# PART 5 — CURSOR PROMPTS

Each prompt is a single block. Send one per fresh Agent chat.

---

## Groundwork

### P01 — Workspace reconnaissance (read-only)

**Needs:** nothing. **Send as-is.**

````text
POLISH rebuild — P01: Workspace reconnaissance. READ-ONLY.

This task changes nothing. Do not edit, create, move, rename, delete, stage or commit
any file. Do not install packages. Do not run aikido_full_scan or any scanner or tool
that sends file contents off this machine — this prompt explicitly says not to.

Goal: an evidence-based inventory of this repository as it is right now, so a rebuild
plan can start from facts instead of memory.

For every item, paste the command you ran and its output (trimmed tails are fine).
Label each finding VERIFIED (ran/read it), INFERRED, or NOT CHECKED.

1. Git. Current branch; HEAD hash and subject; `git status --short`; local and remote
   branches; `git log --oneline -15`; whether any branch protection or ruleset file is
   referenced in the repo.

2. Tests. Run `python -m pytest -q`. Report passed/failed/skipped/errored counts and
   the names of any failures. Do not fix anything.

3. Lint. Run `ruff check .`. Report counts only.

4. CI. Summarise .github/workflows/ci.yml: triggers, Python version(s), each step.
   Summarise .pre-commit-config.yaml hooks.

5. Environment. `python --version`. Installed version of every package in
   requirements.txt. Whether PySide6, pyqtgraph, pytest-qt, pytest-socket,
   cryptography and pyinstaller are installed (version or "not installed").

6. Code map. For every .py file at the repo root and under core/, ui/, petrophysics/,
   tests/: line count; top-level class and function names; which project modules it
   imports. Flag any module under core/ or petrophysics/ that imports tkinter or
   matplotlib.pyplot.

7. Network surface. Search every .py file for imports of socket, requests,
   urllib, http.client, httpx, aiohttp, ftplib, smtplib, webbrowser, and for the
   strings "http://" and "https://". List every hit with path and enclosing symbol.

8. Agent instruction files. For .cursorrules, every file under .cursor/rules/,
   everything under .giga/, and .github/copilot-instructions.md report: path, line
   count, frontmatter (alwaysApply / globs / description), a two-sentence summary, and
   whether it describes advanced_preprocessing_system10.py as the architecture.
   For .cursor/rules/aikido_rules.mdc specifically: which MCP server or tool it
   refers to; whether that server is configured and enabled in this Cursor install;
   and — only from the tool's own description or configuration — whether a scan sends
   file contents off this machine. If you cannot determine that, write
   "cannot determine". Do not guess.

9. Existing test coverage of read-only behaviour. For each test function that touches
   LAS loading, depth identification, curve identification, units, null handling,
   range validation or gap classification: file, function name, one line on what it
   asserts.

10. Fixtures. Every .las file anywhere in the repo (tracked or untracked): path,
    size, tracked or not. For each, by reading the header TEXT directly (do not use
    the application's loaders or lasio): LAS VERS, WRAP, declared NULL, STRT / STOP /
    STEP with units, number of curve lines in ~C, number of those whose mnemonic
    begins with the same prefix (report the most common prefix and its count), and
    the number of DEPTH STEPS in ~A. Careful: in WRAP=YES files one depth step spans
    several text lines (tests/fixtures/159-19A_LFP.las is wrapped) — report both the
    number of text lines and the number of depth steps, and say how you counted.

11. Mnemonic database. Where it lives, file format, number of entries, number of
    families, and what fields each entry has.

12. Depth handling as it exists. Locate (by symbol) the code behind these commits and
    summarise what each does: 7f638ac, a2be560, 962904a, dc0bd5d, f2ccf61.

13. Documents. Every .md, .txt, .ods, .xlsx, .json at the root, under docs/ (if it
    exists) and under distribution/: one line each.

End with a section "Things I could not determine".

Report format:
1 Summary. 2 Findings by item number. 3 Commands run. 4 Things I could not determine.
````

### P02 — Install the governing documents and replace the agent rules

**Needs:** P01 reviewed with me (done); your licence check on the four fixtures. OD-09 decided: Aikido stays. OD-01, OD-08 and OD-26 use the defaults recorded 15 Sept.
**You do first, by hand:**
1. Back up the four untracked fixtures and the governing documents to a folder outside the repo.
2. Create `docs/governing/` and move into it: `POLISH_design_brief.md`, `POLISH_decision_register.md` and `POLISH_specification_v2.md` (currently untracked in the repo root), `POLISH_UIUX_cohesion_supplement.md` (renamed from the `.txt`), and **the latest** `POLISH_build_plan.md` (this document, renamed — the copy in the repo root is an older version; delete it).

````text
POLISH rebuild — P02: Install governing documents and replace the agent rules.

Preconditions — stop and report if any is false:
- docs/governing/ contains POLISH_design_brief.md, POLISH_decision_register.md,
  POLISH_specification_v2.md, POLISH_UIUX_cohesion_supplement.md and
  POLISH_build_plan.md.
- The repo root no longer contains copies of those files.
- `git status --short` shows no modified tracked files (untracked files are expected).

Tasks, one commit each, in this order:

1. Branch. Create branch `rebuild` from the current HEAD of freeze-fix-wip and work
   there.

2. Move the contracts document into docs/governing/ with `git mv` (history preserved).
   Search the whole repo (code, docs, CI, pyproject, HOW_TO_RUN.md, TESTING_GUIDE.md)
   for references to POLISH_pipeline_contracts.md and update every one. List them.

3. Normalise the supplement to UTF-8 with LF line endings. Change no text. Then commit
   every file in docs/governing/ (they are untracked) in one commit.

4. Create docs/governing/README.md containing exactly:
   - one paragraph per governing document saying what it governs;
   - the precedence rule: Contracts C1–C7 are the core invariants. Where a
     decision-register row (revision 2 or later) revises older text, the register row
     wins. The domain expert's answers (POLISH_build_plan.md section 1.0) outrank
     DESIGN-tier rows on petrophysics and workflow. Owner decisions recorded in
     POLISH_build_plan.md section 1.0 apply. The design brief governs interface intent
     where the register is silent. Any conflict not covered: stop and ask.
   - the numbering rule: say "Stages 0–10" (eleven stages); no Phase numbers anywhere;
     only the handles listed in the rules file.
   - the adopted UXC items: all ten, UXC1–UXC10.
   Also create docs/governing/ready_for_P05.md containing exactly the checklist in
   POLISH_build_plan.md section 1.0.6.

5. Archive the legacy agent instructions. Using `git mv`, move into
   docs/archive/legacy_agent_rules/ (preserving each file's name and content exactly):
   .cursorrules, every file in .cursor/rules/ EXCEPT aikido_rules.mdc, .giga/, and
   .github/copilot-instructions.md. aikido_rules.mdc stays where it is and stays
   active (owner decision OD-09, 15 Sept).
   Add docs/archive/legacy_agent_rules/README.md: one line saying these described the
   pre-rebuild architecture and are kept for reference only.

6. Create .cursor/rules/polish-rebuild.mdc with EXACTLY the content I paste below,
   byte for byte. Do not edit it.
   ⟦PASTE THE PART 4 RULES FILE HERE (or the Notion page "Cursor rules file")⟧

7. Fixtures: Track tests/fixtures/*.las with Git LFS: run `git lfs install` then
   `git lfs track "tests/fixtures/*.las"`. .gitattributes already exists — append, do
   not overwrite. Commit .gitattributes, then add the four untracked fixtures in a
   separate commit.

8. Tell me: which rule files Cursor now reports as active for this workspace. The
   expected answer is polish-rebuild.mdc and aikido_rules.mdc, nothing else. If
   anything else is still active, say which and why.

Acceptance:
- `git status --short` clean.
- `python -m pytest -q` gives the same counts as P01 reported.
- `ruff check .` passes.
- `git lfs ls-files` lists the four fixtures.

Do not push. Report in the standard format.
````

### P03 — Contracts vs register reconciliation audit

**Needs:** P02 done. **Changes no code;** writes one document.

````text
POLISH rebuild — P03: Reconcile the contracts document against the decision register.

No code changes. Do not edit any file in docs/governing/. Write one new file:
docs/governing/reconciliation_P03.md.

Read in full: docs/governing/POLISH_pipeline_contracts.md,
docs/governing/POLISH_decision_register.md, docs/governing/POLISH_design_brief.md,
docs/governing/POLISH_UIUX_cohesion_supplement.md, docs/governing/README.md.

Produce four tables.

Table 1 — Register → contracts. For every register row whose status is REVISE, NEW,
RESOLVED, STRUCT, OPEN, PARTIAL or UNVER, and for every adopted UXC item:
  row id | which contract (C1–C7) or contracts section it touches ("none" allowed) |
  does the contracts document reflect it: Yes / No / Partly | the relevant contracts
  sentence, quoted, one line max | proposed contract text (a PROPOSAL — not applied).

Table 2 — Contracts → register. Every normative statement in the contracts document
("must", "forbidden", "never", "always", assertions in §7, sequencing in §8) that no
register row covers or that a register row contradicts. Quote it; name the row.

Table 3 — Brief vs register. Every place the design brief's text conflicts with a
register rev 2 row. Include at least: brief §3 (vs D77/F18), §4.1 (vs D43), §4.2
(vs D44 and D24), §4.8 (vs D86), §5 "ten stages" (vs the 0–10 numbering). Add any
others you find.

Table 4 — Cross-reference errors. Every place any governing document cites a D, F,
C or UXC number whose content doesn't match the citation. Check at least: D21's
reference to "sample status (D38)".

Then a short section: "Contract changes that would be needed before Stage 0 code is
written" — ordered, each one line, each citing its row.

Do not resolve anything. Where two readings are possible, list both.
Report in the standard format.
````

**After P03:** bring the file back here. The actual contract edits should be written and reviewed with me, not by Cursor, then committed.

### P04 — Toolkit spike (throwaway)

**Needs:** nothing, but only worth running while OD-07 is open. **Never merged.**

````text
POLISH rebuild — P04: Toolkit spike. THROWAWAY. Never merge this branch.

Create branch spike/toolkit from the current rebuild branch. Put all code under
spikes/toolkit/. You may install PySide6, pyqtgraph and pyinstaller into the current
environment for this spike; list versions and licences in the report. Do not add them
to requirements.txt.

Goal: measure, don't recommend. I will choose the toolkit from your numbers.

Build three minimal prototypes with identical behaviour:
  A. PySide6 + matplotlib (QtAgg backend)
  B. PySide6 + pyqtgraph
  C. Tkinter + matplotlib (TkAgg) — baseline

Each prototype:
1. Loads a LAS file given on the command line using lasio, read-only. No other
   processing.
2. Left: a coverage strip — one thin column per curve, all curves, full well. A pixel
   is dark where the curve has a value and light where it is null. Decimation rule:
   if ANY sample in a pixel is null, the pixel shows null. Never average.
3. Centre: 8 log tracks sharing one vertical depth axis (first 8 non-depth curves).
   Depth increases downward.
4. Hover: moving the mouse over any track shows depth, curve mnemonic and the value of
   the nearest actual sample (not an interpolated value) in a fixed readout label.
5. Right: a side panel with a toggle button. Showing or hiding it must NOT change the
   size or position of the tracks (reserve its space).
6. Keys: Z = full-well view; G = zoom to a 10 ft window at the current hover depth.

Measure on tests/fixtures/159-19A_LFP.las (171 curves) and
tests/fixtures/WLC_PETRO_COMPUTED_INPUT_1_15_9-F-1B.las (12 MB):
- cold start to first complete render (s)
- hover latency: median and p95 over 200 synthetic mouse-move events across the tracks
  (explain exactly how you generated and timed them, and what that method cannot
  capture)
- full-well → 10 ft zoom redraw time, and back
- peak resident memory (psutil)
- whether the side-panel toggle changes track geometry (assert widget geometry before
  and after)
- PyInstaller one-folder build: success or failure, folder size in MB

Report a single comparison table, then for each prototype: what was awkward to build,
anything that required a workaround, and anything you could not measure. No
recommendation.
````

---

## Truthful vocabulary and skeleton

### P05 — State schema in code

**Needs:** OD-02, OD-04 (Appendix A, as you approve or amend it), OD-05, OD-06.

````text
POLISH rebuild — P05: Canonical state schema.

Implements: D21 (as extended), D44, D79, D86, D33, D34, D32/D40, UXC1, UXC2, UXC9.

Create the new package polish/ (with polish/__init__.py exposing __version__) and
polish/model/ containing states.py only. Nothing else in this prompt.

1. polish/model/states.py — enums and small frozen dataclasses exactly as specified in
   the schema below. Use enum.Enum with explicit string values (stable for
   serialisation). Every enum has a docstring citing its rows. No other logic except
   the two pure functions in step 2.

   ⟦OD-04: PASTE APPENDIX A TABLES A.1–A.3 HERE (approved 15 Sept)⟧
   OD-05 (decided): unit conversion is a recorded re-expression; it never produces
   CHANGED samples. Stored values never change; the unit travels with the value.
   USER_DEFINED evidence is eligible for processing, and every USER_DEFINED curve
   carries the label "User-mapped — not identified by POLISH" (add it as a constant
   in states.py and use it in the glossary).

2. Two pure functions:
   a) permitted_actions(block_type) -> frozenset of UserAction — from the approved
      block-type/action matrix.
   b) export_permission(findings, dependency_states) -> ExportPermission (a frozen
      dataclass: allowed: bool, reasons: tuple of human-readable strings each naming
      the finding id and why). Rules: from the approved matrix. There is no parameter,
      flag or code path that overrides a structural or conformance block.

3. docs/governing/state_glossary.md — one table per enum: value, meaning in one
   sentence, which rows define it, and the preferred user-facing wording (UXC2 and
   UXC9 language tables). Plus the block-type/action matrix as a table.

4. tests/model/test_states.py:
   - every enum value round-trips through its string value;
   - enum string values are unique within each enum;
   - permitted_actions returns no disposition-clearing action for
     STRUCTURAL_DATA_FAILURE, SCIENTIFIC_PRECONDITION_FAILURE,
     EXPORT_CONFORMANCE_FAILURE;
   - export_permission is False whenever any structural or conformance finding exists,
     whatever else is true;
   - export_permission is False when any finding of a dispositionable type has no
     terminal disposition;
   - export_permission is False when any dependency affecting export is STALE;
   - export_permission is True for: one ACCEPTED informational finding, one RESOLVED
     ambiguity, one DEFERRED ambiguity with a reason, nothing stale;
   - a DEFERRED disposition without a non-empty reason cannot be constructed.

5. tests/model/test_states_snapshot.py: writes nothing; compares the set of
   (enum name, value) pairs to a committed snapshot file
   tests/model/states_snapshot.json. Adding values is allowed only by updating the
   snapshot in the same commit; removing or renaming an existing value must fail the
   test with a message citing D61.

Acceptance: pytest and ruff pass. No imports outside the standard library in
polish/model/states.py.
Report in the standard format.
````

### P06 — Package skeleton, identity, dataset model, stage contract, architecture guards

**Needs:** P05, OD-03.

````text
POLISH rebuild — P06: Skeleton, identity, dataset model, stage contract, guards.

Implements: D30 (with F9), D43 (resolved as a child dataset per resample — OD-03,
decided 15 Sept), D46, D17 (read-only input), C5, contracts §6, D14.

Create:

1. polish/model/ids.py
   - FileHash: SHA-256 hex of the input file bytes, computed in streaming chunks.
   - DatasetId: for a source dataset, derived from FileHash; for a child dataset, a new
     id plus the parent DatasetId.
   - CurveRef(dataset_id, column_index). Mnemonics are NOT identity (duplicates are
     legal in LAS).
   - SampleRef(curve_ref, sample_index).
   All frozen, hashable, with stable string serialisation.

2. polish/model/dataset.py
   - SourceFile(path, file_hash, size_bytes, read_at).
   - Curve(ref, original_mnemonic, original_unit, original_description, values):
     values is a numpy array with flags.writeable = False. original_* are verbatim
     from the file and never modified.
   - DepthGrid(values read-only, direction: ASCENDING | DESCENDING, regularity:
     REGULAR | IRREGULAR, declared_step, unit).
   - Dataset(id, parent_id or None, source_file, grid, curves tuple, lineage record
     tuple). Construction of a Dataset never copies-and-mutates a parent's arrays.
   Per OD-03 (child dataset): a child dataset has its own grid and a recorded mapping
   to its parent. Implement only the structure and its invariants here — no
   resampling.

3. polish/model/overlay.py — a sparse overlay store keyed by SampleRef holding
   (new_value, sample_status, provenance_record_id). Reading a curve "with overlay"
   returns a new array and never mutates the source. Reading "original only" returns
   the source values untouched.

4. polish/model/findings.py — Finding(id, stage, block_type, subject refs, evidence
   (structured dict with a declared schema per finding kind), message, created_at) and
   DecisionRecord (fields per the state glossary). Findings are immutable; dispositions
   are separate records linked by finding id.

5. polish/stages/contract.py — the contracts §6 template as data:
   StageContract(stage_number, name, requires, guarantees, declines_when,
   depends_on_stages, alters_data: bool). Stages 0–4 must declare alters_data=False
   (D24). polish/stages/runner.py — runs registered stages in order, checks each
   stage's declared requirements before running it, records a StageResult
   (status: NOT_RUN | COMPLETE | REFUSED_STRUCTURAL | STALE, findings, duration),
   and fails loudly at the boundary if a stage violates its own contract.

6. tests/arch/:
   - test_no_network.py: (a) static — walk every .py under polish/ and polish_ui/ (if
     it exists) with ast and fail on imports of socket, requests, urllib.request,
     http.client, httpx, aiohttp, ftplib, smtplib, webbrowser, PySide6.QtNetwork;
     (b) runtime — a conftest fixture, applied automatically to every test under
     tests/, that replaces socket.socket and socket.create_connection with functions
     that raise; implement it in tests/conftest.py without adding a dependency.
   - test_layering.py: no module under polish/ imports tkinter, PySide6, matplotlib,
     pyqtgraph, polish_ui, or any legacy module (advanced_preprocessing_system10, core,
     ui, petrophysics). Also, in a fresh subprocess, `import polish` plus a full runner
     pass leaves none of tkinter, matplotlib, PySide6 or pyqtgraph in sys.modules
     (catches transitive imports — P01 found three core/ modules that import tkinter).
   - test_immutability.py: writing to any Curve.values raises; the overlay never
     mutates source arrays; reading the input fixture's hash before and after a full
     runner pass gives identical results; the input file is only ever opened in a
     read mode (patch builtins.open / pathlib to assert this during a run).
   - test_contract_runner.py: a stage declaring alters_data=False that modifies data
     fails; a stage whose requirement isn't met doesn't run and the runner reports why.

Do not implement any stage logic yet. Do not touch legacy code.
Acceptance: pytest and ruff pass.
Report in the standard format.
````

### P07 — Golden raw-file facts

**Needs:** P06, OD-26, OD-11 (for the excluded interval).

````text
POLISH rebuild — P07: Golden raw-file facts.

Implements: D5 (KEOUGH as a toleranced regression fixture), D27 (golden files first).

Principle: golden expectations for Stages 0–4 come from the raw files, NOT from the
legacy pipeline's output (which is known to be wrong in places).

1. tests/golden/extract_facts.py — a small, independent header/data reader written
   for this purpose only. It must not import polish/, core/, lasio or the monolith.
   For a given LAS file it records: file SHA-256; LAS VERS; WRAP; declared NULL (as
   text and as number); STRT/STOP/STEP with units as written; every ~C line (index,
   mnemonic, unit, description) verbatim; number of depth steps (WRAP=YES files spread
   one depth step over several lines — handle both wrapped and unwrapped files, and
   test both); per curve: count of values
   equal to the declared NULL (with a tolerance of 1e-6 relative, documented), first
   and last depth with a non-null value, min and max of non-null values.

2. Run it over every fixture in tests/fixtures/ and ⟦KEOUGH path if available,
   e.g. C:\Users\achav\Downloads\2022\1052970244.las — and whether it may be copied
   into tests/fixtures/ per OD-26⟧. Write tests/golden/<fixture-stem>.facts.json.

3. DO NOT treat these files as correct yet. In the report, paste for each fixture the
   headline facts (curve count, depth-step count, NULL, STRT/STOP/STEP, most common mnemonic
   prefix and its count) so I can check them against an independent reading.

4. KEOUGH: exclude the interval 5,326–5,334 ft from any assertion. Record the exclusion
   in the facts file with the reason "D13: header and data disagree; domain expert
   decision 14 Sept is skip and notify".

5. tests/golden/test_facts_stable.py — re-running the extractor on each fixture
   reproduces its committed facts file exactly (guards against fixture corruption).

6. tests/golden/README.md — what these files are, how they were produced, that a human
   reviewed them (leave the reviewer line blank for me to fill), and that tests must
   read counts from these files rather than hardcoding them.

Report must include: for 159-19A_LFP.las, the number of curves and the number whose
mnemonic begins with "LFP_". The planning documents say 147; if your number differs,
say so plainly.
Report in the standard format.
````

---

## Read-only engine (Stages 0–4)

### P08 — Stage 0: Ingest and audit ledger

**Needs:** P07. Review the proposed ledger-to-block-type mapping in the prompt before sending.

````text
POLISH rebuild — P08: Stage 0 Ingest and audit ledger.

Implements: D64, D65 (PARTIAL), D66 (PARTIAL, F19), D68 (PARTIAL), D30, D46, D17,
C3, C6, contracts §6. Evidence level matters: D65/D66/D68 are corroborated by other
implementations, not verified against the CWLS specification (OD-24).

Create polish/stages/s00_ingest.py (alters_data=False, depends_on_stages=()).

Pass 1 — raw text audit (no parser library):
- Detect encoding; never assume. Record what was detected and how.
- Line endings; section markers (only the first character after ~ is significant —
  PARTIAL); duplicate or missing sections; position of ~A (last section);
  VERS and WRAP.
- Header lines split on first period and last colon (PARTIAL); record every line that
  doesn't fit.
- ~C count vs data column count, accounting for WRAP=YES, comment lines and blank lines.
  A mismatch is not automatically a parse failure — record exactly what was found.
- Declared NULL: match with tolerance, never exact float equality. If NULL is not
  declared, do NOT assume -999.25 or any other value.
- Undeclared sentinel scan across every column for candidate values (at least -999.25,
  -999, -9999, 9999, 1e30, -1e30, -1.0e+30): WARNING HEURISTIC ONLY. It never
  converts a value. Scientific notation is NOT nonconforming — do not flag it.
- LAS 3.0 features in a 2.0 file: detect and report; do not parse them optimistically.
- Duplicate mnemonics: legal; record them; identity is CurveRef.

Pass 2 — tolerant parse:
- You may use lasio. `ignore_header_errors=True` may be used only if every error it
  would suppress is first captured as a ledger entry; otherwise don't use it.
- Compare pass-2 results against pass-1 facts; every disagreement is a ledger entry.
- Build SourceFile and the source Dataset (read-only arrays; original mnemonic, unit
  and description verbatim; nulls represented as NaN in the array, with the fact that
  they were nulls recorded — Stage 4 owns sample status).

Ledger entries become Findings. Proposed mapping — APPLY ONLY WHAT I APPROVE HERE:
⟦LEDGER MAPPING — proposal to approve or edit:
  undeclared NULL → AMBIGUOUS_INTERPRETATION
  undeclared sentinel candidate → RULE_DEPENDENT_WARNING
  header line not splitting cleanly → INFORMATIONAL
  unrecognised section → INFORMATIONAL
  LAS 3.0 feature detected → RULE_DEPENDENT_WARNING
  column count irreconcilable after WRAP/comments/blanks → STRUCTURAL_DATA_FAILURE
  no ~A section or no data rows → STRUCTURAL_DATA_FAILURE
  encoding undeterminable → STRUCTURAL_DATA_FAILURE
  pass-1/pass-2 disagreement → INFORMATIONAL, unless it changes a value, then
    STRUCTURAL_DATA_FAILURE⟧
Any ledger condition not in that list: stop and report it; don't classify it yourself.

Harvest: read core/las_loading.py and core/well_loaders.py (and the monolith's LAS
reading by symbol). Reuse what is correct. Do not modify them. In the report, list
what you reused and what you deliberately did not, with reasons.

Tests — tests/stages/test_s00_ingest.py:
- A conformance matrix: parametrised tests built from small synthetic LAS strings,
  one per D65/D66/D68 rule, each test id carrying its evidence level
  (e.g. "PARTIAL-D65-first-char-after-tilde").
- Every fixture: curve count, row count, NULL, header depths equal the golden facts.
- Duplicate mnemonic file: two CurveRefs, both preserved.
- Undeclared NULL file: finding raised; no value converted.
- A value equal to a sentinel candidate but not the declared NULL stays a number.
- Input file hash unchanged; file opened read-only.

Also add polish/cli.py with `python -m polish ingest FILE --json` printing the ledger
and dataset summary to stdout (writes no files).

Report in the standard format, including the reuse list.
````

### P09 — Stage 1: Depth axis

**Needs:** P08. OD-10 is closed; the rounding test is recorded under D12.

````text
POLISH rebuild — P09: Stage 1 Depth axis.

Implements: D3 (as revised by F14), D8 (as resolved), D86, D88, D9 (recording only),
D28 (with F10), D67 (UNVER — record, don't enforce), C4, contracts §4 item 5.

Create polish/stages/s01_depth.py (alters_data=False, depends_on_stages=(0,)).

1. Index identification: the FIRST ~C curve governs (domain expert, 14 Sept). Confirm
   it is plausibly a depth index — its mnemonic is in a small hardcoded depth-alias set
   that lives IN THIS MODULE with a docstring explaining why (D28/F10: Stage 1 needs
   this subset before Stage 2 exists; do not "fix" this by moving it), and it is
   monotonic. If the first curve is not plausibly a depth index →
   STRUCTURAL_DATA_FAILURE; the stage halts with the reason (D3). Other depth-like
   curves (e.g. DEPTH_MD) are preserved unchanged and are NOT treated as ambiguity.
   Harvest the behaviour from commit dc0bd5d (locate by symbol).

2. Direction: Stage 1 accepts strictly ascending depth only (D3). A decrease, or a
   negative STEP, is a structural failure. Nothing is reversed or reordered; the
   file is left untouched and later stages do not run. User message: "valid LAS,
   but POLISH supports increasing depth only."

3. Regularity: REGULAR or IRREGULAR. STEP=0 as the irregular convention is UNVER
   (D67): record what the header says; do not require it.
   Blank STEP (domain expert, 14 Sept: "verify the depth curve and fill the step if
   appropriate; otherwise skip"): if the depth increments are constant within the same
   rounding rule as item 4, record the data-derived step (used in the exported header)
   and raise INFORMATIONAL. Otherwise: ⟦dad's 2.3 meaning — EITHER "skip the well
   (STRUCTURAL_DATA_FAILURE)" OR "accept as irregular (D88) and notify"⟧.
   Irregular-but-ascending grids with a declared STEP: accepted as irregular and
   recorded as irregular (D88, kept half); a regular step is never invented.

4. Header vs data (D8 as resolved via D86), each producing a Finding with evidence
   (declared value, observed value, difference, difference in sample intervals):
   - non-monotonic or duplicate depths → STRUCTURAL_DATA_FAILURE, no override
   - first curve not a plausible index → STRUCTURAL_DATA_FAILURE, halt
   - grid-inconsistent (a declared non-zero STEP that the data increments do not
     match under the rounding rule) → STRUCTURAL_DATA_FAILURE, no override
   - STRT/STOP differences: per the rounding rule below
   Domain-expert rule (D12, answered 14 Sept): "verify whether the difference can be
   due to a rounding error; if so, correct accordingly; otherwise skip the well."
   What counts as rounding is recorded under D12: a depth counts as rounding if it
   is within one full unit of its own last written decimal place of a grid built
   from STRT. The comparison is inclusive (exactly one unit passes). The full-unit
   rule and the per-depth precision rule are awaiting expert review.
   - Explained by rounding → INFORMATIONAL; the result records the data-derived value,
     which the exported header will use (D59); the original file is untouched.
   - Not explained by rounding → STRUCTURAL_DATA_FAILURE: the well is skipped and the
     user notified, with the declared value, observed value and difference.
   The rule and its interpretation live in the rule library entry
   "depth.header_data_rounding", with source "domain expert, worksheet 14 Sept".
   The report must include, for every fixture, which branch it took.
   Create polish/rules/ with a minimal loader and this one entry now; P12 extends it.

5. Depth unit: from the header. Absent or unrecognised → AMBIGUOUS_INTERPRETATION
   (a resolution point, contracts §4). Never assume feet or metres.

6. D9 copy rule: Stage 1 RECORDS the alias decision (source CurveRef, canonical name
   DEPT) as a result field. It does not create a curve. Materialisation belongs to
   export (P29, OD-18).

7. C4: any parameter derived here (e.g. observed step) is stored with the unit it was
   derived under, and a helper asserts that unit at point of use.

Tests — tests/stages/test_s01_depth.py: ascending; a decrease or a negative STEP
is a structural failure and the samples stay as written; irregular ascending;
duplicate depth; non-monotonic; first curve not a depth curve;
STOP differing by pure decimal rounding; STOP differing by less than one
step but not by decimal rounding; STOP differing by more than one step; blank STEP
with regular data; blank STEP with irregular data; missing depth unit; two depth
curves (the first governs, the second is preserved unchanged); every fixture against
golden facts; data never reordered.

Report in the standard format, including how many fixtures raise each finding kind.
````

### P10 — Stage 2: Identification

**Needs:** P09. OD-17 (TVD role name). Decide whether Cursor may read your audit folder (slot).

````text
POLISH rebuild — P10: Stage 2 Identification.

Implements: D31, D32 (with F2), D33, D34, D35, D36, D37, D38, D39, D40, D41, D42,
C3, UXC1, UXC9.

Create polish/stages/s02_identify.py (alters_data=False, depends_on_stages=(0,1)) and
polish/tables/.

1. Tables as versioned data files in polish/tables/ (shipped, read-only):
   - families: id, display name, accepted unit dimension, tool_group (D41, may be
     null), interpolation_domain (D42: LINEAR | LOG | null), display_log_scale
     (a DISPLAY attribute only — never used to choose an interpolation domain, F7),
     source, source_tier.
   - mnemonics: mnemonic → family, qualifiers, role, source, source_tier.
   - vendor_prefixes (D35), qualifier tokens (D39: _MAIN = main pass, _REF = repeat
     pass; _SYNT = synthetic source, D40).
   Values for tool_group and interpolation_domain: ⟦OD-13, OD-14 — until answered,
   ship null for every family except resistivity interpolation_domain = LOG (stated
   in D52)⟧.

2. Migrate the existing mnemonic database (P01 item 11) into these files. Preserve each
   entry's source. Do NOT invent a tier: an entry without traceable evidence becomes
   PROPOSED. Report counts per tier after migration.

3. Local extension tables (D37, D38): loaded from the user's app-data directory if
   present, never transmitted, recorded at tier USER_DEFINED. Path: ⟦path, e.g.
   %APPDATA%\POLISH\tables\⟧. Missing directory is fine; don't create it here.

4. Output per CurveRef: family (or none), qualifiers, role, evidence_tier,
   processing_eligibility (per the approved D34 mapping in the state glossary), and a
   match trace (which table, which entry, which rule — exact, qualifier split, prefix
   strip proposal).

5. Roles: INDEX (from Stage 1), MEASURED_LOG, DRILLING, ANCILLARY, SYNTHETIC_SOURCE,
   ⟦OD-17: TVD role name⟧, UNKNOWN. UNKNOWN declines downstream processing (C3) and is
   preserved.

6. Matching rules: short mnemonics exact-match only (D36); unrecognised trailing
   tokens are never silently stripped (D36); _MAIN/_REF are qualifiers, never noise
   (D39); *_SYNT → SYNTHETIC_SOURCE (D40).

7. Vendor prefixes (D35) are NOT stripped automatically. Detect groups of curves
   sharing a recognised prefix and raise ONE AMBIGUOUS_INTERPRETATION finding per
   group, with evidence: prefix, table version, member CurveRefs, and for each member
   the match it WOULD get after stripping. (P15 turns this into a batch decision.)

8. Duplicate mnemonics (e.g. two GR): each CurveRef identified independently; raise an
   INFORMATIONAL finding naming both.

8a. Main and repeat passes (D39; domain expert, 14 Sept): when a file has more than
   one pass of a family (e.g. GR, GR_MAIN, GR_REF), raise ONE AMBIGUOUS_INTERPRETATION
   finding per family asking which pass to use by default, with scope options
   THIS_WELL and ALL_WELLS (ALL_WELLS = an explicit local rule, P13). No default is
   chosen by POLISH. Every pass is preserved.

8b. Unrecognised mnemonics (domain expert, 14 Sept: "leave unchanged and notify"):
   they stay UNKNOWN, are preserved exactly, are not processed, and each raises an
   INFORMATIONAL notice. Specifically RM, RD, RACEHM, RACELM, RPCEHM, RPCELM,
   ABDCQF01–04, NBGRCFM and ROP must NOT be mapped to any family, despite their units.

9. Observed-tier candidates: ⟦EITHER "Read
   C:\Users\achav\Downloads\LAS_AUDIT_OUTPUT\master_audit_curves.csv and write
   docs/review/observed_tier_candidates.md: mnemonic, frequency, units seen,
   suggested family. Do not add any of them to the tables." OR "Skip."⟧

Harvest: core/curve_identification.py (read, reuse correct logic, don't modify).
Known trap to test against: a family listing both G/CC and KG/M3 with a single range.

Tests — tests/stages/test_s02_identify.py: GR_MAIN → GR + main pass; GR_REF → GR +
repeat pass; LFP_DT_SYNT → DT, SYNTHETIC_SOURCE, qualifiers vendor LFP + synthetic
(after the prefix decision; before it, a group finding exists); AT90 → induction
resistivity, 90-inch; duplicate GR; an unknown mnemonic stays UNKNOWN with no range;
LFP fixture: the prefix-group size equals the count computed from golden facts
(never a literal); no family's display_log_scale is read by any interpolation code
(static test).

Report in the standard format, including recognition rate per fixture (recognised /
total, excluding the index).
````

### P11 — Stage 3: Units

**Needs:** P10. OD-05.

````text
POLISH rebuild — P11: Stage 3 Units.

Implements: C1, C4, C6, contracts §3 and §4 items 2 and 4, D87 (unit fields), OD-05
as decided: unit conversion is a recorded re-expression, never a CHANGED sample.

Create polish/stages/s03_units.py (alters_data=False, depends_on_stages=(0,1,2)) and
polish/tables/unit_spellings (data file: spelling → canonical unit, dimension, source).

1. Unit interpretation per CurveRef: parse the declared unit through the spellings
   table. Outcomes: RECOGNISED (canonical unit), ABSENT, EMPTY, UNRECOGNISED,
   INCOMPATIBLE_WITH_FAMILY (dimension mismatch with the family's accepted dimension).
   Every outcome other than RECOGNISED produces a Finding (ABSENT/EMPTY/UNRECOGNISED →
   AMBIGUOUS_INTERPRETATION; INCOMPATIBLE → RULE_DEPENDENT_WARNING). Never assume.
   The GR case — GAPI vs CPS, and GR with absent/empty/unrecognised unit — must be
   explicit tests (regression for commit fa11217).

2. Target unit system (imperial / metric / as-recorded) is a resolution point
   (contracts §4.4). No default. Until decided, the target is AS_RECORDED and nothing
   is converted.

3. Conversion, per OD-05(a): a ConversionRecord (from unit, to unit, exact factor and
   offset, source of the factor) attached to the curve's view. Values in the Dataset
   are never touched. Provide read_values(curve_ref, unit) that converts on read, and
   require_unit(values_view, reference_unit) that raises a UnitMismatch error (C1 —
   hard error, never a coercion). Unit pairs with no fixed factor (e.g. GR in CPS vs
   API units) have no conversion and are refused with a recorded reason.

4. C4: any depth-derived parameter carries the depth unit it was derived under;
   converting the depth unit invalidates it (the dependency layer in P14 will mark it
   STALE; here, provide the declaration).

5. Declare dependencies: Stage 4 results depend on this stage's unit interpretation
   and target system.

Harvest: core/unit_standardization.py conversion tables (read, reuse factors with their
sources, don't modify). Regression tests for the August failures: G/CC → KG/M3
conversion must never be compared against a g/cc range; depth converted to metres must
invalidate a step derived in feet (the 3.28× decimation).

Tests — tests/stages/test_s03_units.py: every outcome above; round-trip conversion is
exact within floating tolerance; source arrays unchanged after conversion; require_unit
raises on mismatch; CPS↔API has no conversion; the two August regressions.
Report in the standard format.
````

### P12 — Stage 4: Validity, rule library, gap inventory

**Needs:** P11. OD-12 and OD-13 answered — or accept definitional-only rules and "not eligible" everywhere else for now.

````text
POLISH rebuild — P12: Stage 4 Validity assessment, rule library, gap inventory.

Implements: D11 (as revised by F15), D87, D44, D49 (as revised by F6, softened), D50,
D66, D26, C2, C3, UXC6 (interval summaries), UXC9.
This is the keystone stage (D26). Take care.

Create polish/stages/s04_validity.py (alters_data=False, depends_on_stages=(0,1,2,3)).

1. Rule library (extend polish/rules/): each rule is a data entry with exactly the D87
   fields — family; accepted and canonical units; rule type (DOMAIN_CONSTRAINT |
   HARD_IMPOSSIBILITY | PLAUSIBILITY | CONSISTENCY); bound values; environmental
   qualifiers; source; source tier; library version; violation effect (INVALID |
   WARNING | REFUSAL); overridable (bool) and how an override is recorded.
   Ship the domain expert's values (worksheet, 14 Sept), each with source
   "domain expert, worksheet 14 Sept" and the rule type he confirms:
     GR           GAPI    0      to 300       type ⟦1.0.1⟧
     RHOB         g/cc    1.5    to 3.5       type ⟦1.0.1⟧
     NPHI         v/v     -0.15  to 0.6       type ⟦1.0.1⟧
     DT           us/ft   40     to 300       type ⟦1.0.1⟧
     PEF          b/e     -2     to 270       type ⟦1.0.1⟧
     resistivity  ohm-m   -0.02  to 20000     type ⟦1.0.1⟧
     SP           mV      -150   to 50        type ⟦1.0.1⟧
   Until a type is confirmed, that rule ships as PLAUSIBILITY (warning only), so no
   measured sample is marked invalid on an unconfirmed rule. Rules apply in other
   units only through exact conversion (OD-05); GR rules never apply to CPS curves.
   A rule applies only if the curve's unit (Stage 3) matches the rule's unit (C1) and
   the curve is eligible (Stage 2).

2. Per-sample status mask on the source dataset (stored as a Stage 4 result, not by
   mutating arrays): MEASURED; INVALID only when a HARD_IMPOSSIBILITY or
   DOMAIN_CONSTRAINT rule is violated (the sample stays visible and marked);
   ABSENT for nulls. PLAUSIBILITY violations raise a warning and the sample stays
   MEASURED. Non-finite stored values (±inf): ⟦approve — "INVALID under a named domain
   rule 'non-finite value'"⟧.

3. Validation outcome per curve (UXC9): PASSED (with rule-set version),
   PLAUSIBILITY_WARNING, HARD_VIOLATION_EXCLUDED, NOT_ELIGIBLE (low family or unit
   confidence — never INVALID, F15), PENDING, NOT_APPLICABLE. A curve with no
   applicable rule is NOT_ELIGIBLE, never PASSED.

4. Measured interval per curve: first to last non-absent sample.

5. Gap inventory per curve: every contiguous ABSENT run, with depth span and sample
   span. Classification (D49 corrected):
   (1) outside the measured interval → OUTSIDE_INTERVAL
   (2) interior gap shared across curves of the same tool_group → TOOL_ABSENCE_EVIDENCED
       — evidence: which curves share it, overlap measure, confidence. This is evidence
       consistent with tool absence, not proof. "Shared" means: ⟦OD-13⟧.
       If the curve has no tool_group: record "no co-tool evidence available" and fall
       through.
   (3) interior gap shorter than the tool's vertical resolution → DROPOUT
       (resolution per tool: ⟦OD-13⟧; if unset, this case cannot apply — record why)
   (4) otherwise → AMBIGUOUS (threshold: ⟦OD-13⟧; if unset, every remaining interior
       gap is AMBIGUOUS)
   Every gap: fill_outcome = NOT_REQUESTED. TOOL_ABSENCE_EVIDENCED and
   OUTSIDE_INTERVAL are marked structurally non-fillable (D50) — a property of the
   class, with no field or flag that changes it.

6. Interval summaries for rendering (UXC6): per curve, merged spans of
   {measured, invalid, absent-by-class}. No pixel logic here.

7. Findings:
   - HARD / DOMAIN violations (domain expert, 14 Sept: "prompt whether to delete the
     anomalous value, for this well only or for all wells; never clip"): the samples
     are marked INVALID and stay visible. ONE AMBIGUOUS_INTERPRETATION finding per
     curve per rule asks: exclude these samples from the processed output, or keep
     them? Scope options THIS_WELL or ALL_WELLS (explicit local rule, P13). Nothing is
     excluded until the user chooses. Nothing is ever clipped. The original file is
     never altered. The finding lists every affected depth.
   - PLAUSIBILITY → RULE_DEPENDENT_WARNING.
   - NOT_ELIGIBLE curves → INFORMATIONAL with the reason and what evidence would make
     it eligible.

Harvest: read petrophysics/constants.py (the PREFERRED depth-normalised table, D51 —
feet-only) and the monolith's gap-labelling code by symbol. Note in the report where
the legacy code computed a gap label and then ignored it.

Tests — tests/stages/test_s04_validity.py and tests/scenarios/test_scenario1_engine.py:
- KEOUGH (if available): RHOB samples above the density tool's first measured depth
  are ABSENT and classified OUTSIDE_INTERVAL; the interval excluded under OD-11 is not
  asserted.
- Synthetic: an interior gap shared by RHOB/DRHO/PEF with a test tool group →
  TOOL_ABSENCE_EVIDENCED, non-fillable, evidence lists all three.
- A curve with no rule → NOT_ELIGIBLE, never PASSED.
- A GAPI rule is never applied to a CPS curve.
- Report every sample each shipped rule flags on every fixture (plan 1.0.1 has the
  expected counts from an independent scan on 15 Sept — any difference must be
  explained).
- An out-of-limit value is never excluded without a recorded decision, and never
  clipped.
- A value equal to an undeclared sentinel is not treated as absent.
- Sample status mask never changes source arrays.
- Scenario 1 (supplement Part F) at engine level: the interval exists in the
  inventory, is non-fillable, carries evidence and confidence, and appears in
  findings.
Report in the standard format, including per-fixture counts of each gap class.
````

### P13 — Decisions, dispositions and sessions (engine)

**Needs:** P12. OD-22.

````text
POLISH rebuild — P13: Decisions, dispositions and sessions.

Implements: D46, D47, D48, D63 (hash pairing), D79 (as revised by F17), D86, UXC2,
UXC4 (scope field), supplement §4.3 (history is preserved).

Create polish/decisions/.

1. DecisionRecord (from the state glossary): id; finding ids; action (from
   permitted_actions); chosen option; rationale (required for interpretation choices
   and for DEFERRED); scope (see 4); actor (USER | DEFAULT_APPLIED); timestamp; app
   version; rule-library and table versions in force; input FileHash.
   Records are append-only. Changing a decision creates a new record that supersedes
   the old one; the old one stays in history.

2. apply_decision(session, finding, action, ...) validates the action against
   permitted_actions(finding.block_type) and refuses anything else with a clear error.
   There is no function that dispositions a STRUCTURAL_DATA_FAILURE,
   SCIENTIFIC_PRECONDITION_FAILURE or EXPORT_CONFORMANCE_FAILURE.

3. Session file (D47): decisions only — never data values. JSON with a schema version
   and the input FileHash. Location: beside the input file, name ⟦pattern, e.g.
   <input-stem>.polish-session.json⟧. If that directory is not writable: raise a
   Finding and do not write anywhere else unless the caller passes an explicit
   alternative path chosen by the user. Preferences with no well information live in
   app-data (⟦path⟧) and never contain a FileHash, well name or depth.

4. Scope (UXC4, OD-22): every DecisionRecord has scope ∈ {THIS_ITEM, SELECTED_ITEMS,
   DETECTED_GROUP, ALL_MATCHING_IN_FILE, LOCAL_RULE}. In the interface, LOCAL_RULE is
   labelled "all wells" (the domain expert's wording). Which scopes each decision type
   allows: ⟦OD-22 table — at minimum, pass selection (P10 8a) and out-of-limit exclusion
   (P12 7) allow THIS_WELL and LOCAL_RULE⟧. LOCAL_RULE, if allowed, writes a USER_DEFINED table entry
   (P10 local tables) carrying the decision id, and is only ever created by an explicit
   call — never as a side effect.

5. Loading a session (D48):
   - hash matches → decisions re-applied; each re-applied decision is re-validated
     against current findings; any that no longer match a finding are reported, not
     silently dropped.
   - same filename, different hash → an AMBIGUOUS_INTERPRETATION finding: "this looks
     like the same well but the raw file has changed since these decisions were made".
     Nothing is applied until the user chooses.
   - no session → nothing applied; no settings carried over from any other well.

6. Pure query: outstanding(session) → findings lacking a terminal disposition, grouped
   by block type.

Tests — tests/decisions/: every block type × every action (permitted or refused);
DEFERRED without reason refused; superseding keeps history; session contains no
numeric data values (scan the JSON); hash-mismatch path; unwritable directory path;
preferences file never contains a hash or depth; LOCAL_RULE only via explicit call.
Report in the standard format.
````

### P14 — Dependency graph and staleness (engine)

**Needs:** P13.

````text
POLISH rebuild — P14: Dependency graph and staleness.

Implements: D80, UXC3, C4, supplement §4.3, supplement Part F scenario 3 (engine level).

Create polish/deps/.

1. Graph nodes: each stage result, each decision, each derived parameter (C4).
   Edges: from StageContract.depends_on_stages, from decisions to the stage results
   they influence, and from declared parameter dependencies (P09, P11).

2. When an upstream decision changes or a stage is re-run with different inputs:
   mark every dependent node STALE with a StaleCause (triggering node, timestamp,
   human-readable change description). Do NOT recompute automatically. Stale results
   remain readable and are labelled non-current.

3. Dispositions attached to findings of a stale stage become STALE too; their
   history is kept; the user must re-disposition or re-run.

4. impact_summary(trigger) → structured object with: what changed; affected stages;
   count and list of affected decisions; whether data was altered (for Stages 0–4:
   always "No data has been altered"); whether export is blocked; recovery actions
   available (re-run stage N; re-disposition items). This is what UXC3's panel will
   render.

5. export_permission (P05) consumes dependency_state: any STALE node that affects
   exported claims blocks export. Define "affects exported claims" as: ⟦approve —
   "every Stage 0–10 result and every decision, until P28 defines a narrower set"⟧.

6. Re-running a stale stage clears STALE only for nodes whose inputs are now current.

Tests — tests/deps/ and tests/scenarios/test_scenario3_engine.py:
Scenario 3: resolve a unit after Stage 4 has run → Stage 4 and its gap classifications
and dependent decisions are STALE; nothing was silently recomputed; the prior decision
is still in history; export_permission is False with a reason naming the trigger;
re-running Stage 4 clears it. Also: a presentation-only change (none exist in the
engine yet — assert the API has no such entry point) never marks anything stale.
Report in the standard format.
````

### P15 — Batch decisions and scope preview (engine)

**Needs:** P14. OD-22.

````text
POLISH rebuild — P15: Batch decisions with scope preview.

Implements: D35, D82, UXC4, supplement §4.2, supplement Part F scenario 4 (engine level).

Create polish/decisions/batch.py.

1. BatchProposal: action; grouping evidence (why these belong together — rule, table,
   version); exact affected CurveRefs; non-affected CurveRefs in the same file;
   allowed scopes for this decision type (P13); consequences (for each member, the
   result after the action — e.g. the match after prefix stripping, including members
   that would still be UNKNOWN); exceptions (members the user excluded before commit).

2. Sources of proposals: the prefix-group findings from P10; later, any set of
   identical findings of one block type.

3. commit_batch(proposal, excluded=...) produces ONE DecisionRecord listing all
   members, plus per-member outcome, and dispositions every member finding. Excluded
   members keep their findings, undispositioned.

4. Committing a batch triggers P14 staleness for everything downstream of Stage 2.

5. The proposal must be fully inspectable without committing (a pure preview).

Tests — tests/scenarios/test_scenario4_engine.py on 159-19A_LFP.las:
affected count equals the number computed from the golden facts (never a literal —
and report that number); non-affected list is exact; one DecisionRecord; excluding
three members leaves exactly those three undispositioned; preview makes no change;
members that would remain UNKNOWN after stripping are listed as such in consequences.
Report in the standard format.
````

### P16 — Describe report and CLI

**Needs:** P15. Relevant to OD-27.

````text
POLISH rebuild — P16: Describe report (read-only output) and CLI.

Implements: brief §3 (refusals itemised as prominently as actions), D22, D64, UXC1,
UXC7, UXC9, D19 (output path chosen by user; input path blocked).

1. polish/provenance/describe.py builds a DescribeReport from Stages 0–4 plus the
   session: file identity (path, SHA-256, size); ledger; depth axis result; per-curve
   identification (family, qualifiers, role, tier, eligibility, match trace); units;
   validity outcome; gap inventory with classes, evidence and confidence;
   findings grouped by block type with dispositions; refusals and non-eligibility in
   their own section of equal prominence to everything else; rule-library, table and
   app versions; "not verified" items (e.g. PARTIAL / UNVER rules in force).

2. Two renderings: JSON (with a JSON Schema at polish/provenance/schema/describe.schema.json,
   versioned) and a human-readable HTML or Markdown file (⟦choose⟧). Both are
   produced from the same object.

3. Writing: only to a directory the caller names. Refuse if the output path equals the
   input file or its directory would require overwriting the input. Never write beside
   the input unless that directory is what the user chose.

4. CLI: `python -m polish describe FILE --out DIR [--session PATH]`. Exit codes: 0 ok;
   2 structural refusal (report still written, stating the refusal as the result);
   1 error.

5. Use of plain-language composites from UXC1 and UXC9 in the human rendering
   (e.g. "Measured — excluded from processing: hard-limit violation"; "Not
   automatically validated"). Never the bare words "validated", "flagged" or "unknown"
   without the field they refer to. User-mapped curves always carry "User-mapped — not
   identified by POLISH". Every DEFERRED item is listed with its reason.

Tests: JSON validates against the schema for every fixture; a structural-refusal
fixture (synthetic) produces a report whose headline states the refusal as a result;
the refusals section exists even when empty ("No refusals"); no output path can equal
the input; golden facts appear correctly in the report.
Report in the standard format, and attach the human-readable report for 30_3-5 S.las.
````

---

## Read-only trust surface

### P17 — Headless render model

**Needs:** P16. OD-07 decided (P04 numbers in hand).

````text
POLISH rebuild — P17: Headless render model (toolkit-agnostic).

Implements: D73 (as revised by F12), D74, UXC6, D78, D57, D84, C7.2, C7.4.

Create polish/render/. No toolkit imports.

1. bin_curve(values_view, status_view, depth_window, pixel_count) → per-pixel bins:
   min and max of MEASURED values; flags any_created, any_changed, any_resampled,
   any_invalid, any_absent, all_absent. D74: if any sample in a pixel is created or
   changed (or resampled), the pixel is marked modified. Never average measured and
   synthesised values into one mark. Absent renders as a gap, never zero (C7.4).
   Everything is positioned by depth, never by row index (C7.2).

2. Full-well mode (UXC6): exposes only modified vs unmodified, plus major absent /
   refused intervals. "Major": ⟦OD-25, e.g. "≥ 1 pixel at the current scale"⟧.
   Finer distinctions appear only at mid-zoom and below. Zoom thresholds: ⟦OD-25⟧.

3. Interval bands: merged spans per curve from Stage 4 summaries and overlays.

4. hit_test(track, depth, tolerance_px) → nearest actual SampleRef (never an
   interpolated display position), and annotation(sample_ref) → value, unit (with any
   conversion stated, C7.3), depth, sample_status, method, interpolation domain,
   inputs (bracketing samples with depths and values, or source curves and parameters),
   uncertainty — which may be the literal state NOT_QUANTIFIABLE_FOR_METHOD (D57).
   At this milestone only MEASURED / INVALID / ABSENT exist; build the structure for
   all states. User-mapped curves carry "User-mapped — not identified by POLISH".

5. record(sample_ref) → the full provenance record (third disclosure layer, D84).

6. Performance: bin a 20,000-sample curve into 300 pixels in ⟦target from P04⟧.
   Use numpy vectorisation; no Python loops over samples.

Tests — tests/render/: a three-sample synthetic overlay stays visible at full-well
scale and at every zoom level (the "vanishing fill" failure); a pixel with one created
sample among 60 measured is marked modified; absent never bins to zero; depth
positioning follows the recorded ascending depth axis (D3); a decreasing depth file does not reach this view; hit_test never returns a position between
samples; annotation of a linear-interpolated synthetic sample reads its two inputs and
NOT_QUANTIFIABLE_FOR_METHOD; performance test marked and reported.
Report in the standard format.
````

### P18 — Interface shell

**Needs:** P17, OD-07, Appendix B keymap approved.

````text
POLISH rebuild — P18: Interface shell.

Toolkit: ⟦OD-07, e.g. "PySide6 with <plot layer from P04>"⟧. Add it to requirements.txt
with its licence noted in the report. Create polish_ui/. It imports polish/; polish/
never imports it (the P06 layering test must keep passing).

Implements: brief §7.1, §7.3, §7.5, §9; D71, D81, D83, D14; UXC5; supplement §3
(fixed workspace regions).

1. Fixed regions, each with space reserved from the start: stage rail; coverage track;
   log-track area; resolution queue (side panel); detail/record pane; export review
   (reserved, empty for now). Nothing that appears may move anything else: showing or
   hiding a panel never changes the geometry of the depth view.

2. Load flow: File → Open. No settings dialog first. A background worker runs
   Stages 0–4 through the runner. No modal dialogs during processing. Progress shows
   which stage is running and what it found so far (e.g. "Stage 2 — 171 curves,
   1 prefix group found"), never a bare progress bar. A structural refusal is shown
   in the stage rail and main area as a result, in the UXC2 wording
   ("This file cannot proceed past Depth Axis"), not as an error dialog.
   The first thing shown is the depth-axis and units result (domain expert's first
   three checks, worksheet 4.1).

3. Stage rail: Stages 0–10, each showing NOT_RUN / RUNNING / COMPLETE / STALE /
   BLOCKED / REFUSED. Any stage whose preconditions hold can be selected; this is a
   feature tree, not a wizard. Stages 5–8 show as available-but-off.

4. Keyboard: all shortcuts from one config module, per Appendix B:
   ⟦PASTE APPROVED KEYMAP⟧. A help overlay lists them.

5. Friction (D83): presentation actions (zoom, pan, selection, overlay visibility,
   panel visibility) are instant, with no confirmation. They call no engine function
   that changes analytical state (UXC5) — enforce with a test that spies on the
   engine API.

6. No state persists between wells except app-data preferences (P13). Opening a second
   file starts clean.

7. Styling: one theme module. Keep it plain and dense until OD-25 is decided. Do not
   invent visual language for provenance states here.

Tests (pytest-qt or equivalent; list the dependency): panel toggle leaves the depth
view geometry identical; opening a file shows the describe state without any dialog;
a structural-refusal fixture shows the refusal in the rail; presentation actions make
no engine state-changing calls; opening a second file carries no state over.
Report in the standard format, with a screenshot of the shell on 159-19A_LFP.las saved
to docs/review/P18_shell.png.
````

### P19 — Coverage track

**Needs:** P18. **OD-25 Q1 (visual language) decided** — don't send without it.

````text
POLISH rebuild — P19: Coverage track.

Implements: brief §4.3, §4.4, §7.1, §7.2, §10 Q1; D70, D72, D73, D74, UXC6.

1. Render every curve (all 171 on the LFP fixture) as a compact column over the full
   well, using polish/render bins only.

2. Visual language: ⟦OD-25 Q1 — paste the approved spec: states, encodings per zoom
   level, row ordering, labels, greyscale behaviour⟧.
   Colour encodes curve identity only (D72). Provenance and validity states must
   differ in at least one non-hue channel (pattern, hatch, fill density, line style).

3. Full-well scale shows modified vs unmodified and major absent/refused intervals only
   (UXC6). A tooltip or legend at full-well scale must say that finer distinctions
   require zooming (F12 — a deliberate limitation, stated).

4. Interactions (presentation only, UXC5): click a depth → log tracks scroll there;
   select curves → they become candidates for the log-track area (P20).

5. This is the landing view: it renders as soon as Stage 4 completes, with no settings
   and no processing beyond Stages 0–4.

Tests: every state in the spec has a distinct non-hue encoding (assert from the style
spec, and render to an image, convert to greyscale, and assert each state's sample
region differs measurably — document the metric); decimation matches polish/render
(no second implementation); 171 columns render within ⟦time⟧; a three-sample
synthetic overlay is visible at full-well scale.
Report in the standard format, with full-well screenshots of all fixtures in colour and
greyscale saved to docs/review/P19/.
````

### P20 — Log tracks, curve selection, hover and record

**Needs:** P19. OD-25 Q2 (curve selection) and Q8 (density).

````text
POLISH rebuild — P20: Log tracks, curve selection, hover annotation, record pane.

Implements: brief §4.3, §4.6, §7.1, §7.2, §7.6, §10 Q2; D72, D76 (structure), D78,
D84, D88 (display only), C7, UXC5.

1. Log-track area: up to ⟦8⟧ tracks. Default order, left to right (domain expert,
   14 Sept — overrides brief §7.1): compressional sonic (DTP), shear sonic (DTS),
   density, porosity, water saturation (Sw), GR, deep induction resistivity (ILD).
   Matching is by family, never by mnemonic. A family not present in the file:
   ⟦dad — show an empty labelled track, or skip it⟧. The conventional petrophysical
   order (gamma → resistivity → porosity/density) is provided as a saved view.
   Curve colours by family per brief §4.3 (GR green or black; RHOB red; NPHI blue;
   resistivity black and red on a log scale). Resistivity's log display uses the
   family's display_log_scale attribute (a display attribute only, F7).

2. Depth increases downward regardless of the file's direction (display only; data are
   never reordered, D88).

3. Curve selection for 171 curves: ⟦OD-25 Q2 — approved mechanism, e.g. search box;
   filters by family / role / tier / eligibility / prefix group; selection from the
   coverage track; saved view presets as presentation-only configuration (UXC5)⟧.
   A preset never changes identification, validation, gap class or export contents —
   test it.

4. Hover (disclosure layer 2): annotation from polish/render in a fixed readout area
   (reserved space; no floating window that covers the depth view), per brief §4.6.
   Invalid samples remain visible and marked (brief §4.2).

5. Record pane (layer 3): full record for the selected sample or interval, including
   rule id and version, match trace, and decision history.

6. Units: values display in the curve's current view unit, with any conversion stated
   next to the axis (C7.3). A unit mismatch between two series on one track is shown
   as an error in the track, never drawn silently.

7. Density: ⟦OD-25 Q8⟧.

Tests: default track order matches the domain expert's order on every fixture; an accepted file is drawn with increasing depth (D3); a decreasing synthetic file does not reach this view;
hover returns an actual sample; the readout never overlaps
the tracks; presets make no engine state change; C7.3 mismatch shows an in-track
error.
Report in the standard format, with screenshots at full-well, ~200 ft and ~10 ft to
docs/review/P20/.
````

### P21 — Resolution queue, refusals panel, batch review, stale surfaces

**Needs:** P20. OD-25 (queue hierarchy, if not taken from supplement §4.1).

````text
POLISH rebuild — P21: Resolution queue, refusals panel, batch review, staleness UI.

Implements: brief §4.7, §4.8, §4.9, §7.4, §8; D29, D79, D80, D81, D82, D86; UXC2,
UXC3, UXC4, UXC7; supplement §4.1–§4.3.

1. Queue (side panel, never modal, never covers the depth view). Every item has the
   identical anatomy in the identical position (supplement §4.1):
   WHAT POLISH OBSERVED / WHAT IT MEANS / WHAT YOU MAY DO / WHAT HAPPENS NEXT.
   Only actions from permitted_actions(block_type) are shown — no disabled
   "accept anyway" buttons, no hidden ones. Wording from the UXC2 table.
   Selecting an item scrolls the depth view to its subject (presentation only).

2. Keyboard traversal per the keymap: next, previous, open record, and each permitted
   action. A user can work the whole queue without the mouse.

3. Rationale entry is inline (no dialog) where the action requires it.

4. Batch review (UXC4, supplement §4.2): for a group proposal show action, grouping
   evidence with table version, exact affected count, preview list, non-affected
   list, scope options, and per-member exclusion before commit. One commit action,
   labelled with the count ("Apply to N curves"). Uses P15 only.

5. Refusals panel (UXC7): every refusal and non-eligibility as a completed,
   evidence-backed outcome — what was not done, why, where, whether the raw
   measurement remains visible, and whether it blocks export or only the method.
   Not styled as errors. ⟦OD-25: how it earns prominence — layout choice⟧.

6. Staleness (UXC3), in three places: the stage rail (STALE); each affected queue item
   and view (marked non-current, prior result still visible); a persistent impact
   summary built from P14's impact_summary, with direct recovery actions
   (re-run stage / go to items). Copy example from UXC3.

7. Decisions go through P13–P15 only. The UI keeps no decision state of its own.

Tests (UI-level): an item's anatomy and position are identical across block types;
a structural finding shows no disposition controls; traversal and every permitted
action work from the keyboard alone; Scenario 4 (supplement Part F) end to end in the
UI on the LFP fixture — one batch, exact count computed from golden facts, exclusions
work, one record, no per-curve dialogs; Scenario 3 end to end — changing a unit
decision shows STALE in all three places and blocks export in the readiness summary;
refusals panel renders a scientific-precondition refusal (synthetic) with all UXC7
fields.
Report in the standard format, with screenshots to docs/review/P21/.
````

### P22 — Export readiness review (read-only milestone)

**Needs:** P21. OD-27.

````text
POLISH rebuild — P22: Export readiness review and read-only export.

Implements: brief §7.5; D17, D19, D79, D83, D86, D14; UXC10; supplement Part F
scenario 6.

1. Export review region (reserved in P18). Shows, separately: selected export profile
   (only those valid at this milestone: ⟦OD-27⟧); source file identity and SHA-256;
   active dataset (source only at this milestone); transformation summary ("None —
   no values have been altered"); outstanding reviewable dispositions; deferred items
   with their reasons; non-overridable blockers; stale dependencies; the list of
   artifacts that will be written.

2. The confirm control is enabled only when export_permission (P05) says so. The
   reasons list is shown verbatim when it doesn't. There is no other path to export.
   Wording from UXC2: "Export is blocked until this is corrected" — never "force".

3. Writing (deliberate, D83): a save dialog with a suggested name; the input path and
   any path that would overwrite the input are rejected (D19); a final summary step
   states exactly what will be written; then write. Output at this milestone:
   ⟦OD-27 — recommended: "the describe report (HTML + JSON), the session file, and
   QC images"⟧.

3a. QC images (domain expert, 14 Sept — the thing other tools don't give him): PNG
   (and ⟦PDF?⟧) of the coverage track and the current log tracks over ⟦full well /
   chosen interval⟧, rendered from polish/render only, with file name, input SHA-256,
   POLISH version, date and legend printed on the image. They must remain legible in
   greyscale (test as in P19).
   The original file is never opened for writing (D17).

4. "Review completed" and "export permitted" are displayed as two different facts.

Tests: Scenario 6 — one stale decision + one accepted ambiguity + one structural
failure → accepted ambiguity shown as recorded and permitted; stale item shown with
its recovery path; structural failure shown as non-overridable; confirm disabled; no
force control exists (search the UI tree for any action whose text matches
/force|anyway|override/i and assert none); input hash unchanged after a real export;
the output path cannot equal the input.
Report in the standard format.
````

**End of the read-only milestone.** This is where OD-19 (D16) gets answered for real: release here, or continue.

---
## Data-altering (drafts — re-check each with me before sending)

These prompts are complete enough to send, but written before the read-only milestone exists. Expect to revise them against what P08–P22 actually produced.

### P23 — Lineage and Stage 5: Resampling

**Needs:** P22, OD-03, OD-13, OD-14, OD-15, OD-17, OD-19.

````text
POLISH rebuild — P23: Dataset lineage and Stage 5 Resampling.

Implements: D2, D25, D43 as resolved (child dataset per resample, OD-03), D7 (with F2,
F13), C2, C4,
contracts §4 item 5, UXC8, supplement Part F scenario 5.

1. polish/stages/s05_resample.py (alters_data=True, default OFF, depends on 0–4).
   Output is a CHILD dataset: new DatasetId with parent id; its own DepthGrid; a
   recorded source-to-child depth mapping. The parent is untouched and remains the
   only thing called "original".

2. Target spacing is a resolution point (contracts §4.5): native increment, or an
   explicit value with its unit. No default. C4: the spacing carries its unit and is
   asserted at use.

3. Interpolation: bounded — only between two measured samples of the parent that
   bracket the child depth; never across a gap longer than ⟦OD-13/OD-15 rule⟧; never
   beyond the measured interval (C2). Domain from the family's interpolation_domain
   (D42); null domain → the curve is not resampled and a
   SCIENTIFIC_PRECONDITION_FAILURE finding records why. LOG domain with a non-positive
   bracket → not resampled at that depth, recorded (D52).

4. Child samples: RESAMPLED where interpolated, with a provenance record (method,
   domain, the two parent SampleRefs and values, child depth); ABSENT where no
   bracket exists. A child sample that coincides exactly with a parent depth: ⟦approve
   — "MEASURED, with a pointer to the parent SampleRef"⟧.

5. TVD (⟦OD-17 role⟧): resampled with the index (D7). Where the parent's increment
   changes by more than ⟦OD-17 threshold⟧ within a bracket, raise a
   RULE_DEPENDENT_WARNING citing F13.

6. Lineage record for UXC8: active dataset, parent, grid relationship, ordered
   transformation list.

7. Staleness: changing any Stage 5 parameter marks the child and everything
   downstream STALE.

8. Provenance is stored per interval (one record per contiguous resampled run), never
   one Python object per sample. Per-sample detail is computed on request.

9. Scale test: resample all eligible curves of 159-19A_LFP.las and of the 12 MB
   fixture, at native step and at one-fifth of it. Report peak memory and wall time.
   Fail if ⟦limits, set after P04⟧.

Harvest knowledge, not code: the legacy resample_to_standard_spacing called interpolate
with no limit and fabricated 3,268 RHOB values (contracts §1). Write that as a
regression test that must fail against a naive implementation.

Tests — tests/scenarios/test_scenario5_engine.py: child is a distinct dataset; parent
arrays unchanged; "original" always resolves to the parent; no child value exists over
a parent gap classified TOOL_ABSENCE_EVIDENCED or OUTSIDE_INTERVAL; every RESAMPLED
sample reproduces from its record alone; the KEOUGH RHOB fabrication regression.
````

### P24 — Stage 6: Derivation

**Needs:** P23, OD-16, OD-28.

````text
POLISH rebuild — P24: Stage 6 Derivation.

Implements: D55, D58, D40, D54, D57, D53 (as clarified by ⟦OD-16⟧).

1. Equation library (data, versioned): id, output family and unit, input families and
   units, formula, citation, source tier, validity conditions, whether an uncertainty
   is quantifiable and how. Entries: ⟦OD-28⟧.

2. polish/stages/s06_derive.py (alters_data=True, default OFF). Creates a NEW curve
   whose every sample is CREATED, each with a provenance record (equation id and
   version, input SampleRefs and values, parameters, uncertainty or
   NOT_QUANTIFIABLE_FOR_METHOD).

3. Inputs must be MEASURED_LOG curves that are ELIGIBLE and unit-resolved. Never a
   SYNTHETIC_SOURCE (D40, D58) — enforced structurally. A derived curve may feed
   another derivation only with an explicit recorded consent (D58).

4. Where a model is FITTED (not a fixed formula), MIN_POINTS applies per ⟦OD-16⟧;
   too few points → SCIENTIFIC_PRECONDITION_FAILURE, method not run, recorded.
   It never falls back to another method silently.

5. Any input sample that is INVALID or ABSENT → the output at that depth is ABSENT,
   recorded.

Tests: synthetic-source input rejected with no override path; derived-from-derived
without consent rejected; every output sample reproducible from its record alone;
fitted model with too few points declines.
````

### P25 — Stage 7: Gap filling

**Needs:** P24, OD-13, OD-14, OD-15, OD-16.

````text
POLISH rebuild — P25: Stage 7 Gap filling.

Implements: D49, D50, D51, D52 (with F7), D53 (with F8), D54, D56, D57, C2, UXC7,
supplement Part F scenarios 1 and 2.

1. polish/stages/s07_fill.py (alters_data=True, default OFF, consent PER GAP).
   A fill is offered only for gaps of class ⟦OD-13: e.g. DROPOUT, and AMBIGUOUS only
   with an explicit recorded choice⟧. OUTSIDE_INTERVAL and TOOL_ABSENCE_EVIDENCED have
   no fill code path at all (D50) — not a disabled path, no path.

2. Bounded interpolation needs exactly the two bracketing measured samples (F8);
   MIN_POINTS does not apply to it.

3. Domain from interpolation_domain (D42). LOG with a non-positive endpoint → the
   fill is refused, gap left open, gap_fill_outcome REFUSED with the reason (D52;
   scenario 2 copy from UXC7). Null domain → refused, reason recorded.

4. Any filled value that would violate a HARD or DOMAIN rule → the WHOLE fill is
   refused and the gap left open (D56). Never clipped.

5. Thresholds are depth-normalised (D51) from the rule library, in the curve's unit:
   ⟦OD-15⟧. Feet-only values never applied to metric curves.

6. Every filled interval emits a regeneration record (D54): span, depth interval,
   method, domain, exact inputs, parameters, per-sample uncertainty (or
   NOT_QUANTIFIABLE_FOR_METHOD, D57), selection criterion. Filled samples are CREATED,
   in the overlay (or in the active child dataset per OD-03).

7. Acceptance test from D54: a third party reproduces every value from the record
   alone — implement tests/stages/test_regeneration_reproducible.py with an
   independent evaluator that reads only the record.

Harvest: core/gap_filling.py (1,226 lines, little test coverage) — write
characterisation tests of its current behaviour FIRST, then port only logic that
satisfies the rules above. Do not port any path that fills without a limit, fills
tool-absence gaps, or clips. Investigate and report the legacy GR minimum dropping
from 4.03 to 0 across fill_gaps (contracts §9).

Tests: scenario 1 (no fill action exists for the tool-absence interval; the hole stays
visible; non-action recorded); scenario 2 (non-positive resistivity endpoint → refused,
distinct from an error, export not blocked, recorded); bound violation refuses the
whole fill; reproducibility test.
````

### P26 — Stage 8: Denoising and the difference view

**Needs:** P25. **Lowest priority** — your father smooths "almost never" (worksheet 4.3). Candidate to move after v1.

````text
POLISH rebuild — P26: Stage 8 Denoising and difference data.

Implements: D25, D45, D77 (positive framing only), brief §4.2 (changed), §4.5.

1. polish/stages/s08_denoise.py (alters_data=True, default OFF, last data-altering
   stage). Output samples are CHANGED: the original measured value is kept; the new
   value lives in the overlay with a provenance record (method, parameters, window,
   input SampleRefs).

2. Never denoise INVALID, ABSENT, CREATED or SYNTHETIC_SOURCE samples unless ⟦decide⟧.

3. Difference data: for every CHANGED sample, delta = new − original, and a per-curve
   summary (count changed, max |delta|, where).

4. Methods from the legacy core/denoising.py only after characterisation tests.
   Regression: the TBHV collapse to 3.1e-05 (contracts §9) — reproduce against legacy,
   confirm the new stage doesn't do it.

Tests: originals unchanged and retrievable; every CHANGED sample has both values and a
record; delta correct; TBHV regression.
````

### P27 — Interface for altered data and lineage

**Needs:** P26, OD-25 Q3.

````text
POLISH rebuild — P27: Rendering created/changed data and lineage in the interface.

Implements: brief §4.1, §4.2, §4.5, §4.6, §10 Q3; D45, D73, D74, D75, D76, D77, D78;
UXC6, UXC8.

1. Original and created/changed are drawn on the SAME track; created layered over
   original; original never removed (D76).

2. The overlay is ON when a file opens (D75). Exactly two view states: original-only,
   original-plus-overlay. There is no created-only view — add a test asserting no
   such mode, action or setting exists.

3. Encodings by zoom (D73, UXC6): marker shape at sample zoom (measured = filled
   circle); line style at mid zoom; hatched background band at full well (modified vs
   unmodified only). From the approved spec: ⟦OD-25⟧.

4. Two values at one depth for CHANGED samples: ⟦OD-25 Q3⟧. Must not look like a
   plotting error.

5. Difference view: plots delta for CHANGED samples (P26), on its own track, presented
   as "original versus modified", without claims about other products (F18).

6. Lineage (UXC8): breadcrumb showing active dataset, parent, grid relationship and
   ordered transformations; a dataset selector; source-versus-child comparison that
   never implies index alignment when grids differ.

7. Hover and record (P17) now show CREATED / CHANGED / RESAMPLED detail, including
   NOT_QUANTIFIABLE_FOR_METHOD where applicable.

Tests: overlay on at open; no created-only mode; a 3-sample fill visible at every zoom;
two-value rendering distinguishable in greyscale; breadcrumb correct after resample +
fill; scenario 5 end to end in the UI.
````

---

## Export and release (drafts — re-check before sending)

### P28 — Stage 9: Manifest and status code table

**Needs:** P27, OD-21, OD-04 final.

````text
POLISH rebuild — P28: Stage 9 Manifest and status curves.

Implements: D54, D61, D22, contracts §5, UXC10 (provenance artifacts), brief §3.

1. Full manifest: everything in the describe report (P16) plus, per curve: final
   unit and every unit re-expression; sample counts by status; measured interval vs
   spanned interval; rules applied with versions; every resolution point hit, what was
   chosen, by whom; every regeneration record; every refusal; every deferral with its
   reason; lineage. Human-readable
   and JSON, with a versioned JSON Schema.

2. Status code table (D61): integer code ↔ sample_status value, fixed, append-only,
   versioned, written into every export. Commit tests/provenance/status_codes_snapshot.json;
   a test fails if any existing code's meaning changes or a code is removed (D61).

3. Per-sample method detail in the file: ⟦OD-21 — domain expert wants it only when
   asked for; this conflicts with D61. Your decision; see plan 1.0.3⟧.

4. Narrow P14's "affects exported claims" definition to what the manifest actually
   asserts, and update the test.

Tests: schema validation on every fixture after a full synthetic run; snapshot
immutability; refusals present and as prominent as actions in the human rendering.
````

### P29 — Stage 10: Export profiles and round-trip

**Needs:** P28, OD-18, OD-24.

````text
POLISH rebuild — P29: Stage 10 Export with three profiles, and round-trip invariants.

Implements: D17, D18, D19, D59, D60, D62, D63, D85, D86 (export conformance), D9 per
⟦OD-18⟧, D68, UXC10, supplement Part F scenario 6.

1. Profiles (D85):
   a) Original input — never opened for writing. Not "exported"; referenced by hash.
   b) Auditable processed — original mnemonics and curve order preserved (D62);
      POLISH's interpretation in annotations and the manifest; DEPT copy: ⟦OD-18⟧.
   c) Interoperability — explicit selection only; canonical aliases added as ADDITIONAL
      curves with a complete mapping record; never replaces a source mnemonic.
   The profile is chosen per job (domain expert, 14 Sept: "I'd ask the client").

2. Header generated from final state (D59), never inherited and patched. The
   preserved-verbatim fields are an explicit list in a data file: ⟦list⟧.

3. Written into the processed file: the status code legend (D18, D61), status curves
   per OD-21, the input SHA-256 (D63), the POLISH and rule-library versions, and a
   "User-mapped — not identified by POLISH" annotation for every user-mapped curve.

4. Export conformance (LAS 2.0, D68) checked before writing, against the matrix from
   ⟦OD-24: CWLS document⟧. Any failure → EXPORT_CONFORMANCE_FAILURE; nothing written.

5. Round-trip invariants (D60), as tests on every fixture and every profile:
   re-loading a POLISH-written file triggers no refusal; re-importing rebuilds an
   identical status mask.

6. Deliberate write per P22's flow. Input path blocked (D19).

Tests: original file hash unchanged; auditable profile preserves every source
mnemonic and order; interoperability profile only adds curves; round-trip invariants;
scenario 6 end to end; no force path.
````

### P30 — Offline licensing

**Needs:** OD-23.

````text
POLISH rebuild — P30: Offline licence verification.

Implements: D14, D15 (with F11).

1. A signed licence file verified offline with a public key embedded in the binary.
   Signature scheme: ⟦e.g. Ed25519 via the `cryptography` package — list licence and
   version⟧. The private key never enters this repository; add a test that scans the
   repo for private-key markers.

2. Clock-rollback guard (F11): store the last-seen trusted timestamp locally; if the
   system clock is earlier, refuse with a clear message.

3. What an invalid, missing or expired licence permits: ⟦OD-23⟧.

4. No network call of any kind (the P06 guards must keep passing). The legacy
   Render/Firebase licence server is not used.

Tests: valid, tampered, expired, missing, clock-rolled-back; no network; key-marker scan.
````

### P31 — Packaging and the cohesion scenario suite

**Needs:** P30.

````text
POLISH rebuild — P31: Packaging and end-to-end cohesion scenarios.

Implements: D14, D1 (PyInstaller constraints), supplement Part F scenarios 1–6.

1. PyInstaller spec committed to the repo; one-folder build first. Report size and
   start time. dlisio remains optional (imported inside a function).

2. Scenario suite in tests/scenarios/: scenarios 1–6 end to end through the interface
   where automatable; a written manual checklist in docs/release/manual_checks.md for
   what can't be automated (greyscale print of a full-well screenshot; keyboard-only
   afternoon session on the LFP fixture).

3. Isolation check for the built application: ⟦choose — e.g. a manual procedure run on
   a machine with networking disabled, plus a static scan of the bundle for network
   libraries⟧. Report what the check can and cannot prove.

4. Release checklist: all OD items relevant to this release closed; register updated;
   contracts document updated; LGPL obligations for the toolkit reviewed (a human task
   — list it, don't perform it).
````

---

# APPENDIX A — Proposed state schema (for OD-04)

**Authority: DESIGN tier — proposed by Claude, lowest authority in the register's own terms.** Approve, edit or reject each table before P05.

### A.1 Fields and values

| Field | Applies to | Values | Rows |
|---|---|---|---|
| `curve_role` | curve | INDEX · MEASURED_LOG · DRILLING · ANCILLARY · SYNTHETIC_SOURCE · GEOMETRIC_REFERENCE *(proposed name for TVD — OD-17)* · UNKNOWN | D32, D40, F2 |
| `evidence_tier` | curve identification | CITED · OBSERVED · USER_DEFINED · PROPOSED · NO_MATCH | D33 |
| `processing_eligibility` | curve | ELIGIBLE · NOT_ELIGIBLE_NO_FAMILY · NOT_ELIGIBLE_LOW_TIER · NOT_ELIGIBLE_UNIT_UNRESOLVED · NOT_ELIGIBLE_ROLE | D34, D11/F15 |
| `unit_interpretation` | curve | RECOGNISED · ABSENT · EMPTY · UNRECOGNISED · INCOMPATIBLE_WITH_FAMILY | C1, D87 |
| `validation_outcome` | curve | PASSED · PLAUSIBILITY_WARNING · HARD_VIOLATION_EXCLUDED · NOT_ELIGIBLE · PENDING · NOT_APPLICABLE | UXC9 |
| `sample_status` | sample | MEASURED · INVALID · ABSENT · CREATED · CHANGED · RESAMPLED | D44 — **no REFUSED**. INVALID = outside a confirmed hard limit; *whether* it is excluded from the processed file is a separate recorded decision (dad, 1.2) |
| `output_inclusion` | invalid sample | PENDING_DECISION · EXCLUDED · KEPT | Dad 1.2; D17 (original never altered) |
| `gap_class` | gap | OUTSIDE_INTERVAL · TOOL_ABSENCE_EVIDENCED · DROPOUT · AMBIGUOUS (+ evidence, confidence) | D49 |
| `gap_fill_outcome` | gap | NOT_REQUESTED · FILLED · REFUSED (+ reason) · NOT_FILLABLE (structural class) | D50, D52, D56 |
| `method_outcome` | method attempt (Stages 5–8) | RUN · NOT_RUN_PRECONDITION · NOT_RUN_NOT_REQUESTED | D86, UXC7 |
| `block_type` | finding | INFORMATIONAL · AMBIGUOUS_INTERPRETATION · RULE_DEPENDENT_WARNING · SCIENTIFIC_PRECONDITION_FAILURE · STRUCTURAL_DATA_FAILURE · EXPORT_CONFORMANCE_FAILURE | D86 |
| `issue_disposition` | finding | UNDISPOSITIONED *(non-terminal)* · RESOLVED · ACCEPTED · DEFERRED (+ reason) · DECLINED_BY_POLISH | D79 |
| `dependency_state` | result, decision | CURRENT · STALE (+ cause) | D80 |
| `stage_status` | stage | NOT_RUN · RUNNING · COMPLETE · STALE · BLOCKED · REFUSED_STRUCTURAL | D80, D86 |

**Approved 15 Sept, with one addition:**

1. **Tier → eligibility (D34).** CITED, OBSERVED and USER_DEFINED are eligible; PROPOSED and NO_MATCH are not. **Addition:** a USER_DEFINED curve is always labelled "User-mapped — not identified by POLISH" wherever it appears, including the export.
2. **`DECLINED_BY_POLISH`** replaces D79's bare "Refused" in `issue_disposition`.

### A.2 How "refused" resolves under this schema

| Brief / register usage | Where it lives now |
|---|---|
| Brief §4.2 "Refused — a gap POLISH declined to fill" (display state) | Samples are ABSENT; the **gap** carries `gap_fill_outcome = REFUSED` + reason. Hover reads the gap record. |
| D79 "Refused — POLISH declined, reason recorded" (disposition) | `issue_disposition = DECLINED_BY_POLISH` |
| D86 "scientific precondition failure — method declined" | `method_outcome = NOT_RUN_PRECONDITION`, finding `block_type = SCIENTIFIC_PRECONDITION_FAILURE` |
| Brief §8 "refuse" as a keyboard verb (user action) | `REJECT_PROPOSAL` → disposition RESOLVED, with the rejection recorded as the choice |

### A.3 Block type → permitted user actions → export effect

| Block type | User actions | Resulting disposition | Export effect |
|---|---|---|---|
| INFORMATIONAL | ACKNOWLEDGE · DEFER (reason) | ACCEPTED · DEFERRED | Blocks only until dispositioned |
| AMBIGUOUS_INTERPRETATION | CHOOSE_OPTION (rationale) · REJECT_PROPOSAL (rationale) · DEFER (reason) | RESOLVED · RESOLVED · DEFERRED | Blocks until dispositioned; deferrals listed in the export |
| RULE_DEPENDENT_WARNING | MAP_EXPLICITLY (label) · ACKNOWLEDGE (processing stays ineligible) · DEFER (reason) | RESOLVED · ACCEPTED · DEFERRED | Blocks until dispositioned |
| SCIENTIFIC_PRECONDITION_FAILURE | view only | DECLINED_BY_POLISH, set automatically | Doesn't block (UXC7 example); recorded in refusals |
| STRUCTURAL_DATA_FAILURE | view only | none — stays open | Blocks export; workflow stops at that stage |
| EXPORT_CONFORMANCE_FAILURE | fix upstream only | none | Blocks until the condition no longer holds |

**Deferral (decided 15 Sept):** DEFERRED is terminal and does not block export. Every deferral and its reason is listed prominently in the export.

---

# APPENDIX B — Proposed keyboard map (for P18)

**DESIGN tier — a starting point, not a decision.** Letter keys act only while focus is in the queue. No key triggers export.

| Key | Action |
|---|---|
| J / ↓ | Next queue item |
| K / ↑ | Previous queue item |
| Enter | Open the item's record |
| A | Acknowledge (informational, rule-dependent) |
| 1–9 | Choose option n (ambiguity) |
| X | Reject POLISH's proposal (ambiguity) |
| M | Map explicitly (rule-dependent) — inline field |
| D | Defer — inline reason field |
| B | Open batch review for this item's group |
| Esc | Close an inline field without recording anything |
| O | Toggle overlay (original-only ↔ original-plus-overlay) |
| Z | Full-well view |
| G | Go to the current item's depth |
| + / − | Zoom |
| [ / ] | Previous / next stage in the rail |
| ? | Show this keymap |

---

*End of build plan.*
