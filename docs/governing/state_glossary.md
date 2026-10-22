# State glossary

OD-04, Appendix A, approved 15 Sept. This file names each value, the rows that define it, and the user-facing wording where UXC2 or UXC9 gives one. It does not add values.

A `USER_DEFINED` curve carries this label wherever it appears, including the export: "User-mapped — not identified by POLISH".

Unit conversion is a recorded re-expression (OD-05). It never produces a `CHANGED` sample. Stored values stay as they are; the unit travels with the value.

CITED, OBSERVED and USER_DEFINED evidence are eligible for processing. PROPOSED and NO_MATCH are not (D34). Eligibility can still fail for family, unit, or role.

Where the UXC2 and UXC9 language tables are silent, the wording cell says so. No display sentence is invented here.

## Where "refused" is recorded

Appendix A.2. It is not a sample status.

| Brief / register usage | Where it lives now |
|---|---|
| Brief §4.2 "Refused — a gap POLISH declined to fill" (display state) | Samples are `ABSENT`; the gap carries `gap_fill_outcome = REFUSED` plus a reason. Hover reads the gap record. |
| D79 "Refused — POLISH declined, reason recorded" (disposition) | `issue_disposition = DECLINED_BY_POLISH` |
| D86 "scientific precondition failure — method declined" | `method_outcome = NOT_RUN_PRECONDITION`, finding `block_type = SCIENTIFIC_PRECONDITION_FAILURE` |
| Brief §8 "refuse" as a keyboard verb (user action) | `REJECT_PROPOSAL` → disposition `RESOLVED`, with the rejection recorded as the choice |

## curve_role

Applies to a curve. Rows: D32, D40, F2.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `INDEX` | The depth index of the file. | D32 | Not specified in the UXC2 or UXC9 language tables. |
| `MEASURED_LOG` | A measured log curve. | D32 | Not specified in the UXC2 or UXC9 language tables. |
| `DRILLING` | A drilling measurement carried in the file. | D32 | Not specified in the UXC2 or UXC9 language tables. |
| `ANCILLARY` | Coordinates, facies labels, or QC flags: loaded and displayed, excluded from processing. | D40 | Not specified in the UXC2 or UXC9 language tables. |
| `SYNTHETIC_SOURCE` | Vendor-computed (`*_SYNT`): never treated as a measurement, never a derivation input. | D40 | Not specified in the UXC2 or UXC9 language tables. |
| `GEOMETRIC_REFERENCE` | Computed geometric reference that must track the index. The TVD role. Name decided by the owner 9 Oct 2026 (OD-17). | D32, F2, OD-17 | Not specified in the UXC2 or UXC9 language tables. |
| `UNKNOWN` | The role is not established. | D32 | Not specified in the UXC2 or UXC9 language tables. |

## evidence_tier

Applies to curve identification. Rows: D33. Appendix A adds `NO_MATCH` to D33's four tiers.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `CITED` | The identification is cited from a published or shipped mnemonic source. | D33 | Not specified in the UXC2 or UXC9 language tables. |
| `OBSERVED` | The identification was observed from the file, not only cited. | D33 | Not specified in the UXC2 or UXC9 language tables. |
| `USER_DEFINED` | The user mapped this curve. Eligible for processing, and always labelled as user-mapped. | D33, D34 | User-mapped — not identified by POLISH |
| `PROPOSED` | POLISH proposed the identification and it is not yet a user mapping. Not eligible for processing. | D33, D34 | Not specified in the UXC2 or UXC9 language tables. |
| `NO_MATCH` | No identification matched. Not eligible for processing. | D33, Appendix A | Not specified in the UXC2 or UXC9 language tables. |

## processing_eligibility

Applies to a curve. Rows: D34, D11, F15.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `ELIGIBLE` | The curve may enter automatic processing. | D34 | Not specified in the UXC2 or UXC9 language tables. |
| `NOT_ELIGIBLE_NO_FAMILY` | No family is established, so automatic range validation does not run. | D34, D11, F15 | Not specified in the UXC2 or UXC9 language tables. |
| `NOT_ELIGIBLE_LOW_TIER` | The evidence tier is PROPOSED or NO_MATCH, so it does not unlock range validation or gap filling. | D34 | Not specified in the UXC2 or UXC9 language tables. |
| `NOT_ELIGIBLE_UNIT_UNRESOLVED` | The unit is not resolved, so a range check would be a comparison without a unit. | D34, D11, F15, C1 | Not specified in the UXC2 or UXC9 language tables. |
| `NOT_ELIGIBLE_ROLE` | The role excludes the curve from processing (ancillary, synthetic source, or unknown). | D34, D40, C3 | Not specified in the UXC2 or UXC9 language tables. |

## unit_interpretation

Applies to a curve. Rows: C1, D87.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `RECOGNISED` | The declared unit is a known unit for this curve. | C1, D87 | Not specified in the UXC2 or UXC9 language tables. |
| `ABSENT` | The file declares no unit. | C1, D87 | Not specified in the UXC2 or UXC9 language tables. |
| `EMPTY` | A unit field is present and empty. | C1, D87 | Not specified in the UXC2 or UXC9 language tables. |
| `UNRECOGNISED` | A unit text is present and is not a known unit. | C1, D87 | Not specified in the UXC2 or UXC9 language tables. |
| `INCOMPATIBLE_WITH_FAMILY` | The unit is recognised and its dimension is not one of the family's accepted dimensions. | C1, D87, D90 | Not specified in the UXC2 or UXC9 language tables. |

## validation_outcome

Applies to a curve. Rows: UXC9. Wording is the UXC9 status language table.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `PASSED` | A rule was applied and no violation was found. | UXC9 | Validated against rule set vX.Y |
| `PLAUSIBILITY_WARNING` | A rule was applied and a warning-level violation was found. | UXC9 | Plausibility warning |
| `HARD_VIOLATION_EXCLUDED` | A rule was applied and a hard violation was found; the measured value is excluded from processing. | UXC9 | Measured value excluded from processing |
| `NOT_ELIGIBLE` | No eligible rule was applied, because family or unit confidence is insufficient. This is not a pass. | UXC9, D11, D87 | Not automatically validated |
| `PENDING` | Validation has not been run. | UXC9 | Validation pending |
| `NOT_APPLICABLE` | The validation method is blocked by a scientific precondition. | UXC9 | Validation method not applicable |

## sample_status

Applies to a sample. Rows: D44. There is no `REFUSED` value.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `MEASURED` | The sample is the measured value. | D44 | Not specified in the UXC2 or UXC9 language tables. |
| `INVALID` | The value is outside a confirmed hard limit. Exclusion from the processed file is a separate `output_inclusion` decision. | D44 | Not specified in the UXC2 or UXC9 language tables. |
| `ABSENT` | No value is present at this sample, including samples in a gap POLISH declined to fill. | D44, Appendix A.2 | Not specified in the UXC2 or UXC9 language tables. |
| `CREATED` | POLISH created this value, and a provenance record exists for it. | D44 | Not specified in the UXC2 or UXC9 language tables. |
| `CHANGED` | POLISH changed the stored value. Unit conversion is not this status (OD-05). | D44, OD-05 | Not specified in the UXC2 or UXC9 language tables. |
| `RESAMPLED` | The sample belongs to a resampled child dataset. | D44 | Not specified in the UXC2 or UXC9 language tables. |

## output_inclusion

Applies to an invalid sample, and to a measured sample outside a plausibility limit (D92). Rows: D17, D44, D92 (dad, 1.2). The original file is never opened for writing.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `PENDING_DECISION` | The sample has no recorded include-or-exclude decision yet. | D44, D92 | Not specified in the UXC2 or UXC9 language tables. |
| `EXCLUDED` | The sample is left out of the processed file and not used by later stages. It stays visible and marked. The original file is unchanged. | D17, D44, D92 | Not specified in the UXC2 or UXC9 language tables. |
| `KEPT` | A recorded decision keeps the sample in the processed file; a kept suspicious sample is used by later stages and stays flagged. | D44, D92 | Not specified in the UXC2 or UXC9 language tables. |

## gap_class

Applies to a gap. Rows: D49, F6. The record also carries evidence and a confidence. No numeric confidence scale is approved, so both are text supplied with the classification.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `OUTSIDE_INTERVAL` | The gap lies outside the measured interval. | D49, F6 | Not specified in the UXC2 or UXC9 language tables. |
| `TOOL_ABSENCE_EVIDENCED` | An interior gap is shared across curves from the same tool. That is evidence consistent with tool absence, not proof. | D49, F6 | Not specified in the UXC2 or UXC9 language tables. |
| `DROPOUT` | An interior gap is shorter than the tool's vertical resolution. | D49, F6 | Not specified in the UXC2 or UXC9 language tables. |
| `AMBIGUOUS` | The gap is none of the classes above. Only this class needs a threshold, and the evidence and confidence are part of the record. | D49, F6 | Not specified in the UXC2 or UXC9 language tables. |

## gap_fill_outcome

Applies to a gap. Rows: D50, D52, D56.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `NOT_REQUESTED` | Filling was not requested. | D50 | Not specified in the UXC2 or UXC9 language tables. |
| `FILLED` | The gap was filled. | D50 | Not specified in the UXC2 or UXC9 language tables. |
| `REFUSED` | POLISH declined to fill. A non-empty reason is required. Samples stay `ABSENT`. | D52, D56, Appendix A.2 | Not specified in the UXC2 or UXC9 language tables. |
| `NOT_FILLABLE` | The class is structurally not fillable. Tool absence is never fillable, and that is not an overridable flag. | D50 | Not specified in the UXC2 or UXC9 language tables. |

## method_outcome

Applies to one method attempt in Stages 5–8. Rows: D86, UXC7.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `RUN` | The method ran. | D86 | Not specified in the UXC2 or UXC9 language tables. |
| `NOT_RUN_PRECONDITION` | The method did not run because a scientific precondition failed. | D86, UXC7 | This method was not run |
| `NOT_RUN_NOT_REQUESTED` | The method did not run because it was not requested. | D86 | Not specified in the UXC2 or UXC9 language tables. |

"This method was not run" is the UXC2 wording for a scientific precondition failure. It is the display sentence for `NOT_RUN_PRECONDITION`, not for `NOT_RUN_NOT_REQUESTED`.

## block_type

Applies to a finding. Rows: D86, UXC2. Wording is the UXC2 language table. The Avoid column is part of that table.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `INFORMATIONAL` | A discrepancy that is recorded. It blocks export only until it is dispositioned. | D86, D79 | Recorded — no action required. Avoid "Warning" if nothing is wrong. |
| `AMBIGUOUS_INTERPRETATION` | An ambiguity that needs a recorded choice. It blocks export until it is dispositioned. A deferral is listed in the export. | D86, D79 | Your decision is required. Avoid "Error". |
| `RULE_DEPENDENT_WARNING` | Processing stays ineligible until the user maps the curve explicitly, acknowledges that, or defers. | D86 | Map explicitly to continue this analysis. Avoid "Auto-fix". |
| `SCIENTIFIC_PRECONDITION_FAILURE` | The method was declined. Disposition is `DECLINED_BY_POLISH`, set automatically. Export is not blocked. | D86, UXC7 | This method was not run. Avoid "Failed" or "try again". |
| `STRUCTURAL_DATA_FAILURE` | The file or the workflow cannot continue. No disposition clears it. Export stays blocked. | D86 | This file cannot proceed past Depth Axis. Avoid "Accept anyway". |
| `EXPORT_CONFORMANCE_FAILURE` | The export itself does not conform. It is fixed upstream or it is not exported. | D86 | Export is blocked until this is corrected. Avoid "Force export". |

UXC2's structural sentence names Depth Axis because that is the worked example of a structural stop. The export effect for every structural finding is the same: export is blocked and the workflow stops at that stage.

## issue_disposition

Applies to a finding. Rows: D79, D86.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `UNDISPOSITIONED` | Non-terminal. The finding has no disposition yet. Silence is not a completed state. | D79 | Not specified in the UXC2 or UXC9 language tables. |
| `RESOLVED` | Terminal. The user chose an option, rejected a proposal, or mapped explicitly. | D79, Appendix A.3 | Not specified in the UXC2 or UXC9 language tables. |
| `ACCEPTED` | Terminal. The user acknowledged an informational finding, or acknowledged a rule-dependent warning and left processing ineligible. | D79, Appendix A.3 | Not specified in the UXC2 or UXC9 language tables. |
| `DEFERRED` | Terminal. Does not block export. A non-empty reason is required, and the deferral is listed in the export. | D79, Appendix A.3 | Not specified in the UXC2 or UXC9 language tables. |
| `DECLINED_BY_POLISH` | Terminal. POLISH declined. Replaces D79's bare "Refused". Set automatically on a scientific precondition failure. | D79, D86 | Not specified in the UXC2 or UXC9 language tables. |

## dependency_state

Applies to a result or a decision. Rows: D80. STALE carries a cause.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `CURRENT` | The result or decision still matches its inputs. | D80 | Not specified in the UXC2 or UXC9 language tables. |
| `STALE` | An upstream change has marked this result or decision non-current. The cause is recorded. Export is blocked when this dependency affects export. | D80 | Not specified in the UXC2 or UXC9 language tables. |

Which nodes affect exported claims is not decided in this schema.

## stage_status

Applies to a stage. Rows: D80, D86.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `NOT_RUN` | The stage has not been run. | D80, D86 | Not specified in the UXC2 or UXC9 language tables. |
| `RUNNING` | The stage is running. | D80, D86 | Not specified in the UXC2 or UXC9 language tables. |
| `COMPLETE` | The stage has completed. | D80, D86 | Not specified in the UXC2 or UXC9 language tables. |
| `STALE` | An upstream change has marked the stage non-current. | D80, D86 | Not specified in the UXC2 or UXC9 language tables. |
| `BLOCKED` | The stage is blocked. | D80, D86 | Not specified in the UXC2 or UXC9 language tables. |
| `REFUSED_STRUCTURAL` | The stage stopped on a structural data failure. | D80, D86 | This file cannot proceed past Depth Axis |

## user_action

Actions a person may take. Rows: D79, D86, UXC2. Which of these a block type permits is the matrix below.

| Value | Meaning | Rows | Preferred user-facing wording |
|---|---|---|---|
| `ACKNOWLEDGE` | Accept an informational finding, or accept a rule-dependent warning while processing stays ineligible. | D86, Appendix A.3 | Not specified in the UXC2 or UXC9 language tables. |
| `DEFER` | Defer the finding with a reason. Terminal, and listed in the export. | D79, Appendix A.3 | Not specified in the UXC2 or UXC9 language tables. |
| `CHOOSE_OPTION` | Choose among interpretations. A rationale is recorded. Yields `RESOLVED`. | D86, Appendix A.3 | Not specified in the UXC2 or UXC9 language tables. |
| `REJECT_PROPOSAL` | Reject POLISH's proposal. The rejection is the recorded choice. Yields `RESOLVED`. | Appendix A.2, D79 | Not specified in the UXC2 or UXC9 language tables. |
| `MAP_EXPLICITLY` | Map the curve explicitly: choose the family it behaves as (required) and, optionally, your own name for it. Yields `RESOLVED`. | D86, D91, Appendix A.3 | Map explicitly to continue this analysis |

## Block type, actions, disposition, export

Appendix A.3. Deferral, decided 15 Sept: `DEFERRED` is terminal and does not block export. Every deferral and its reason is listed prominently in the export.

There is no action, parameter, or flag that overrides a structural or conformance block.

| Block type | User actions | Resulting disposition | Export effect | Preferred UI language (UXC2) |
|---|---|---|---|---|
| `INFORMATIONAL` | `ACKNOWLEDGE` · `DEFER` (reason) | `ACCEPTED` · `DEFERRED` | Blocks only until dispositioned | Recorded — no action required |
| `AMBIGUOUS_INTERPRETATION` | `CHOOSE_OPTION` (rationale) · `REJECT_PROPOSAL` (rationale) · `DEFER` (reason) | `RESOLVED` · `RESOLVED` · `DEFERRED` | Blocks until dispositioned; deferrals listed in the export | Your decision is required |
| `RULE_DEPENDENT_WARNING` | `MAP_EXPLICITLY` (label) · `ACKNOWLEDGE` (processing stays ineligible) · `DEFER` (reason) | `RESOLVED` · `ACCEPTED` · `DEFERRED` | Blocks until dispositioned | Map explicitly to continue this analysis |
| `SCIENTIFIC_PRECONDITION_FAILURE` | view only | `DECLINED_BY_POLISH`, set automatically | Does not block export; recorded in refusals | This method was not run |
| `STRUCTURAL_DATA_FAILURE` | view only | none — stays open | Blocks export; workflow stops at that stage | This file cannot proceed past Depth Axis |
| `EXPORT_CONFORMANCE_FAILURE` | fix upstream only | none | Blocks until the condition no longer holds | Export is blocked until this is corrected |
