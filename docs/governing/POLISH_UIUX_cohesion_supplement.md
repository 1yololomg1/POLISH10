POLISH — UI/UX Cohesion Supplement
Purpose: Define the UI/UX responsibilities needed to make POLISH’s honesty, provenance, refusal, and auditability claims coherent in the user’s actual workflow.

Relationship to existing documents:

Document	Governs	This supplement does
Design Brief	Product intent, interface principles, fixed pipeline, open interface questions	Converts the interface principles into explicit cohesion requirements
Decision Register	Decisions, findings, conflicts, provenance, review status	Adds UI/UX-specific decisions and traces their backend dependencies
Pipeline contracts	Data and behavioral invariants between stages	Must eventually define the implementation contracts that make these UI requirements truthful
This supplement	Cross-layer UX behavior	Defines what users must be able to see, understand, decide, inspect, and export safely
Core rule:

UI/UX is not a cosmetic layer over POLISH. It is the user-facing verification surface for a scientific data-processing system. It must never make a state look simpler, safer, more measured, or more certain than the underlying data model supports.

1. The big-picture model
POLISH’s plan has five connected layers. UI/UX owns one layer directly, influences two heavily, and depends on two that it cannot repair.

System layer	Main responsibility	UI/UX authority	UI/UX obligation
Scientific/domain rules	Gap classification, interpolation eligibility, physical limits, unit semantics, derivation preconditions	None	Do not imply certainty, eligibility, or safety that the rules have not established
Pipeline and data model	Dataset identity, source/child lineage, status records, state transitions, stale dependencies	None	Make lineage, state, dependency, and consequences intelligible
Audit/provenance	Regeneration records, source inputs, transformation history, rule versions, refusal reasons	Shared	Make evidence inspectable at appropriate levels of detail
Interaction design	Coverage track, queue, curve selection, scope controls, keyboard workflow, state hierarchy	Primary	Make serious review work usable, efficient, stable, and non-misleading
Export and re-import	Profile selection, blockers, manifest/status output, round-trip invariants	Shared	Make export consequences visible and require deliberate confirmation
The user should experience these layers as one coherent tool:

Load
→
Describe
→
Inspect
→
Decide
→
Alter with consent
→
Inspect again
→
Export deliberately
Load→Describe→Inspect→Decide→Alter with consent→Inspect again→Export deliberately
The backend performs the scientific and data-model work. The UI makes the result reviewable without converting the workflow into a maze of dialogs or a settings panel.

Part A — UI/UX Cohesion Worksheet
Use this worksheet before designing a screen, interaction, or feature. Each row asks whether the UI is merely displaying a fact, helping the user make a legitimate decision, or accidentally trying to cover for an unresolved backend problem.

A1. Screen-level worksheet
Question	Required answer before UI design	Example of a good answer	Red flag
What underlying state does this screen represent?	Name the canonical backend fields, not a vague label	sample_status=created, validation_outcome=not_assessed, issue_disposition=accepted, dependency_state=current	“This is a flagged curve” when “flagged” could mean many things
Who owns the truth?	Identify pipeline stage, rule library, or user decision source	Stage 4 determined the gap inventory; user accepted a mapping; D87 rule library supplied the unit/bound rule	UI infers a scientific status from visual appearance
What can the user change?	Identify reversible view controls versus recorded analytical decisions	Zoom is reversible; curve mapping is recorded; export profile is deliberate	A display toggle silently changes eligibility or processing
What cannot the user override?	Name block type and reason	Structural depth failure; scientific precondition failure; export conformance failure	An “accept anyway” button exists merely to clear a blocker
What evidence explains the state?	Define the minimum visible evidence and the path to full record	Gap span, co-tool evidence, confidence, rule version, input endpoints	A red warning icon with no inspectable reason
What changes if the user acts?	State downstream impact and stale dependencies	Changing units marks Stage 4 and later dependent decisions stale	UI gives a false sense of local/reversible change
What survives export?	Specify export profile and provenance output	Source mnemonic, family annotation, status code, rule version, regeneration record	UI implies an exported value is self-explanatory when provenance is lost
What does “done” mean?	Define completion/blocked/stale/dispositioned in backend terms	All reviewable items dispositioned; no non-overridable blocks; no stale dependencies	Green “complete” badge while export would be unsafe or blocked
A2. Cross-layer truth test
For every proposed UI control, ask these six questions.

Does the control alter data, alter interpretation, or alter only presentation?

Data-altering actions belong to consented stages and must produce provenance.

Interpretation-altering actions—such as mapping a mnemonic or accepting a unit—must be recorded with scope and rationale where appropriate.

Presentation-only actions—such as zoom, selection, track order, or overlay visibility—must not change analytical results.

Does the system have the backend state needed to support this control truthfully?
A “show original” control is not trustworthy if resampling has destroyed the stable relationship between source and child sample indices.

Can a user tell whether the control is local, grouped, file-specific, or persistent?
Scope must never be inferred silently.

Can the user inspect the evidence behind a consequence?
If the UI says “not eligible,” “invalid,” “refused,” or “stale,” the user needs a path to the reason.

Can a later reader understand the decision from the export or session record?
If not, the interface created an un-auditable hidden state.

Would this screen still be truthful at full-well scale, 171 curves, and after an upstream change?
If not, it is not yet a POLISH screen.

A3. UX classification worksheet
Use this to classify a proposed requirement before adding it to a mockup.

Classification	Meaning	Examples in POLISH	UI action
UI-owned	The interface can decide and implement the behavior itself	Queue layout, keyboard traversal, curve-selection workflow, non-reflowing panels	Design and test directly
Shared UX/backend	Backend establishes facts; UI makes them understandable and actionable	Staleness, disposition gating, rule eligibility, export blocks, dataset lineage	Define a joint contract before mockups harden
Backend/domain-owned	Requires data-model, algorithmic, or scientific-rule resolution	Resampling lineage, physical limits, tool grouping, interpolation domain, LAS parse correctness	Do not “solve” through visual design; write dependency requirement
Deferred/open	Valid need, but no settled policy exists	Scope persistence rules, tolerance values, some physical-limit categories, alternate workflow presets	Design the decision point, not a false default
Part B — UI/UX Decision Register Addendum
The following entries are proposed as UXC decisions: UI/UX cohesion decisions. They should be added to the decision register only after review. They do not replace D-number decisions already present; they make their user-facing implications explicit.

UXC1 — UI state must derive from orthogonal operational fields
Field	Proposed decision
ID	UXC1
Decision	No UI label, badge, color, or queue category may rely on the ambiguous umbrella terms “validated,” “flagged,” or “unknown” alone. Every displayed state must derive from explicit operational fields: curve role, identity evidence tier, processing eligibility, sample status, validation outcome, issue disposition, dependency state, and block type.
Rationale	D21 identifies that trust state, sample status, and curve role were conflated. The UI will otherwise make false or contradictory claims even if each backend stage works independently. 
UI effect	Use plain-language composites: “Measured — excluded from processing: hard-limit violation,” “Unknown family — preserved, not eligible for automatic validation,” “Decision accepted — stale after unit change.”
Backend dependency	Canonical state schema and transition rules
Status	Proposed; architecture-dependent
Priority	Must settle before detailed queue, legend, hover, and export-review design
Acceptance test
A user can distinguish, without opening raw data:

A measured-but-invalid sample.

A created sample that passed a fill precondition.

An unknown curve that was intentionally preserved.

A curve with low identification confidence.

A decision that was once accepted but is now stale.

A method refused because a scientific precondition failed.

UXC2 — Every finding must reveal its block type and user authority
Field	Proposed decision
ID	UXC2
Decision	Every surfaced finding must visibly state whether it is informational, an ambiguity requiring a recorded choice, a rule-dependent warning, a scientific precondition failure, a structural data failure, or an export conformance failure. The UI must expose only the actions permitted by that block type.
Rationale	D79’s disposition model is corrected by D86: not every finding is dispositionable, and no general “force export” override may exist. 
UI effect	Different queue and detail states for “choose,” “acknowledge,” “map explicitly,” “method declined,” “workflow stopped,” and “fix before export.”
Backend dependency	D86 block-type classification must be returned with every finding
Status	Proposed; shared UX/backend
Priority	Must settle before export flow and resolution queue implementation
Required user-facing language
Block type	Preferred UI language	Avoid
Informational discrepancy	“Recorded — no action required”	“Warning” if nothing is wrong
Ambiguous interpretation	“Your decision is required”	“Error”
Rule-dependent warning	“Map explicitly to continue this analysis”	“Auto-fix”
Scientific precondition failure	“This method was not run”	“Failed” or “try again”
Structural data failure	“This file cannot proceed past Depth Axis”	“Accept anyway”
Export conformance failure	“Export is blocked until this is corrected”	“Force export”
UXC3 — Staleness must be visible globally, locally, and causally
Field	Proposed decision
ID	UXC3
Decision	When an upstream decision changes, the UI must show stale state in three places: the stage rail, every directly affected queue item/view, and a persistent impact summary that identifies the triggering change and the required recovery path.
Rationale	The brief requires stale as a blocking fifth state and rejects silent internal inconsistency. 
UI effect	A user can see what changed, what is affected, whether a rerun is required, and whether prior dispositions must be revisited.
Backend dependency	Dependency graph and stale-propagation engine
Status	Proposed; shared UX/backend
Priority	Must settle before any editable upstream decision is exposed
Example UI copy
Stage 3 Units changed
Validity assessment, gap classification, and 14 related decisions are now stale.
No data has been altered. Export remains blocked until these items are rerun or re-dispositioned.

This language distinguishes a legitimate state transition from a software error.

UXC4 — Scope must be explicit whenever a decision can propagate
Field	Proposed decision
ID	UXC4
Decision	Every decision capable of applying beyond one item must require an explicit scope selection and show a human-readable preview of the affected population before commitment. Scope options must be defined per decision type rather than assumed globally.
Rationale	The 171-curve fixture requires batch disposition, while sessions are tied to file hash and settings must not persist invisibly between wells. 
UI effect	The user sees whether an action applies to one curve, selected curves, a detected prefix group, all matching items in this file, or a reusable local rule.
Backend dependency	Target-query definition, scope persistence policy, session-record schema
Status	Proposed; shared UX/backend
Priority	Must settle before batch disposition and local-table editing
Example batch review
text
Apply mapping action

Action:
Strip recognized prefix "LFP_" and re-match mnemonic

Scope:
147 curves in this file with prefix "LFP_"

Will not affect:
24 curves without the prefix
Any future file
Existing exported files

Evidence:
Prefix recognized from local mnemonic table, version 1.4

[Preview affected curves]   [Cancel]   [Apply to 147 curves]
This makes batch operation auditable rather than merely convenient.

UXC5 — Presentation controls must be analytically non-operative
Field	Proposed decision
ID	UXC5
Decision	Track selection, track order, zoom, pan, overlay visibility, legend expansion, and display presets must be presentation-only. They may not alter curve identity, validation eligibility, gap classification, transformation state, export contents, or recorded scientific decisions.
Rationale	The brief distinguishes reversible exploration from consequential actions, and prohibits invisible cross-well settings. 
UI effect	Users can safely inspect alternative track arrangements or presets without changing the dataset’s interpretation.
Backend dependency	Clear separation between view state, session decision state, and transformation state
Status	Proposed; UX-owned once separation exists
Priority	Must settle before presets, custom layouts, and saved workspace behavior
Consequence
A “sonic/well-tie” track preset, if adopted, is a view configuration. It must not silently cause different validation, derivation, or fill behavior.

UXC6 — Full-well provenance communicates modification, not every detailed state
Field	Proposed decision
ID	UXC6
Decision	At full-well scale, the UI must preserve the distinction between unmodified and modified data and make major absent/refused intervals visible. It must not claim that created, changed, invalid, refused, and other detailed states are all independently legible at that scale. Detailed distinctions belong to mid-zoom, sample inspection, and the record layer.
Rationale	The brief requires multi-scale provenance and a conservative decimation rule; F12 notes that a single full-well band channel cannot carry four or five distinct states reliably. 
UI effect	A full-well screenshot cannot hide modification, but it does not overload the display with illegible hatch/tint combinations.
Backend dependency	Pixel-level modified-state aggregation; interval-level status summaries
Status	Proposed; UI-owned with rendering support
Priority	Must settle before coverage-track visual design locks in
Display hierarchy
Zoom level	Required truth
Full well	“Has this interval been modified or refused/absent?”
Mid interval	“Was it created, changed, invalid, or refused?”
Sample/hover	“What exactly happened, by which method, using which inputs, under which rule?”
Record	“Can a third party reproduce and audit the decision?”
UXC7 — Refusal must show evidence, consequence, and non-action without shame
Field	Proposed decision
ID	UXC7
Decision	A refused method or gap must be displayed as a completed, evidence-backed scientific outcome. The UI must state what was not done, why it was not done, the affected depth interval or curve, whether the raw measurement remains visible, and whether the refusal blocks export or only the method.
Rationale	The brief explicitly defines refusal as a correct result and requires a refusals panel. D52 and D86 provide examples of scientific-precondition refusal. 
UI effect	Refusals are understandable and useful rather than looking like a list of software failures.
Backend dependency	Refusal reason, block type, affected entity/span, available alternative actions
Status	Proposed; shared UX/backend
Priority	Must settle before gap-filling and derivation interfaces
Example refusal card
text
Not filled — resistivity endpoint is non-positive

Interval:
8,421.5–8,426.0 ft

Method considered:
Log-domain interpolation

Reason:
The left endpoint is 0.0 ohm·m. Logarithmic interpolation is undefined for
non-positive values, so POLISH did not create values in this interval.

Result:
Gap remains open. Original samples are unchanged.

Export effect:
Does not block export. Recorded in provenance and refusals report.
This implements the plan’s core promise: a refusal explains itself.

UXC8 — Dataset lineage must remain visible after transformations
Field	Proposed decision
ID	UXC8
Decision	When transformations create a child dataset—especially resampling—the UI must identify the active dataset, parent dataset, depth-grid relationship, applied transformations, and available source view. It must never imply that source and child samples are the same index-aligned measurements when they are not.
Rationale	D43/F1 identifies the overlay/resampling conflict as structural. The UI cannot solve lineage, but it can either preserve or destroy user understanding of the eventual architecture. 
UI effect	A lineage breadcrumb, dataset selector, and explicit source-versus-derived comparison mode.
Backend dependency	Dataset lineage graph, transformation records, coordinate mapping
Status	Proposed; architecture-dependent
Priority	Block detailed transformed-data UI until lineage model is settled
Required UI concept
text
Viewing: Resampled dataset
Parent: Unit-normalized source dataset
Source LAS: immutable
Depth grid: 0.5 ft regularized child grid
Transformations:
1. Units normalized
2. Resampled
3. Two accepted dropout fills
The interface should never compress this into a misleading binary “original / processed” toggle when several meaningful descendants exist.

UXC9 — Validation eligibility must be visible and distinct from validation pass
Field	Proposed decision
ID	UXC9
Decision	The UI must distinguish “passed validation,” “failed validation,” “warning,” “not assessed,” and “not eligible for automatic validation.” A curve whose family or unit confidence is insufficient must never appear to have passed merely because no violation was found.
Rationale	D11 and D87 require separation of rule types and explicit eligibility where unit/family confidence is low. 
UI effect	Users can see why a curve was not range-validated and what evidence or mapping would be needed to make it eligible.
Backend dependency	Validation-rule library, family/unit confidence, eligibility outcome, rule-version data
Status	Proposed; shared UX/backend
Priority	Must settle before validity/coverage legends and curve-detail panels
Recommended status language
Backend condition	UI label
Rule applied, no violation	“Validated against rule set vX.Y”
Rule applied, warning-level violation	“Plausibility warning”
Rule applied, hard violation	“Measured value excluded from processing”
No eligible rule due to uncertainty	“Not automatically validated”
Not yet run	“Validation pending”
Rule blocked by scientific precondition	“Validation method not applicable”
UXC10 — Export review must explain both permission and provenance
Field	Proposed decision
ID	UXC10
Decision	Export review must separately display: selected export profile; source and active dataset identity; transformation summary; outstanding reviewable dispositions; non-overridable blockers; stale dependencies; and provenance artifacts included in the export. Confirmation is enabled only when the backend declares exportable state.
Rationale	The brief requires deliberate irreversible export; D85 defines three export profiles; D86 disallows a general force-export override; D60 requires round-trip invariants. 
UI effect	The user knows whether they are exporting an auditable processed result, an interoperability output, or merely retaining original input—and knows what the file claims.
Backend dependency	Export eligibility engine, profile rules, lineage/provenance manifest, stale/block state
Status	Proposed; shared UX/backend
Priority	Must settle before export implementation
Part C — Design Brief Supplement
1. UI/UX role in the product
POLISH’s interface is the user’s verification surface. It does not determine whether a curve is physically plausible, whether a gap is tool absence, whether a log-domain transformation is mathematically defined, or whether an LAS file conforms to the target specification. It must instead make those determinations visible, inspectable, and appropriately actionable.

The interface must satisfy five user-facing obligations:

Show what POLISH knows.
Curve identity, confidence, units, validation status, gap classification, alteration state, and export readiness must be available without requiring users to infer them from output values.

Show what POLISH does not know.
Unknown mnemonic, insufficient unit confidence, ambiguous gap class, unquantifiable uncertainty, and unsupported format behavior must be presented as bounded uncertainty—not silently coerced into a confident result.

Show what POLISH did not do.
Refused fills, ineligible validation, declined derivation, structural workflow stops, and blocked exports are first-class outcomes.

Keep reversible exploration cheap.
Zoom, track selection, overlay visibility, display presets, and inspection must be immediate and non-destructive.

Make consequential choices deliberate and recorded.
Mapping ambiguity, accepting a reviewable discrepancy, choosing an export profile, and committing an export must show scope, evidence, consequences, and persistence behavior before commitment.

2. Interface boundaries
UI/UX may decide
Coverage-track visual language within the stated provenance and readability constraints.

Curve-selection mechanics for a 171-curve file.

Track density, selection presets, and presentation-only view customization.

Queue layout, keyboard traversal, filter behavior, grouping, and batch-review affordances.

Information hierarchy for evidence, options, consequences, and dispositions.

Refusal-panel organization and tone.

Staleness communication and recovery navigation.

Hover, annotation, record, and export-review interaction patterns.

How fixed-position panels reserve space and avoid reflow.

UI/UX must not decide alone
Whether a sample is invalid, only implausible, or merely unvalidated.

Which physical limits apply to a curve.

Whether a gap is tool absence, dropout, or ambiguous.

Whether a method is scientifically eligible.

Whether a finding is structurally non-overridable.

Whether a transformed dataset preserves original-sample identity.

Whether a file meets LAS export conformance.

Whether a curve may be used as a derivation input.

UI/UX must demand from architecture
Canonical state fields rather than vague trust labels.

A block type for every finding.

A lineage model for every transformed dataset.

Explicit scope and persistence semantics for every user decision.

Rule identifiers, confidence tiers, unit information, and eligibility explanations.

Stale-dependency graphs and recovery actions.

Export eligibility and manifest summaries.

Stable identifiers for source file, dataset, curve, and sample/interval references.

3. The primary workspace
The primary workspace should allow a professional user to answer four questions quickly:

What is in this file?

What is suspicious, uncertain, missing, altered, or refused?

What decisions, if any, require me?

Can I export a result whose claims I understand?

Fixed workspace regions
Region	Purpose	Must show	Must not do
Stage rail	Pipeline and dependency state	Current, incomplete, stale, blocked, completed stages	Pretend stages are a one-way wizard
Coverage track	Whole-well evidence map	Measured/modified summary, significant gaps, absences, refusals, selection location	Require settings or processing before first inspection
Log-track area	Chosen detailed curve subset	Conventional composite-log readability; source versus overlay state	Reflow or displace depth context
Resolution queue	Actionable review items	Evidence, permitted actions, consequence, scope, disposition/block status	Turn every issue into a modal dialog
Detail/record pane	Inspection and audit	Rule/method/input provenance, source values, confidence, history	Cover the depth view or erase spatial context
Export review	Deliberate final action	Profile, lineage, changes, blocks, stale status, included provenance	Provide a generic “force export” escape hatch
This preserves the brief’s depth-as-spine and non-reflow principles while adding the system-level state needed for cohesion.

4. Required interaction patterns
4.1 The evidence-first queue item
Every reviewable queue item must have the same stable anatomy:

text
WHAT POLISH OBSERVED
- Entity and depth/curve context
- Evidence and confidence
- Relevant source/units/rule/method

WHAT IT MEANS
- Operational consequence
- What POLISH will and will not do automatically
- Downstream impact if changed

WHAT YOU MAY DO
- Only actions permitted by block type
- Explicit scope, where propagation is possible
- Rationale field where an interpretation is being recorded

WHAT HAPPENS NEXT
- Resulting disposition
- Stale dependencies, if any
- Export effect
This extends the brief’s evidence/options/consequence/disposition model by making permitted authority and downstream impact explicit.

4.2 Batch decisions
Batch actions must be framed as a review of a proposed rule application, not as a bulk convenience control.

A batch action must show:

The action to be applied.

Why the system believes the items form a group.

The exact affected count.

The list or preview of affected items.

The non-affected items.

Scope: current file only, current session only, or explicit local rule creation.

The evidence source and table/rule version.

The possibility of reviewing exceptions before commit.

This satisfies the 171-curve stress requirement without creating silent mass interpretation.

4.3 Stale recovery
A stale item is not merely “unfinished.” It was previously assessed under assumptions that are no longer current.

The stale recovery interaction must:

Identify the triggering upstream change.

List affected stages, queue items, and export claims.

State whether the existing result remains visible but non-current.

Offer a direct route to rerun or re-disposition.

Preserve the previous decision in history rather than silently overwriting it.

Block export when the stale relationship affects export correctness.

4.4 Provenance inspection
Provenance must work at four levels:

Level	User question	UI response
Full well	“Where has anything been modified or refused?”	Interval-level modified/refused/absent encoding
Track/interval	“What happened here?”	State, method category, interval span, summary evidence
Sample hover	“What is this value and where did it come from?”	Value, measured/created/changed status, method, inputs, unit, source/child context
Record	“Can I audit or reproduce the claim?”	Full regeneration record, rule version, source samples/curves, parameters, decision history
The UI must never force a user to inspect the deepest record merely to know whether a visible value is measured or generated.

Part D — Backend Dependency Register for UI/UX
This is the “do not mock past this boundary” list. A UI design can sketch these states, but implementation must wait until the underlying contract is defined.

Dependency ID	UI feature affected	Required backend truth	Existing source basis	UI risk if unresolved
DEP-UX-01	Original/processed comparison, overlay toggle, hover provenance	Dataset lineage and source-to-child coordinate mapping after resampling	D43/F1	UI falsely suggests original and resampled samples are directly comparable by index
DEP-UX-02	Legends, queue categories, status badges, export summary	Orthogonal state schema	D21/F4	Same word means different things in different panels; contradictory states appear
DEP-UX-03	Queue actions, export blockers, refusal handling	Block-type classification and override policy	D79, D86	UI exposes unsafe “accept” paths or treats blocks as normal review items
DEP-UX-04	Validation screen, curve-detail state, warnings	Versioned validation-rule library and eligibility engine	D11, D87	“No warning” is mistaken for “validated”
DEP-UX-05	Gap map, fill consent, refusal detail	Tool grouping, evidence model, confidence, classification result	D41, D49	UI presents a guessed gap class as fact
DEP-UX-06	Interpolation controls and error states	Explicit interpolation domain and scientific precondition results	D42, D52	UI offers a fill action that cannot be performed scientifically
DEP-UX-07	TVD and depth-related views	Computed-geometric-reference role and resampling rules	D7, D32/F2	TVD is rendered or processed as if it were an ordinary measured log
DEP-UX-08	Export review, profile picker, re-import messaging	Profile schema, output manifest, round-trip tests	D59–D63, D85	User cannot know what the exported file preserves or claims
DEP-UX-09	File audit and refusal screen	LAS parser audit ledger and conformance outcomes	D64–D68	UI must invent confidence about malformed or ambiguous input
DEP-UX-10	Scope control and stored preferences	Decision persistence and scope policy	D46–D48, D38	A file-specific decision silently becomes a global preference or vice versa
Part E — Implementation Priority
Vocabulary (before Stage 0 scaffolding)
Before detailed mockups, document the state vocabulary and the lineage assumptions. This work produces no polished interface; it produces a truthful vocabulary. POLISH numbering is Stages 0–10 (D6, D23); the retired "Phase 0–3" labels are not used.

Deliverables:

Canonical state glossary.

Block-type/action matrix.

Dataset-lineage diagram.

Decision-scope model.

Export-state matrix.

Validation-eligibility states.

Staleness dependency model.

Why first: Without these, UI mockups will create attractive but false simplifications.

Read-only trust surface (Stages 0–4)
Build the portion of POLISH that provides value without altering data:

Ingest audit ledger.

Depth-axis outcome.

Curve identification and evidence tiers.

Unit interpretation and eligibility state.

Stage 4 validity/gap coverage track.

Curve-selection workflow.

Non-modal resolution queue for read-only ambiguities.

Refusal/blocked workflow presentation.

Provenance and audit record surfaces where applicable.

This aligns with the register’s suggestion that Stages 0–4 can form a sellable, low-liability read-only milestone and validates the product’s trust model before introducing creation of data.

UI goal: A user can load a difficult LAS file and immediately understand what POLISH found, what it trusts, what it declines to infer, and what needs their attention.

Consent and state transitions (Stages 0–4)
Add:

Explicit mapping decisions.

Scope preview and batch disposition.

Staleness propagation.

Session/hash mismatch review.

Validation-rule detail.

Export-readiness summary without writing altered logs.

UI goal: A user can make and revisit recorded decisions without hidden persistence, silent scope expansion, or uncertainty about downstream consequences.

Data-altering operations (Stages 5–8)
Only after dataset lineage and state contracts are implemented:

Resampling as child-dataset creation.

Derivation with source/equation provenance.

Gap filling with evidence, preconditions, consent, and refusal.

Denoising with changed-sample visibility.

Original-versus-child comparison.

Full regeneration record.

Export-profile selection and deliberate export confirmation.

UI goal: Every generated or altered value is visibly different from a measurement, inspectable, reversible at the view level, and traceable into export.

Part F — Final Cohesion Test
Before declaring the UI/UX complete, run these scenarios end-to-end.

Scenario 1 — Tool absence must remain empty
A density curve is absent over a long interval and co-tool evidence supports tool absence.

The UI passes only if it:

Shows the interval in the coverage track.

States that it is non-fillable by default.

Shows the evidence and confidence.

Allows inspection of the refusal.

Does not offer an enticing “fill anyway” action.

Keeps the original hole visible.

Records the non-action in the final report/export.

Scenario 2 — Non-positive resistivity blocks log interpolation
A short resistivity gap has a zero or negative endpoint.

The UI passes only if it:

Shows that interpolation was considered.

States the scientific precondition failure.

Explains that logarithmic interpolation cannot use non-positive endpoints.

Leaves the gap unfilled.

Distinguishes this from a parser crash or generic error.

States whether export remains allowed.

Records the refusal.

Scenario 3 — Units change after gap review
A user changes or resolves a unit interpretation after Stage 4 has classified gaps.

The UI passes only if it:

Marks dependent validity and gap decisions stale.

Identifies the cause.

Does not silently recalculate and hide the difference.

Blocks export if the stale state affects exported claims.

Gives a direct recovery route.

Preserves the prior review decision in history.

Scenario 4 — 147 related curves receive a batch mapping
A file contains 147 curves sharing a recognized vendor prefix.

The UI passes only if it:

Presents the grouping evidence.

Previews the exact 147 affected curves.

Explains the mapping rule.

States scope and persistence.

Allows inspection of exceptions.

Records one auditable batch decision.

Avoids 147 dialogs and unnecessary scrolling.

Scenario 5 — Resampling creates a child dataset
A user elects to resample a curve set.

The UI passes only if it:

Identifies the active child dataset.

Preserves access to the immutable source dataset.

Makes the changed depth grid explicit.

Does not render source and child as if they have identical sample identities.

Shows resampling as creation of new samples.

Preserves provenance through hover, record, and export review.

Does not let the “original” view become a synthesized representation.

Scenario 6 — Export is blocked correctly
A user attempts export with one stale decision, one accepted ambiguity, and one structural data failure.

The UI passes only if it:

Shows the accepted ambiguity as recorded and permitted.

Shows the stale decision and its recovery path.

Shows the structural failure as non-overridable.

Keeps export disabled.

Does not offer force export.

Clearly separates “review completed” from “export permitted.”

Final Decision
The UI/UX layer can make POLISH’s plan coherent for the user if it is designed as an evidence-and-state-navigation system rather than a settings surface.

It must:

Expose the distinction between data, interpretation, and presentation.

Make uncertainty and refusal useful rather than embarrassing.

Preserve the user’s ability to inspect origin, method, scope, consequence, and export meaning.

Scale to 171 curves without hiding mass decisions.

Make stale dependencies and non-overridable blocks obvious.

Preserve the difference between “not validated,” “validated,” “invalid,” “unknown,” “created,” “changed,” and “refused.”

Never use visual simplicity to conceal scientific or architectural uncertainty.

The UI/UX cannot repair an undefined lineage model, unresolved validation rules, ambiguous state schema, incorrect gap classification, or incomplete export invariant. Those must be settled underneath it.

But once those foundations exist, the interface is not a secondary layer. It is the place where POLISH keeps its promise: the user can see what the system found, what it changed, what it refused to change, and why.