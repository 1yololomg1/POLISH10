# Governing documents

`POLISH_pipeline_contracts.md` governs the pipeline invariants C1–C7: what must stay true between stages so a measured value is never silently invented, converted, clipped, or dropped.

`POLISH_decision_register.md` governs decisions D1–D99 and findings F1–F19 (revision 2). It records what was decided, by whom, and where an older sentence has been revised.

`POLISH_design_brief.md` governs interface intent: what POLISH is, who it is for, the load–describe–inspect–decide–alter–export loop, and the open visual questions. It does not govern petrophysical rule values.

`POLISH_specification_v2.md` is the 10 September work-order rewrite of the contracts document. It is kept as a historical specification; live invariants are C1–C7 in `POLISH_pipeline_contracts.md`, as revised by the register.

`POLISH_UIUX_cohesion_supplement.md` governs the user-facing verification surface: what must be visible, decidable, and exportable so honesty, provenance, refusal, and auditability hold in the workflow. Adopted items are UXC1–UXC10 (all ten).

`POLISH_build_plan.md` governs open decisions OD-nn, Cursor prompts Pnn, and the recorded answers in section 1.0 (owner and domain expert). It is the rebuild sequence, not a second set of scientific invariants.

`state_glossary.md` records the OD-04 state schema in words: one table per enum, the UXC2 and UXC9 wording where those tables give it, and the block-type/action matrix. It does not add values or export rules beyond Appendix A.

## Precedence

Contracts C1–C7 are the core invariants. Where a decision-register row (revision 2 or later) revises older text, the register row wins. The domain expert's answers (POLISH_build_plan.md section 1.0) outrank DESIGN-tier rows on petrophysics and workflow. Owner decisions recorded in POLISH_build_plan.md section 1.0 apply. The design brief governs interface intent where the register is silent. Any conflict not covered: stop and ask.

## Numbering

Use only these handles: Stages 0–10 (eleven stages), D-numbers, F-numbers, UXC-numbers, C1–C7, OD-numbers, P-numbers. Do not introduce steps, phases, tasks, or any other numbered scheme.

## Adopted UXC items

All ten: UXC1, UXC2, UXC3, UXC4, UXC5, UXC6, UXC7, UXC8, UXC9, UXC10.
