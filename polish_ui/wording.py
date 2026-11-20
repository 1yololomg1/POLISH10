"""User-facing wording, copied from docs/governing/state_glossary.md. UXC1, UXC2, UXC9.

Nothing here is invented display language. A code with no glossary sentence
is shown as its canonical token.
"""

from __future__ import annotations

from polish.model.states import BlockType, IssueDisposition, UserAction, ValidationOutcome

# Title-case names for Stages 0-10. The shell's STAGES tuple is this list.
# UXC2's structural sentence is built from the name of the stage that refused.
STAGE_NAMES = (
    "Ingest",
    "Depth Axis",
    "Identification",
    "Units",
    "Validity",
    "Resampling",
    "Derivation",
    "Gap Filling",
    "Denoising",
    "Manifest",
    "Export",
)


def structural_sentence(stage_name: str) -> str:
    """UXC2 structural stop, naming the stage that refused.

    The glossary's "past Depth Axis" sentence is this rule for Stage 1, which
    is the worked example. A Stage 0 refusal names Ingest.
    """

    return f"This file cannot proceed past {stage_name}"


def block_sentence(block: BlockType, stage_name: str) -> str:
    """The sentence shown for one finding. UXC2.

    A structural finding uses ``structural_sentence`` with its own stage.
    """

    if block is BlockType.STRUCTURAL_DATA_FAILURE:
        return structural_sentence(stage_name)
    return BLOCK_WORDING[block]


# state_glossary.md, block_type table, "Preferred user-facing wording" (UXC2).
# STRUCTURAL_DATA_FAILURE here is the Stage 1 example. Display goes through
# block_sentence so a refusal at another stage names that stage.
BLOCK_WORDING = {
    BlockType.INFORMATIONAL: "Recorded — no action required",
    BlockType.AMBIGUOUS_INTERPRETATION: "Your decision is required",
    BlockType.RULE_DEPENDENT_WARNING: "Map explicitly to continue this analysis",
    BlockType.SCIENTIFIC_PRECONDITION_FAILURE: "This method was not run",
    BlockType.STRUCTURAL_DATA_FAILURE: structural_sentence("Depth Axis"),
    BlockType.EXPORT_CONFORMANCE_FAILURE: "Export is blocked until this is corrected",
}

# state_glossary.md, "Block type, actions, disposition, export" table, Export effect column.
EXPORT_EFFECT = {
    BlockType.INFORMATIONAL: "Blocks only until dispositioned",
    BlockType.AMBIGUOUS_INTERPRETATION: "Blocks until dispositioned; deferrals listed in the export",
    BlockType.RULE_DEPENDENT_WARNING: "Blocks until dispositioned",
    BlockType.SCIENTIFIC_PRECONDITION_FAILURE: "Does not block export; recorded in refusals",
    BlockType.STRUCTURAL_DATA_FAILURE: "Blocks export; workflow stops at that stage",
    BlockType.EXPORT_CONFORMANCE_FAILURE: "Blocks until the condition no longer holds",
}

# Same table, Resulting disposition column.
ACTION_RESULT = {
    UserAction.ACKNOWLEDGE: IssueDisposition.ACCEPTED,
    UserAction.DEFER: IssueDisposition.DEFERRED,
    UserAction.CHOOSE_OPTION: IssueDisposition.RESOLVED,
    UserAction.REJECT_PROPOSAL: IssueDisposition.RESOLVED,
    UserAction.MAP_EXPLICITLY: IssueDisposition.RESOLVED,
}

# The order actions are listed in. Not a recommendation.
ACTION_ORDER = (
    UserAction.CHOOSE_OPTION,
    UserAction.MAP_EXPLICITLY,
    UserAction.ACKNOWLEDGE,
    UserAction.REJECT_PROPOSAL,
    UserAction.DEFER,
)

VIEW_ONLY = "View only. No disposition clears this finding."
NO_OPTIONS = "CHOOSE_OPTION is not offered: this finding lists no options."
NO_MNEMONIC = "MAP_EXPLICITLY is not offered: this finding is not about one named curve."


def validation_wording(outcome: ValidationOutcome, rule_set_version: str | None) -> str:
    """The UXC9 status sentence for one validation outcome."""

    if outcome is ValidationOutcome.PASSED:
        if not rule_set_version:
            return "PASSED (no rule-set version recorded)"
        return f"Validated against rule set v{rule_set_version}"
    table = {
        ValidationOutcome.PLAUSIBILITY_WARNING: "Plausibility warning",
        ValidationOutcome.HARD_VIOLATION_EXCLUDED: "Measured value excluded from processing",
        ValidationOutcome.NOT_ELIGIBLE: "Not automatically validated",
        ValidationOutcome.PENDING: "Validation pending",
        ValidationOutcome.NOT_APPLICABLE: "Validation method not applicable",
    }
    return table[outcome]
