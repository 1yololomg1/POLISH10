"""Finding records and the decisions linked to them.

Implements D79, D86, UXC2. ``block_type`` is ``states.BlockType``. This module
does not define a second finding type for the export gate: ``project_for_export``
builds the existing ``states.Finding``.

``states.Finding`` stays the export-gate record. ``FindingRecord`` is the fuller
record a stage will raise. Dispositions live on ``DecisionRecord``, not on the
finding.

Evidence keys are not invented here. Each block type has a declared schema with
an empty allowed-key set until a stage prompt names the keys. A finding kind
with no declared schema cannot be constructed.

DecisionRecord carries the glossary disposition (issue disposition and reason)
and the finding it is linked to. Scope, actor, and library versions are named
by a later prompt and are not in the glossary tables, so they are not fields
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from polish.model.ids import CurveRef, DatasetId, FileHash, SampleRef
from polish.model.states import (
    BlockType,
    Disposition,
    Finding,
    IssueDisposition,
    UserAction,
)

# Stages 0–10 inclusive. Eleven stages. D6, D23.
_STAGE_MIN = 0
_STAGE_MAX = 10

SubjectRef = DatasetId | CurveRef | SampleRef | FileHash
EvidenceValue = str | int | float | bool


def _nonempty(text: str) -> bool:
    return isinstance(text, str) and text.strip() != ""


@dataclass(frozen=True)
class EvidenceSchema:
    """The only evidence keys a finding kind may carry. D79, D86.

    ``allowed_keys`` is the whole set. A key that is absent is not filled in.
    """

    kind: str
    allowed_keys: frozenset[str]

    def __post_init__(self) -> None:
        if not _nonempty(self.kind):
            raise ValueError("An evidence schema needs a kind.")
        if type(self.allowed_keys) is not frozenset or not all(isinstance(key, str) for key in self.allowed_keys):
            raise TypeError("allowed_keys must be a frozenset of str")


# Stage 1 names the header-versus-data evidence. Stage 2 names the
# identification evidence. A finding may omit a key; it may not add one
# that is not listed.
_STAGE1_EVIDENCE_KEYS = frozenset(
    {"declared", "observed", "difference", "difference_in_step_intervals"}
)
_STAGE1_EVIDENCE_BLOCKS = frozenset(
    {
        BlockType.INFORMATIONAL,
        BlockType.AMBIGUOUS_INTERPRETATION,
        BlockType.STRUCTURAL_DATA_FAILURE,
    }
)
# Stage 2 (P10). Structured lists are JSON text because an evidence value
# is a string, number, or bool, not a nested object.
_STAGE2_INFORMATIONAL_KEYS = frozenset(
    {
        "mnemonic",
        "curve_refs",
        "preserved",
        "suggested_family",
        "similarity_score",
        "method",
        "suggestion_tier",
        "suggestions",
    }
)
_STAGE2_AMBIGUOUS_KEYS = frozenset(
    {
        "prefix",
        "table_version",
        "members",
        "stripped_matches",
        "family_id",
        "scope_options",
        "base_mnemonic",
        "passes",
        "shipped_entry",
        "local_entry",
        "mnemonic",
    }
)
# Stage 3 (P11). A candidate list is JSON text. RULE_DEPENDENT_WARNING had
# no evidence keys before this stage.
_STAGE3_AMBIGUOUS_KEYS = frozenset(
    {
        "declared_unit",
        "outcome",
        "candidates",
        "family_id",
        "mnemonic",
    }
)
_STAGE3_RULE_DEPENDENT_KEYS = frozenset(
    {
        "declared_unit",
        "outcome",
        "canonical_unit",
        "dimension",
        "family_id",
        "accepted_dimensions",
        "mnemonic",
    }
)
# Stage 4 (P12). Depth lists and sharing lists are JSON text. A finding may
# omit a key; the hard-violation question and the gap notice do not share keys.
_STAGE4_AMBIGUOUS_KEYS = frozenset(
    {
        "rule_id",
        "library_version",
        "depths",
        "sample_indexes",
        "sample_count",
        "question",
        "output_inclusion",
        "violation_effect",
        "minimum",
        "maximum",
        "mnemonic",
        "family_id",
    }
)
_STAGE4_RULE_DEPENDENT_KEYS = frozenset(
    {
        "rule_id",
        "library_version",
        "depths",
        "sample_indexes",
        "sample_count",
        "outside_count",
        "non_absent_count",
        "outside_fraction_more_than",
        "log_level",
        "minimum",
        "maximum",
        "mnemonic",
        "family_id",
    }
)
_STAGE4_INFORMATIONAL_KEYS = frozenset(
    {
        "reason",
        "eligibility_evidence",
        "gap_class",
        "gap_evidence",
        "confidence",
        "not_fillable",
        "sharing_curves",
        "overlap_measure",
        "depth_at_start",
        "depth_at_end",
        "sample_start",
        "sample_end",
        "mnemonic",
        "family_id",
    }
)


def _evidence_keys(block_type: BlockType) -> frozenset[str]:
    keys: frozenset[str] = frozenset()
    if block_type in _STAGE1_EVIDENCE_BLOCKS:
        keys = keys | _STAGE1_EVIDENCE_KEYS
    if block_type is BlockType.INFORMATIONAL:
        keys = keys | _STAGE2_INFORMATIONAL_KEYS
        keys = keys | _STAGE4_INFORMATIONAL_KEYS
    if block_type is BlockType.AMBIGUOUS_INTERPRETATION:
        keys = keys | _STAGE2_AMBIGUOUS_KEYS
        keys = keys | _STAGE3_AMBIGUOUS_KEYS
        keys = keys | _STAGE4_AMBIGUOUS_KEYS
    if block_type is BlockType.RULE_DEPENDENT_WARNING:
        keys = keys | _STAGE3_RULE_DEPENDENT_KEYS
        keys = keys | _STAGE4_RULE_DEPENDENT_KEYS
    return keys


DECLARED_EVIDENCE_SCHEMAS: dict[str, EvidenceSchema] = {
    block_type.value: EvidenceSchema(block_type.value, _evidence_keys(block_type)) for block_type in BlockType
}


@dataclass(frozen=True)
class Evidence:
    """Structured evidence for one finding kind. D79, D86.

    ``kind`` must be a declared schema. ``fields`` may contain only that
    schema's keys. Values are str, int, float, or bool. There is no default
    for a missing key.
    """

    kind: str
    fields: tuple[tuple[str, EvidenceValue], ...] = ()

    def __post_init__(self) -> None:
        schema = DECLARED_EVIDENCE_SCHEMAS.get(self.kind)
        if schema is None:
            raise ValueError(f"No evidence schema is declared for finding kind {self.kind!r}.")
        if type(self.fields) is not tuple:
            raise TypeError("fields must be a tuple of pairs")
        seen: set[str] = set()
        for pair in self.fields:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("each evidence field must be a (key, value) pair")
            key, value = pair
            if not isinstance(key, str):
                raise TypeError("evidence key must be a str")
            if key in seen:
                raise ValueError(f"Duplicate evidence key {key!r}.")
            seen.add(key)
            if key not in schema.allowed_keys:
                raise ValueError(f"Evidence key {key!r} is not declared for kind {self.kind!r}.")
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                continue
            if type(value) is int:
                continue
            if type(value) is float:
                continue
            raise TypeError(f"Evidence value for {key!r} must be str, int, float, or bool.")

    def as_dict(self) -> dict[str, EvidenceValue]:
        """A copy of the fields. The record itself stays a tuple."""

        return dict(self.fields)


@dataclass(frozen=True)
class FindingRecord:
    """One immutable finding. D79, D86, UXC2.

    The disposition is not stored here. It is a DecisionRecord linked by
    ``id``. ``block_type`` is ``states.BlockType``.
    """

    id: str
    stage: int
    block_type: BlockType
    subject_refs: tuple[SubjectRef, ...]
    evidence: Evidence
    message: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not _nonempty(self.id):
            raise ValueError("finding id must be a non-empty string")
        if type(self.stage) is not int or not _STAGE_MIN <= self.stage <= _STAGE_MAX:
            raise ValueError("stage must be an int from 0 to 10")
        if not isinstance(self.block_type, BlockType):
            raise TypeError("block_type must be a BlockType")
        if type(self.subject_refs) is not tuple:
            raise TypeError("subject_refs must be a tuple")
        for ref in self.subject_refs:
            if not isinstance(ref, (DatasetId, CurveRef, SampleRef, FileHash)):
                raise TypeError("subject ref must be a DatasetId, CurveRef, SampleRef, or FileHash")
        if not isinstance(self.evidence, Evidence):
            raise TypeError("evidence must be an Evidence")
        if self.evidence.kind != self.block_type.value:
            raise ValueError("evidence kind must be this finding's block type")
        if not _nonempty(self.message):
            raise ValueError("message must be a non-empty string")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")


@dataclass(frozen=True)
class DecisionRecord:
    """Disposition linked to one finding. D79, D86.

    Glossary fields used here: issue disposition and its reason, via
    ``states.Disposition``, and the user action from the glossary action
    table when a person took one. ``action`` is None when no user action
    applies (a scientific refusal is recorded by POLISH, not by a user action).

    Scope, actor, app version, and rule-library versions are not columns in
    the state glossary. They are not stored here.
    """

    decision_id: str
    finding_id: str
    disposition: Disposition
    action: UserAction | None

    def __post_init__(self) -> None:
        if not _nonempty(self.decision_id):
            raise ValueError("decision_id must be a non-empty string")
        if not _nonempty(self.finding_id):
            raise ValueError("finding_id must be a non-empty string")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("disposition must be a Disposition")
        if self.action is not None and not isinstance(self.action, UserAction):
            raise TypeError("action must be a UserAction or None")


def project_for_export(finding: FindingRecord, decision: DecisionRecord | None) -> Finding:
    """Project a finding and its current decision into the export-gate shape.

    Implements D79, D86, UXC2. The result is a ``states.Finding``. No decision
    projects as UNDISPOSITIONED. This does not invent DECLINED_BY_POLISH; the
    caller records that decision when POLISH declines.
    """

    if not isinstance(finding, FindingRecord):
        raise TypeError("finding must be a FindingRecord")
    if decision is None:
        disposition = Disposition(IssueDisposition.UNDISPOSITIONED)
    else:
        if not isinstance(decision, DecisionRecord):
            raise TypeError("decision must be a DecisionRecord or None")
        if decision.finding_id != finding.id:
            raise ValueError("DecisionRecord.finding_id does not match the finding id.")
        disposition = decision.disposition
    return Finding(finding.id, finding.block_type, disposition)
