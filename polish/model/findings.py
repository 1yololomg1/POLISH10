"""Finding records and the decisions linked to them.

Implements D46, D47, D63, D79, D86, D91, D92, D94, D96, UXC2, UXC4. ``block_type`` is
``states.BlockType``. This module does not define a second finding type for
the export gate: ``project_for_export`` builds the existing ``states.Finding``.

``states.Finding`` stays the export-gate record. ``FindingRecord`` is the fuller
record a stage will raise. Dispositions live on ``DecisionRecord``, not on the
finding.

Evidence keys are not invented here. Each block type has a declared schema.
A finding kind with no declared schema cannot be constructed.

DecisionRecord is the one decision type. It carries the glossary disposition
(issue disposition and reason) and the finding it is linked to. P13 adds the
chosen option, scope, actor, timestamp, app version, the versions in force,
the input file hash, and the decision it supersedes. The rationale for
CHOOSE_OPTION, REJECT_PROPOSAL and MAP_EXPLICITLY, and the reason for DEFER,
are ``Disposition.reason``. There is no second reason field.

Structure
---------
DecisionScope, DecisionActor
    The one scope vocabulary and the two actors.
Evidence, FindingRecord, DecisionRecord
    Immutable records. A changed decision is a new record.
finding_key, evidence_digest
    A finding's stable id, and the digest a decision records. D96.
project_for_export
    The export-gate ``states.Finding``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from polish.model.ids import CurveRef, DatasetId, FileHash, SampleRef
from polish.model.states import (
    BlockType,
    CurveRole,
    Disposition,
    Finding,
    IssueDisposition,
    UserAction,
)

# Stages 0–10 inclusive. Eleven stages. D6, D23.
_STAGE_MIN = 0
_STAGE_MAX = 10

# D96. A check name is the code that raised the finding. An identity name is
# what that finding is about. A measured value, a count, a score, a version,
# a message, a file hash, a dataset id, or a CurveRef string is not an identity.
_CHECK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_IDENTITY_NAMES = frozenset(
    {
        "section",
        "line",
        "field",
        "column",
        "mnemonic",
        "sentinel",
        "endpoint",
        "family_id",
        "prefix",
        "rule_id",
        "sample_start",
        "sample_end",
        "decision_id",
        "file",
    }
)
_DECIMAL_IDENTITY = frozenset({"line", "column", "sample_start", "sample_end"})
_DECIMAL_TEXT = re.compile(r"^(0|[1-9][0-9]*)$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

SubjectRef = DatasetId | CurveRef | SampleRef | FileHash
EvidenceValue = str | int | float | bool


def _validate_identity(identity: tuple[tuple[str, str], ...]) -> None:
    """Refuse an identity that is not a unique list of allowed names. D96."""

    if type(identity) is not tuple:
        raise TypeError("identity must be a tuple of (name, value) pairs")
    seen: set[str] = set()
    for pair in identity:
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError("each identity entry must be a (name, value) pair")
        name, value = pair
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("identity name and value must be strings")
        if name not in _IDENTITY_NAMES:
            raise ValueError(f"Identity name {name!r} is not allowed.")
        if name in seen:
            raise ValueError(f"Duplicate identity name {name!r}.")
        seen.add(name)
        if name in _DECIMAL_IDENTITY:
            if _DECIMAL_TEXT.fullmatch(value) is None:
                raise ValueError(f"Identity {name} must be decimal text, for example '7'.")
        elif name != "mnemonic" and value == "":
            raise ValueError(f"Identity {name} must be a non-empty string.")


def finding_key(stage: int, check: str, identity: tuple[tuple[str, str], ...]) -> str:
    """Stable id for one finding. D96.

    The id is the stage, the check that raised the finding, and the first 16
    hex digits of the SHA-256 of the canonical JSON of that stage, check, and
    identity. Identity pairs are sorted by name. The id does not include the
    finding's position in a run or an observed value.

    Parameters
    ----------
    stage:
        Stage number, from 0 to 10.
    check:
        The check that raised the finding. A lowercase name.
    identity:
        What the finding is about, as (name, value) pairs. Names are unique
        and come from the allowed set. A mnemonic may be empty when the file
        gives none. Line, column, and sample indexes are decimal text.

    Returns
    -------
    str
        ``sNN.check.<16 hex>``.

    Raises
    ------
    ValueError
        The stage, check, or identity is not in the allowed form.
    """

    if type(stage) is not int or not _STAGE_MIN <= stage <= _STAGE_MAX:
        raise ValueError("stage must be an int from 0 to 10")
    if not isinstance(check, str) or _CHECK_NAME.fullmatch(check) is None:
        raise ValueError("check must be a lowercase name.")
    _validate_identity(identity)
    pairs = tuple(sorted(identity, key=lambda item: item[0]))
    encoded = json.dumps(
        {"stage": stage, "check": check, "identity": pairs},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"s{stage:02d}.{check}.{digest}"


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
    {
        "declared",
        "observed",
        "difference",
        "difference_in_step_intervals",
        "compared_quantity",
        "compared_value",
        "options",
    }
)
_STAGE1_EVIDENCE_BLOCKS = frozenset(
    {
        BlockType.INFORMATIONAL,
        BlockType.AMBIGUOUS_INTERPRETATION,
        BlockType.STRUCTURAL_DATA_FAILURE,
    }
)
# Stage 2 (P10, P13). Structured lists are JSON text because an evidence value
# is a string, number, or bool, not a nested object. The mapping notice uses
# the same keys on RULE_DEPENDENT_WARNING (D91). user_label is the optional
# name on a mapping that has no family.
_STAGE2_NOTICE_KEYS = frozenset(
    {
        "mnemonic",
        "curve_refs",
        "preserved",
        "suggested_family",
        "similarity_score",
        "method",
        "suggestion_tier",
        "suggestions",
        "user_label",
    }
)
# A recorded pass that this file does not list names the family, the pass,
# and the decision (D94). family_id and decision_id are already permitted on
# INFORMATIONAL through later sets; pass_name is declared for this notice.
_STAGE2_INFORMATIONAL_KEYS = _STAGE2_NOTICE_KEYS | frozenset({"decision_id", "family_id", "pass_name"})
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
        "options",
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
# Stage 4 (P12, P13, D92). Depth lists are JSON text. A hard-limit question
# and a suspicious keep-or-exclude question share the ambiguous keys.
# The SP log flag is informational. Per-sample plausibility is no longer a
# rule-dependent warning, so those keys are not on RULE_DEPENDENT_WARNING.
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
        "options",
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
        "rule_id",
        "library_version",
        "sample_count",
        "outside_count",
        "non_absent_count",
        "outside_fraction_more_than",
        "log_level",
        "minimum",
        "maximum",
    }
)
# A session load reports a hash mismatch and a decision that no longer
# matches a finding. These are not stage results. D48, D63.
_SESSION_AMBIGUOUS_KEYS = frozenset({"input_file_hash", "session_file_hash"})
_SESSION_INFORMATIONAL_KEYS = frozenset({"decision_id", "finding_id"})


def _evidence_keys(block_type: BlockType) -> frozenset[str]:
    keys: frozenset[str] = frozenset()
    if block_type in _STAGE1_EVIDENCE_BLOCKS:
        keys = keys | _STAGE1_EVIDENCE_KEYS
    if block_type is BlockType.INFORMATIONAL:
        keys = keys | _STAGE2_INFORMATIONAL_KEYS
        keys = keys | _STAGE4_INFORMATIONAL_KEYS
        keys = keys | _SESSION_INFORMATIONAL_KEYS
    if block_type is BlockType.AMBIGUOUS_INTERPRETATION:
        keys = keys | _STAGE2_AMBIGUOUS_KEYS
        keys = keys | _STAGE3_AMBIGUOUS_KEYS
        keys = keys | _STAGE4_AMBIGUOUS_KEYS
        keys = keys | _SESSION_AMBIGUOUS_KEYS
    if block_type is BlockType.RULE_DEPENDENT_WARNING:
        keys = keys | _STAGE2_NOTICE_KEYS
        keys = keys | _STAGE3_RULE_DEPENDENT_KEYS
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
    """One immutable finding. D79, D86, D96, UXC2.

    The disposition is not stored here. It is a DecisionRecord linked by
    ``id``. ``block_type`` is ``states.BlockType``. ``id`` is
    ``finding_key(stage, check, identity)``. ``check`` is the check that
    raised the finding. ``identity`` is what it is about, not a measured value.
    """

    id: str
    stage: int
    check: str
    identity: tuple[tuple[str, str], ...]
    block_type: BlockType
    subject_refs: tuple[SubjectRef, ...]
    evidence: Evidence
    message: str
    created_at: datetime

    def __post_init__(self) -> None:
        if type(self.stage) is not int or not _STAGE_MIN <= self.stage <= _STAGE_MAX:
            raise ValueError("stage must be an int from 0 to 10")
        expected = finding_key(self.stage, self.check, self.identity)
        if self.id != expected:
            raise ValueError("finding id must be the key of its stage, check and identity.")
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


def evidence_digest(finding: FindingRecord) -> str:
    """SHA-256 hex of one finding's evidence. D96.

    The digest is the canonical JSON of the evidence fields, sorted by key.
    When the evidence has no fields, the digest covers the message text.
    The timestamp is not included. A change in what was observed changes
    this digest and leaves the finding id unchanged.

    Parameters
    ----------
    finding:
        The finding whose evidence is digested.

    Returns
    -------
    str
        64 lowercase hex characters.

    Raises
    ------
    TypeError
        ``finding`` is not a FindingRecord.
    """

    if not isinstance(finding, FindingRecord):
        raise TypeError("finding must be a FindingRecord")
    if finding.evidence.fields:
        payload = {key: value for key, value in finding.evidence.fields}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        raw = encoded.encode("utf-8")
    else:
        raw = finding.message.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class DecisionScope(Enum):
    """Where a decision applies. UXC4, OD-22, D91, D92.

    One vocabulary. In a message, LOCAL_RULE is "all wells" and THIS_ITEM is
    "this well". SELECTED_ITEMS, DETECTED_GROUP and ALL_MATCHING_IN_FILE are
    named here and are not applied to one finding (P15).
    """

    THIS_ITEM = "THIS_ITEM"
    SELECTED_ITEMS = "SELECTED_ITEMS"
    DETECTED_GROUP = "DETECTED_GROUP"
    ALL_MATCHING_IN_FILE = "ALL_MATCHING_IN_FILE"
    LOCAL_RULE = "LOCAL_RULE"


class DecisionActor(Enum):
    """Who recorded the decision. D48, OD-22.

    DEFAULT_APPLIED is a rule the user saved for all wells, applied on a
    later well and listed so it can be superseded.
    """

    USER = "USER"
    DEFAULT_APPLIED = "DEFAULT_APPLIED"


@dataclass(frozen=True)
class DecisionRecord:
    """Disposition linked to one finding. D46, D63, D79, D86, D91, D92, D96, UXC4.

    Glossary fields used here: issue disposition and its reason, via
    ``states.Disposition``, and the user action from the glossary action
    table when a person took one. ``action`` is None when no user action
    applies (a scientific refusal is recorded by POLISH, not by a user action).

    ``versions_in_force`` is the rule library and every shipped table, as
    the loaders report them: (name, version) pairs. ``supersedes`` is the
    decision id this record replaces, or None. The replaced record stays.
    ``evidence_digest`` is the digest of the finding's evidence when this
    decision was recorded. A later difference holds the decision. D96.
    ``family_id`` is required for MAP_EXPLICITLY and is a shipped family id.
    ``user_label`` is the optional name the user keeps. ``role`` is the curve
    role of a mapping. It is required for MAP_EXPLICITLY and is never
    UNKNOWN. Records are append-only: a change is a new record.
    """

    decision_id: str
    finding_id: str
    disposition: Disposition
    action: UserAction | None
    chosen_option: str | None
    scope: DecisionScope
    actor: DecisionActor
    timestamp: datetime
    app_version: str
    versions_in_force: tuple[tuple[str, str], ...]
    input_file_hash: FileHash
    supersedes: str | None
    evidence_digest: str
    family_id: str | None = None
    user_label: str | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        if not _nonempty(self.decision_id):
            raise ValueError("decision_id must be a non-empty string")
        if not _nonempty(self.finding_id):
            raise ValueError("finding_id must be a non-empty string")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("disposition must be a Disposition")
        if self.action is not None and not isinstance(self.action, UserAction):
            raise TypeError("action must be a UserAction or None")
        if self.chosen_option is not None and not _nonempty(self.chosen_option):
            raise ValueError("chosen_option must be a non-empty string or None")
        if not isinstance(self.scope, DecisionScope):
            raise TypeError("scope must be a DecisionScope")
        if not isinstance(self.actor, DecisionActor):
            raise TypeError("actor must be a DecisionActor")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if not _nonempty(self.app_version):
            raise ValueError("app_version must be a non-empty string")
        if type(self.versions_in_force) is not tuple or not self.versions_in_force:
            raise ValueError("versions_in_force must name the library and the tables in force")
        seen: set[str] = set()
        for pair in self.versions_in_force:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("each version is a (name, version) pair")
            name, version = pair
            if not _nonempty(name) or not _nonempty(version):
                raise ValueError("a version name and its version are non-empty strings")
            if name in seen:
                raise ValueError(f"Duplicate version name {name!r}.")
            seen.add(name)
        if not isinstance(self.input_file_hash, FileHash):
            raise TypeError("input_file_hash must be a FileHash")
        if self.supersedes is not None and not _nonempty(self.supersedes):
            raise ValueError("supersedes must be a decision id or None")
        if not isinstance(self.evidence_digest, str) or _DIGEST.fullmatch(self.evidence_digest) is None:
            raise ValueError("evidence_digest must be 64 lowercase hex characters.")
        if self.family_id is not None and not _nonempty(self.family_id):
            raise ValueError("family_id must be a family id or None")
        if self.user_label is not None and not _nonempty(self.user_label):
            raise ValueError("user_label must be non-empty text or None")
        if self.role is not None and not _nonempty(self.role):
            raise ValueError("role must be a curve role or None")
        if self.role is not None:
            try:
                parsed_role = CurveRole(self.role)
            except ValueError as exc:
                raise ValueError(f"role {self.role!r} is not a curve role.") from exc
            if parsed_role is CurveRole.UNKNOWN:
                raise ValueError("A mapped curve cannot have role UNKNOWN.")
        if self.action is UserAction.MAP_EXPLICITLY and not _nonempty(self.family_id or ""):
            raise ValueError("MAP_EXPLICITLY requires a family_id")
        if self.action is UserAction.MAP_EXPLICITLY and self.role is None:
            raise ValueError("MAP_EXPLICITLY requires a role.")


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
