"""Stage 2: identify each curve. It does not rewrite samples.

Implements D31, D32 (with F2), D33, D34, D35, D36, D37, D38, D39, D40, D41,
D42, D91, C3, UXC1, UXC4, and UXC9.

Structure
---------
CONTRACT
    Stage 2 contract. alters_data is False. depends_on_stages is (0, 1).
processing_eligibility_for
    The D34 mapping from tier, family, and role. Units are not an input.
identify
    Read a Stage 0 result and a completed Stage 1 result. Return one
    identification per curve, plus findings.
registered_stage
    The same identification behind the stage runner.

Matching, in order: exact whole token; a recognised qualifier split; a
vendor-prefix proposal that is not applied; a similarity suggestion that is
not applied. Pattern fallback, unit scoring, and the legacy confidence
formula are not used.

Similarity is copied from core/curve_identification.py
``CurveIdentificationEngine._fuzzy_match_mnemonic``, including its guards.
A suggestion never becomes the curve's family.

UXC1: role, evidence tier, and processing eligibility stay separate fields.
UXC9: this stage does not set a validation outcome. Not eligible is not a pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from polish.model.mapping import ExplicitMapping, roles_stated_by_family
from polish.model.findings import DecisionScope, Evidence, FindingRecord
from polish.model.ids import CurveRef
from polish.model.states import (
    USER_MAPPED_LABEL,
    BlockType,
    CurveRole,
    EvidenceTier,
    ProcessingEligibility,
    StageStatus,
)
from polish.stages.contract import StageContract, StageGuarantees, StageRequires
from polish.stages.runner import RegisteredStage, RunContext, StageOutcome
from polish.stages.s00_ingest import IngestResult
from polish.stages.s01_depth import DepthAxisResult
from polish.tables.loader import (
    MnemonicRecord,
    QualifierToken,
    ShippedTables,
    VendorPrefix,
    default_local_tables_directory,
    load_local_mnemonics,
    load_shipped_tables,
)

# Legacy constant, not yet reviewed by the domain expert.
# Source: core/curve_identification.py SHORT_MNEMONIC_MAX_LEN.
SHORT_MNEMONIC_MAX_LEN = 2

# Legacy constant, not yet reviewed.
# Source: core/curve_identification.py CurveIdentificationEngine._fuzzy_match_mnemonic
# threshold parameter, default 0.7.
SIMILARITY_THRESHOLD = 0.7

# Copied from core/curve_identification.py EXACT_ONLY_MNEMONICS. Every name in
# the set is already at most SHORT_MNEMONIC_MAX_LEN characters. The similarity
# guard consults the set, so the set is kept with the guard.
_EXACT_ONLY_MNEMONICS = frozenset(
    {
        "K",
        "U",
        "TH",
        "PE",
        "MI",
        "MN",
        "SP",
        "BS",
        "MD",
        "DF",
        "KB",
        "FP",
        "C1",
        "C2",
        "C3",
        "C4",
    }
)

# Domain expert, 14 Sept 2026: leave these unchanged and notify. They are not
# mapped to a family, and they get no similarity suggestion.
UNCHANGED_MNEMONICS = frozenset(
    {
        "RM",
        "RD",
        "RACEHM",
        "RACELM",
        "RPCEHM",
        "RPCELM",
        "ABDCQF01",
        "ABDCQF02",
        "ABDCQF03",
        "ABDCQF04",
        "NBGRCFM",
        "ROP",
    }
)

_LOW_TIER = frozenset({EvidenceTier.PROPOSED, EvidenceTier.NO_MATCH})
_ROLE_EXCLUDED = frozenset({CurveRole.ANCILLARY, CurveRole.SYNTHETIC_SOURCE, CurveRole.UNKNOWN})
_APPLIED_RULES = frozenset({"exact", "qualifier_split"})
_PASS_OPTIONS = (DecisionScope.THIS_ITEM.value, DecisionScope.LOCAL_RULE.value)


def _is_short_or_exact_only_mnemonic(mnemonic: str) -> bool:
    """Whole-token gate. Copied from core/curve_identification.py ``_is_short_or_exact_only_mnemonic``."""

    clean = mnemonic.upper().strip()
    if not clean:
        return False
    if len(clean) <= SHORT_MNEMONIC_MAX_LEN:
        return True
    return clean in _EXACT_ONLY_MNEMONICS


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Edit distance. Copied from core/curve_identification.py ``CurveIdentificationEngine._levenshtein_distance``."""

    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    # i is the growing prefix length. The legacy helper uses it the same way.
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _fuzzy_match_mnemonic(mnemonic: str, known_mnemonic: str, threshold: float) -> tuple[bool, float]:
    """Suggestion guard. Copied from core/curve_identification.py ``_fuzzy_match_mnemonic``.

    The short-name gate, the containment refusal, the length-ratio refusal,
    and the threshold are unchanged. A true result is a suggestion, not an
    identification.
    """

    mnemonic_clean = mnemonic.upper().strip()
    known_clean = known_mnemonic.upper().strip()
    if mnemonic_clean == known_clean:
        return True, 1.0
    max_len = max(len(mnemonic_clean), len(known_clean))
    if max_len == 0:
        return False, 0.0
    distance = _levenshtein_distance(mnemonic_clean, known_clean)
    similarity = 1.0 - (distance / max_len)
    if _is_short_or_exact_only_mnemonic(known_clean):
        return False, similarity
    if _is_short_or_exact_only_mnemonic(mnemonic_clean):
        return False, similarity
    if mnemonic_clean in known_clean or known_clean in mnemonic_clean:
        return False, similarity
    shorter = min(len(mnemonic_clean), len(known_clean))
    longer = max(len(mnemonic_clean), len(known_clean))
    if shorter <= SHORT_MNEMONIC_MAX_LEN and longer > shorter:
        return False, similarity
    if longer >= shorter * 2 and shorter <= 3:
        return False, similarity
    return similarity >= threshold, similarity


@dataclass(frozen=True)
class Qualifier:
    """A qualifier a table states. D31, D39, D40."""

    token: str
    name: str


@dataclass(frozen=True)
class Suggestion:
    """A similarity proposal. It is not the curve's identification. D33, D34."""

    family_id: str
    mnemonic: str
    score: float
    method: str
    tier: EvidenceTier

    def __post_init__(self) -> None:
        if self.method != "similarity":
            raise ValueError("A suggestion's method is similarity.")
        if self.tier is not EvidenceTier.PROPOSED:
            raise ValueError("A suggestion's tier is PROPOSED.")
        if isinstance(self.score, bool) or not isinstance(self.score, float):
            raise TypeError("A suggestion score must be a float.")


@dataclass(frozen=True)
class TraceStep:
    """One line of the match trace. D31.

    ``applied`` is false for a prefix proposal, a similarity suggestion, and
    a local/shipped clash. Those are recorded and not used as the family.
    """

    table: str
    entry: str
    rule: str
    applied: bool


@dataclass(frozen=True)
class HypotheticalMatch:
    """The match a recognised prefix would produce if it were stripped. D35.

    This match is not applied. P15 is the batch decision that could apply it.
    """

    mnemonic: str | None
    family_id: str | None
    role: CurveRole
    tier: EvidenceTier
    qualifiers: tuple[Qualifier, ...]


@dataclass(frozen=True)
class PrefixProposal:
    """One curve's unapplied prefix-strip proposal. D35."""

    prefix: str
    table_version: str
    match: HypotheticalMatch


@dataclass(frozen=True)
class CurveIdentification:
    """Identification of one curve. D31, D32, D33, D34, UXC1.

    ``family_id`` is None when no table entry was applied. ``label`` is
    USER_MAPPED_LABEL exactly when the tier is USER_DEFINED. Suggestions do
    not change the family, the tier, or the role.
    """

    curve: CurveRef
    mnemonic: str
    family_id: str | None
    qualifiers: tuple[Qualifier, ...]
    role: CurveRole
    evidence_tier: EvidenceTier
    processing_eligibility: ProcessingEligibility
    suggestions: tuple[Suggestion, ...]
    trace: tuple[TraceStep, ...]
    label: str | None
    user_label: str | None
    prefix_proposal: PrefixProposal | None

    def __post_init__(self) -> None:
        if not isinstance(self.curve, CurveRef):
            raise TypeError("curve must be a CurveRef.")
        if not isinstance(self.role, CurveRole):
            raise TypeError("role must be a CurveRole.")
        if not isinstance(self.evidence_tier, EvidenceTier):
            raise TypeError("evidence_tier must be an EvidenceTier.")
        if not isinstance(self.processing_eligibility, ProcessingEligibility):
            raise TypeError("processing_eligibility must be a ProcessingEligibility.")
        if self.evidence_tier is EvidenceTier.USER_DEFINED:
            if self.label != USER_MAPPED_LABEL:
                raise ValueError("A USER_DEFINED curve must carry USER_MAPPED_LABEL.")
        elif self.label is not None:
            raise ValueError("Only a USER_DEFINED curve carries USER_MAPPED_LABEL.")


@dataclass(frozen=True)
class IdentificationResult:
    """What Stage 2 records. Source arrays are not in this result. C5, D17."""

    status: StageStatus
    findings: tuple[FindingRecord, ...]
    curves: tuple[CurveIdentification, ...]


@dataclass(frozen=True)
class _TextMatch:
    """The mnemonic-table result before the Stage 1 index role is applied."""

    family_id: str | None
    matched_mnemonic: str | None
    role: CurveRole
    tier: EvidenceTier
    qualifiers: tuple[Qualifier, ...]
    label: str | None
    user_label: str | None
    trace: tuple[TraceStep, ...]
    suggestions: tuple[Suggestion, ...]
    origin: str | None


@dataclass(frozen=True)
class _Catalog:
    shipped: ShippedTables
    shipped_by_key: dict[str, MnemonicRecord]
    local_by_key: dict[str, MnemonicRecord]
    clashes: frozenset[str]
    qualifiers: tuple[QualifierToken, ...]


def processing_eligibility_for(
    tier: EvidenceTier,
    family_id: str | None,
    role: CurveRole,
) -> ProcessingEligibility:
    """Map tier, family, and role to eligibility. D34, D40, C3, UXC9.

    Low tier is first: PROPOSED and NO_MATCH are not eligible. A missing
    family is next. ANCILLARY, SYNTHETIC_SOURCE, and UNKNOWN then decline.
    UNKNOWN declines because an unrecognised role is not processed (C3).
    Unit-based outcomes belong to Stage 3 and are not returned here.

    Parameters
    ----------
    tier:
        Evidence tier of the identification.
    family_id:
        Shipped family id, or None when no family was applied.
    role:
        Curve role after Stage 1 index assignment and qualifier rules.

    Returns
    -------
    ProcessingEligibility
        One of the D34 values other than NOT_ELIGIBLE_UNIT_UNRESOLVED.
    """

    if not isinstance(tier, EvidenceTier):
        raise TypeError("tier must be an EvidenceTier.")
    if family_id is not None and not isinstance(family_id, str):
        raise TypeError("family_id must be a str or None.")
    if not isinstance(role, CurveRole):
        raise TypeError("role must be a CurveRole.")
    if tier in _LOW_TIER:
        return ProcessingEligibility.NOT_ELIGIBLE_LOW_TIER
    if family_id is None or family_id.strip() == "":
        return ProcessingEligibility.NOT_ELIGIBLE_NO_FAMILY
    if role in _ROLE_EXCLUDED:
        return ProcessingEligibility.NOT_ELIGIBLE_ROLE
    return ProcessingEligibility.ELIGIBLE


def _table_name(origin: str) -> str:
    if origin == "local":
        return "local_mnemonics"
    return "shipped_mnemonics"


def _catalog(shipped: ShippedTables, local: tuple[MnemonicRecord, ...]) -> _Catalog:
    shipped_by_key = {row.mnemonic.upper(): row for row in shipped.mnemonics}
    local_by_key = {row.mnemonic.upper(): row for row in local}
    clashes = frozenset(key for key in local_by_key if key in shipped_by_key)
    qualifiers = tuple(sorted(shipped.qualifiers, key=lambda token: len(token.token), reverse=True))
    return _Catalog(shipped, shipped_by_key, local_by_key, clashes, qualifiers)


def _split_qualifiers(key: str, tokens: tuple[QualifierToken, ...]) -> tuple[str, tuple[QualifierToken, ...]]:
    """Strip recognised trailing tokens only. D36, D39, D40.

    An unrecognised trailing token stops the split. Nothing else is removed.
    """

    stem = key
    found: list[QualifierToken] = []
    while True:
        matched: QualifierToken | None = None
        for token in tokens:
            if len(stem) > len(token.token) and stem.endswith(token.token):
                matched = token
                break
        if matched is None:
            break
        stem = stem[: -len(matched.token)]
        found.append(matched)
    found.reverse()
    return stem, tuple(found)


def _qualifiers_of(row: MnemonicRecord, tokens: tuple[QualifierToken, ...]) -> tuple[Qualifier, ...]:
    stated = tuple(Qualifier(token=item, name=item) for item in row.qualifiers)
    split = tuple(Qualifier(token=token.token, name=token.name) for token in tokens)
    return stated + split


def _role_of(row: MnemonicRecord, tokens: tuple[QualifierToken, ...]) -> CurveRole:
    if any(token.role is CurveRole.SYNTHETIC_SOURCE for token in tokens):
        return CurveRole.SYNTHETIC_SOURCE
    if row.role is None:
        return CurveRole.UNKNOWN
    return row.role


def _tier_of(row: MnemonicRecord) -> EvidenceTier:
    try:
        return EvidenceTier(row.source_tier)
    except ValueError as exc:
        raise ValueError(
            f"Mnemonic {row.mnemonic!r} has source tier {row.source_tier!r}, which is not an evidence tier."
        ) from exc


def _applied(row: MnemonicRecord, tokens: tuple[QualifierToken, ...], trace: list[TraceStep], rule: str) -> _TextMatch:
    trace.append(TraceStep(_table_name(row.origin), row.mnemonic, rule, True))
    tier = _tier_of(row)
    label = USER_MAPPED_LABEL if tier is EvidenceTier.USER_DEFINED else None
    return _TextMatch(
        family_id=row.family_id,
        matched_mnemonic=row.mnemonic,
        role=_role_of(row, tokens),
        tier=tier,
        qualifiers=_qualifiers_of(row, tokens),
        label=label,
        user_label=row.user_label if row.origin == "local" else None,
        trace=tuple(trace),
        suggestions=(),
        origin=row.origin,
    )


def _lookup(catalog: _Catalog, key: str) -> MnemonicRecord | None:
    if key in catalog.clashes:
        return None
    local = catalog.local_by_key.get(key)
    if local is not None:
        return local
    return catalog.shipped_by_key.get(key)


def _suggestions_for(mnemonic: str, catalog: _Catalog) -> tuple[Suggestion, ...]:
    found: list[Suggestion] = []
    for row in catalog.shipped.mnemonics:
        if row.family_id is None:
            continue
        matched, score = _fuzzy_match_mnemonic(mnemonic, row.mnemonic, SIMILARITY_THRESHOLD)
        if not matched:
            continue
        found.append(
            Suggestion(
                family_id=row.family_id,
                mnemonic=row.mnemonic,
                score=float(score),
                method="similarity",
                tier=EvidenceTier.PROPOSED,
            )
        )
    found.sort(key=lambda item: (-item.score, item.mnemonic))
    return tuple(found)


def _unresolved(
    role: CurveRole,
    qualifiers: tuple[Qualifier, ...],
    trace: list[TraceStep],
    suggestions: tuple[Suggestion, ...],
) -> _TextMatch:
    return _TextMatch(
        family_id=None,
        matched_mnemonic=None,
        role=role,
        tier=EvidenceTier.NO_MATCH,
        qualifiers=qualifiers,
        label=None,
        user_label=None,
        trace=tuple(trace),
        suggestions=suggestions,
        origin=None,
    )


def _clash_trace(catalog: _Catalog, key: str, trace: list[TraceStep]) -> None:
    shipped = catalog.shipped_by_key[key]
    local = catalog.local_by_key[key]
    trace.append(TraceStep("shipped_mnemonics", shipped.mnemonic, "local_shipped_clash", False))
    trace.append(TraceStep("local_mnemonics", local.mnemonic, "local_shipped_clash", False))


def _role_from_tokens(tokens: tuple[QualifierToken, ...]) -> CurveRole:
    if any(token.role is CurveRole.SYNTHETIC_SOURCE for token in tokens):
        return CurveRole.SYNTHETIC_SOURCE
    return CurveRole.UNKNOWN


def _match_text(mnemonic: str, catalog: _Catalog, *, similarity: bool) -> _TextMatch:
    """Exact, then qualifier split. Similarity is optional and never applied."""

    key = mnemonic.upper().strip()
    trace: list[TraceStep] = []
    if key in UNCHANGED_MNEMONICS:
        trace.append(TraceStep("unchanged_mnemonics", mnemonic.strip(), "unchanged", False))
        return _unresolved(CurveRole.UNKNOWN, (), trace, ())
    if key in catalog.clashes:
        _clash_trace(catalog, key, trace)
        return _unresolved(CurveRole.UNKNOWN, (), trace, ())
    row = _lookup(catalog, key)
    if row is not None:
        return _applied(row, (), trace, "exact")
    stem, tokens = _split_qualifiers(key, catalog.qualifiers)
    if tokens:
        qualifiers = tuple(Qualifier(token=token.token, name=token.name) for token in tokens)
        for token in tokens:
            trace.append(TraceStep("qualifier_tokens", token.token, "qualifier_split", True))
        if stem in catalog.clashes:
            _clash_trace(catalog, stem, trace)
            return _unresolved(_role_from_tokens(tokens), qualifiers, trace, ())
        row = _lookup(catalog, stem)
        if row is not None:
            return _applied(row, tokens, trace, "qualifier_split")
        trace.append(TraceStep("shipped_mnemonics", stem, "qualifier_split", False))
        suggestions = _suggestions_for(mnemonic, catalog) if similarity else ()
        for item in suggestions:
            trace.append(TraceStep("shipped_mnemonics", item.mnemonic, "similarity", False))
        return _unresolved(_role_from_tokens(tokens), qualifiers, trace, suggestions)
    suggestions = _suggestions_for(mnemonic, catalog) if similarity else ()
    for item in suggestions:
        trace.append(TraceStep("shipped_mnemonics", item.mnemonic, "similarity", False))
    trace.append(TraceStep("shipped_mnemonics", key, "no_match", False))
    return _unresolved(CurveRole.UNKNOWN, (), trace, suggestions)


def _prefix_for(mnemonic: str, prefixes: tuple[VendorPrefix, ...]) -> VendorPrefix | None:
    key = mnemonic.strip().upper()
    hits = [prefix for prefix in prefixes if key.startswith(prefix.prefix.upper())]
    if not hits:
        return None
    return max(hits, key=lambda prefix: len(prefix.prefix))


def _hypothetical(match: _TextMatch) -> HypotheticalMatch:
    return HypotheticalMatch(
        mnemonic=match.matched_mnemonic,
        family_id=match.family_id,
        role=match.role,
        tier=match.tier,
        qualifiers=match.qualifiers,
    )


def _stated_roles(catalog: _Catalog, family_id: str) -> tuple[CurveRole, ...]:
    """Roles the shipped mnemonics of this family state, in table order. D91.

    UNKNOWN is not a mapping role. A family that states more than one role
    does not have one chosen here. The list is ``roles_stated_by_family``
    on the shipped tables this catalog already holds.
    """

    if not isinstance(catalog, _Catalog):
        raise TypeError("catalog must be a _Catalog")
    return roles_stated_by_family(catalog.shipped, family_id)


def _explicit_text(mnemonic: str, mapping: ExplicitMapping, catalog: _Catalog) -> _TextMatch:
    """A this-well mapping. The mnemonic is kept. D91.

    The role is the one on the mapping. It is never UNKNOWN, and it is one
    of the roles this family's shipped mnemonics state.
    """

    if mapping.role is CurveRole.UNKNOWN:
        raise ValueError("A mapped curve cannot have role UNKNOWN.")
    stated = _stated_roles(catalog, mapping.family_id)
    if mapping.role not in stated:
        shown = ", ".join(item.value for item in stated) if stated else "none"
        raise ValueError(f"Role {mapping.role.value} is not one of {mapping.family_id}'s roles ({shown}).")
    trace = (TraceStep("session_mapping", mnemonic.strip(), "map_explicitly", True),)
    return _TextMatch(
        family_id=mapping.family_id,
        matched_mnemonic=mnemonic.strip(),
        role=mapping.role,
        tier=EvidenceTier.USER_DEFINED,
        qualifiers=(),
        label=USER_MAPPED_LABEL,
        user_label=mapping.user_label,
        trace=trace,
        suggestions=(),
        origin="session",
    )


def _identify_curve(
    curve_ref: CurveRef,
    mnemonic: str,
    index_ref: CurveRef,
    catalog: _Catalog,
    mapping: ExplicitMapping | None,
) -> CurveIdentification:
    if mapping is not None:
        if mnemonic.upper().strip() in UNCHANGED_MNEMONICS:
            raise ValueError(f"{mnemonic} is on the leave-unchanged list and is never mappable.")
        if catalog.shipped.family(mapping.family_id) is None:
            raise ValueError(f"Family {mapping.family_id!r} is not in the shipped tables.")
        text = _explicit_text(mnemonic, mapping, catalog)
    else:
        text = _match_text(mnemonic, catalog, similarity=True)
    trace = list(text.trace)
    proposal: PrefixProposal | None = None
    prefix = _prefix_for(mnemonic, catalog.shipped.prefixes)
    if prefix is not None:
        remainder = mnemonic.strip()[len(prefix.prefix) :]
        hypothetical = _match_text(remainder, catalog, similarity=False)
        trace.append(TraceStep("vendor_prefixes", prefix.prefix, "prefix_strip_proposal", False))
        if hypothetical.matched_mnemonic is not None:
            origin = hypothetical.origin if hypothetical.origin is not None else "shipped"
            trace.append(
                TraceStep(_table_name(origin), hypothetical.matched_mnemonic, "prefix_strip_proposal", False)
            )
        proposal = PrefixProposal(prefix.prefix, catalog.shipped.prefixes_version, _hypothetical(hypothetical))
    role = CurveRole.INDEX if curve_ref == index_ref else text.role
    if curve_ref == index_ref:
        trace.append(TraceStep("stage_1", mnemonic, "index", True))
    return CurveIdentification(
        curve=curve_ref,
        mnemonic=mnemonic,
        family_id=text.family_id,
        qualifiers=text.qualifiers,
        role=role,
        evidence_tier=text.tier,
        processing_eligibility=processing_eligibility_for(text.tier, text.family_id, role),
        suggestions=text.suggestions,
        trace=tuple(trace),
        label=text.label,
        user_label=text.user_label,
        prefix_proposal=proposal,
    )


class _Ledger:
    """Findings for one identification. Identifiers are stable inside that call."""

    def __init__(self, ingest: IngestResult) -> None:
        self._read_at = ingest.source_file.read_at
        self.items: list[FindingRecord] = []

    def add(
        self,
        block: BlockType,
        message: str,
        refs: tuple[CurveRef, ...],
        fields: dict[str, str | int | float | bool],
    ) -> None:
        number = len(self.items) + 1
        evidence = Evidence(block.value, tuple(sorted(fields.items())))
        self.items.append(
            FindingRecord(
                id=f"s02-{number:04d}",
                stage=2,
                block_type=block,
                subject_refs=refs,
                evidence=evidence,
                message=message,
                created_at=self._read_at,
            )
        )


def _json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True)


def _clash_key(ident: CurveIdentification) -> str | None:
    for step in ident.trace:
        if step.rule == "local_shipped_clash":
            return step.entry.upper()
    return None


def _base_mnemonic(ident: CurveIdentification) -> str | None:
    if ident.family_id is None:
        return None
    for step in ident.trace:
        if step.applied and step.rule in _APPLIED_RULES and step.table.endswith("_mnemonics"):
            return step.entry
    return None


def _pass_name(ident: CurveIdentification) -> str:
    tokens = {qualifier.token for qualifier in ident.qualifiers}
    if "_MAIN" in tokens:
        return "main pass"
    if "_REF" in tokens:
        return "repeat pass"
    return "unqualified"


def _add_unknown_notices(ledger: _Ledger, curves: tuple[CurveIdentification, ...]) -> None:
    for ident in curves:
        if ident.family_id is not None or _clash_key(ident) is not None:
            continue
        fields: dict[str, str | int | float | bool] = {
            "mnemonic": ident.mnemonic,
            "preserved": True,
        }
        unchanged = ident.mnemonic.upper().strip() in UNCHANGED_MNEMONICS
        no_family = ident.evidence_tier is EvidenceTier.USER_DEFINED
        if unchanged:
            block = BlockType.INFORMATIONAL
            message = (
                f"Mnemonic {ident.mnemonic} is not recognised. It is preserved exactly and is not processed."
            )
        elif no_family:
            block = BlockType.RULE_DEPENDENT_WARNING
            if ident.user_label is not None:
                fields["user_label"] = ident.user_label
            message = (
                f"Curve {ident.mnemonic} was mapped by you with no family. "
                "POLISH cannot process it until you choose which family it behaves as."
            )
        else:
            block = BlockType.RULE_DEPENDENT_WARNING
            message = (
                f"Mnemonic {ident.mnemonic} is not recognised. It is preserved exactly and is not "
                "processed until you choose which family it behaves as."
            )
        if ident.suggestions and not unchanged and not no_family:
            best = ident.suggestions[0]
            fields["suggested_family"] = best.family_id
            fields["similarity_score"] = best.score
            fields["method"] = "similarity"
            fields["suggestion_tier"] = EvidenceTier.PROPOSED.value
            fields["suggestions"] = _json(
                [
                    {
                        "family_id": item.family_id,
                        "method": item.method,
                        "mnemonic": item.mnemonic,
                        "score": item.score,
                        "tier": item.tier.value,
                    }
                    for item in ident.suggestions
                ]
            )
        ledger.add(block, message, (ident.curve,), fields)


def _add_duplicates(ledger: _Ledger, curves: tuple[CurveIdentification, ...]) -> None:
    groups: dict[str, list[CurveIdentification]] = {}
    for ident in curves:
        groups.setdefault(ident.mnemonic.upper().strip(), []).append(ident)
    for key in sorted(groups):
        members = groups[key]
        if len(members) < 2:
            continue
        columns = ", ".join(str(item.curve.column_index) for item in members)
        spellings = ", ".join(item.mnemonic for item in members)
        ledger.add(
            BlockType.INFORMATIONAL,
            f"Mnemonic {spellings} is on more than one curve (columns {columns}). Each curve is identified on its own.",
            tuple(item.curve for item in members),
            {
                "mnemonic": members[0].mnemonic,
                "curve_refs": _json([str(item.curve) for item in members]),
            },
        )


def _add_passes(ledger: _Ledger, curves: tuple[CurveIdentification, ...]) -> None:
    by_family: dict[str, list[tuple[CurveIdentification, str]]] = {}
    for ident in curves:
        base = _base_mnemonic(ident)
        if ident.family_id is None or base is None:
            continue
        by_family.setdefault(ident.family_id, []).append((ident, base))
    for family_id in sorted(by_family):
        by_base: dict[str, list[CurveIdentification]] = {}
        for ident, base in by_family[family_id]:
            by_base.setdefault(base, []).append(ident)
        members: list[CurveIdentification] = []
        bases: list[str] = []
        for base in sorted(by_base):
            group = by_base[base]
            has_pass = any(_pass_name(ident) != "unqualified" for ident in group)
            if len(group) >= 2 and has_pass:
                members.extend(group)
                bases.append(base)
        if len(members) < 2:
            continue
        ledger.add(
            BlockType.AMBIGUOUS_INTERPRETATION,
            (
                f"More than one pass of {family_id} is present ({', '.join(bases)}). "
                "Which pass should be used by default? Scope options are this well and all wells. "
                "No default is chosen. Every pass is preserved."
            ),
            tuple(item.curve for item in members),
            {
                "family_id": family_id,
                "base_mnemonic": ",".join(bases),
                "scope_options": _json(list(_PASS_OPTIONS)),
                "options": _json(sorted({_pass_name(item) for item in members})),
                "passes": _json(
                    [
                        {
                            "curve": str(item.curve),
                            "mnemonic": item.mnemonic,
                            "pass": _pass_name(item),
                        }
                        for item in members
                    ]
                ),
            },
        )


def _add_prefixes(ledger: _Ledger, curves: tuple[CurveIdentification, ...]) -> None:
    groups: dict[str, list[CurveIdentification]] = {}
    for ident in curves:
        if ident.prefix_proposal is None:
            continue
        groups.setdefault(ident.prefix_proposal.prefix, []).append(ident)
    for prefix in sorted(groups):
        members = groups[prefix]
        version = members[0].prefix_proposal.table_version if members[0].prefix_proposal is not None else ""
        stripped = []
        for ident in members:
            proposal = ident.prefix_proposal
            if proposal is None:
                continue
            match = proposal.match
            stripped.append(
                {
                    "curve": str(ident.curve),
                    "mnemonic": ident.mnemonic,
                    "would_match_family": match.family_id,
                    "would_match_mnemonic": match.mnemonic,
                    "would_match_role": match.role.value,
                    "would_match_tier": match.tier.value,
                }
            )
        ledger.add(
            BlockType.AMBIGUOUS_INTERPRETATION,
            (
                f"Curves share the recognised prefix {prefix}. The prefix is not stripped. "
                "A decision is required before any prefix is removed."
            ),
            tuple(item.curve for item in members),
            {
                "prefix": prefix,
                "table_version": version,
                "members": _json([str(item.curve) for item in members]),
                "stripped_matches": _json(stripped),
            },
        )


def _add_clashes(ledger: _Ledger, curves: tuple[CurveIdentification, ...], catalog: _Catalog) -> None:
    groups: dict[str, list[CurveIdentification]] = {}
    for ident in curves:
        key = _clash_key(ident)
        if key is None:
            continue
        groups.setdefault(key, []).append(ident)
    for key in sorted(groups):
        shipped = catalog.shipped_by_key[key]
        local = catalog.local_by_key[key]
        local_family = local.family_id if local.family_id is not None else "none"
        members = groups[key]
        ledger.add(
            BlockType.AMBIGUOUS_INTERPRETATION,
            (
                f"Mnemonic {shipped.mnemonic} is in the shipped table (family {shipped.family_id}) "
                f"and the local table (family {local_family}). Neither entry is applied."
            ),
            tuple(item.curve for item in members),
            {
                "mnemonic": shipped.mnemonic,
                "shipped_entry": _json(
                    {
                        "family": shipped.family_id,
                        "mnemonic": shipped.mnemonic,
                        "source_tier": shipped.source_tier,
                        "table": "shipped_mnemonics",
                    }
                ),
                "local_entry": _json(
                    {
                        "family": local.family_id,
                        "mnemonic": local.mnemonic,
                        "table": "local_mnemonics",
                        "user_label": local.user_label,
                    }
                ),
            },
        )


CONTRACT = StageContract(
    stage_number=2,
    name="Identification",
    requires=StageRequires(
        units="Units are not interpreted here. A family's accepted dimension is recorded and not applied to samples.",
        completeness="Stage 0 has read the file and Stage 1 has classified the depth index.",
        preconditions=(
            "Stage 0 returned the dataset.",
            "Stage 1 completed and named the index curve.",
        ),
    ),
    guarantees=StageGuarantees(
        row_count="Unchanged. Stage 2 does not add, drop, or reorder a sample.",
        units="Unchanged. No unit is converted or invented.",
        validity=(
            "Each curve has a family or none, qualifiers, a role, an evidence tier, "
            "and a processing eligibility. Sample status is not set. Validation outcome is not set."
        ),
    ),
    declines_when=(
        "An unrecognised mnemonic stays UNKNOWN, with no family, and is not processed.",
        "A recognised vendor prefix is not stripped.",
        "A local mnemonic that repeats a shipped mnemonic is not applied.",
        "A similarity suggestion is not adopted as the identification.",
    ),
    depends_on_stages=(0, 1),
    alters_data=False,
)


def identify(
    ingest: IngestResult,
    depth: DepthAxisResult,
    local_tables_directory: Path | None = None,
    explicit_mappings: tuple[ExplicitMapping, ...] = (),
) -> IdentificationResult:
    """Identify every curve. D31–D42, D91, C3, UXC1, UXC4, UXC9.

    Does not write the dataset, the overlay, or the local tables directory.
    When ``local_tables_directory`` is omitted, the app-data path is used if
    APPDATA is set. A missing directory is empty. It is not created.
    ``explicit_mappings`` are THIS_ITEM mappings from a session. They are
    applied only because the caller passed them. Marking an earlier result
    stale is not done here.

    Parameters
    ----------
    ingest:
        The Stage 0 result whose dataset is classified.
    depth:
        A completed Stage 1 result. Its alias is the INDEX curve.
    local_tables_directory:
        Directory of local mnemonic tables, or None to use the default path.
    explicit_mappings:
        THIS_ITEM mappings to apply on this call. Empty applies none.

    Returns
    -------
    IdentificationResult
        One CurveIdentification per curve, in column order, and the findings.

    Raises
    ------
    TypeError
        If ``ingest`` or ``depth`` has the wrong type.
    ValueError
        If Stage 0 has no dataset or Stage 1 did not complete with an index.
    """

    if not isinstance(ingest, IngestResult):
        raise TypeError("ingest must be an IngestResult.")
    if not isinstance(depth, DepthAxisResult):
        raise TypeError("depth must be a DepthAxisResult.")
    if ingest.dataset is None:
        raise ValueError("Stage 2 requires the Stage 0 dataset.")
    if depth.status is not StageStatus.COMPLETE:
        raise ValueError("Stage 2 requires a completed Stage 1 result.")
    if depth.alias is None:
        raise ValueError("Stage 1 completed without an index alias.")
    dataset = ingest.dataset
    if depth.alias.source.dataset_id != dataset.id:
        raise ValueError("Stage 1 index alias is not a curve of this dataset.")
    directory = default_local_tables_directory() if local_tables_directory is None else local_tables_directory
    shipped = load_shipped_tables()
    local = load_local_mnemonics(directory, frozenset(record.id for record in shipped.families))
    catalog = _catalog(shipped, local)
    by_curve = _mapping_index(explicit_mappings, tuple(curve.ref for curve in dataset.curves))
    index_ref = depth.alias.source
    identified = tuple(
        _identify_curve(curve.ref, curve.original_mnemonic, index_ref, catalog, by_curve.get(curve.ref))
        for curve in dataset.curves
    )
    ledger = _Ledger(ingest)
    _add_unknown_notices(ledger, identified)
    _add_duplicates(ledger, identified)
    _add_passes(ledger, identified)
    _add_prefixes(ledger, identified)
    _add_clashes(ledger, identified, catalog)
    return IdentificationResult(StageStatus.COMPLETE, tuple(ledger.items), identified)


def _mapping_index(
    mappings: tuple[ExplicitMapping, ...],
    curves: tuple[CurveRef, ...],
) -> dict[CurveRef, ExplicitMapping]:
    """Index THIS_ITEM mappings by curve. A curve not in this file is refused. D91."""

    if type(mappings) is not tuple:
        raise TypeError("explicit_mappings must be a tuple of ExplicitMapping")
    known = set(curves)
    found: dict[CurveRef, ExplicitMapping] = {}
    for mapping in mappings:
        if not isinstance(mapping, ExplicitMapping):
            raise TypeError("explicit_mappings must be a tuple of ExplicitMapping")
        if mapping.curve not in known:
            raise ValueError(f"Explicit mapping {mapping.curve} is not a curve of this dataset.")
        if mapping.curve in found:
            raise ValueError(f"Two explicit mappings name {mapping.curve}.")
        found[mapping.curve] = mapping
    return found


def registered_stage(
    ingest: IngestResult,
    depth: DepthAxisResult,
    local_tables_directory: Path | None = None,
    explicit_mappings: tuple[ExplicitMapping, ...] = (),
) -> RegisteredStage:
    """A runner stage that identifies ``ingest`` and does not replace it. D24, C5.

    ``local_tables_directory`` is forwarded to ``identify``. None uses the
    app-data path and does not create it.
    """

    def unmet(context: RunContext) -> tuple[str, ...]:
        if ingest.dataset is None:
            return ("Stage 0 did not return a dataset.",)
        if context.dataset is not ingest.dataset:
            return ("Stage 2 requires the Stage 0 dataset.",)
        if depth.status is not StageStatus.COMPLETE:
            return ("Stage 1 did not complete.",)
        return ()

    def run(_context: RunContext) -> StageOutcome:
        result = identify(
            ingest,
            depth,
            local_tables_directory=local_tables_directory,
            explicit_mappings=explicit_mappings,
        )
        return StageOutcome(result.status, result.findings)

    return RegisteredStage(CONTRACT, run, unmet)
