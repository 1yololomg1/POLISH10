"""Load the shipped and local mnemonic tables, and the unit spellings. D31, D35, D37, D38, D39, D41, D42, D90, C1.

Structure
---------
FamilyRecord, MnemonicRecord, QualifierToken, VendorPrefix, ShippedTables
    The rows as read. No scientific field is filled in when the file omits it.
SpellingRecord, ConversionRecord, UnitSpellings
    spelling to canonical unit and dimension, and exact factor-and-offset
    conversions inside one dimension.
load_shipped_tables
    Read the four mnemonic JSON files under polish/tables. Read-only.
load_unit_spellings
    Read unit_spellings.json. Read-only.
default_local_tables_directory
    %APPDATA%\\POLISH\\tables . This function does not create it.
load_local_mnemonics
    Read mnemonics.json from a directory that already exists. Missing is empty.
load_local_decisions
    Read local_decisions.json the same way. Missing is empty.

A local mnemonic row is tier USER_DEFINED. ``decision_id`` is optional.
Saving rows is polish.decisions (P13). Nothing here is sent off the machine (D38).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from polish.model.states import CurveRole

_TABLES = Path(__file__).resolve().parent
_LOCAL_FILE_NAME = "mnemonics.json"
LOCAL_MNEMONICS_FILE = _LOCAL_FILE_NAME
LOCAL_DECISIONS_FILE = "local_decisions.json"
_INTERPOLATION_DOMAINS = frozenset({"LINEAR", "LOG"})
_LOCAL_KEYS = frozenset({"mnemonic", "family", "user_label", "unit", "role", "tier", "decision_id"})
_LOCAL_DECISION_KEYS = frozenset(
    {"decision_id", "question_kind", "match_key", "chosen_option", "reason", "created"}
)
_LOCAL_DECISIONS_NAME = LOCAL_DECISIONS_FILE
_EVIDENCE_DELIMITER = frozenset({";", "|", "\n", "\r"})


def _object(payload: object, what: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{what} must be a JSON object.")
    return payload


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{what} must be a non-empty string.")
    return value


def _optional_text(value: object, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a string or null.")
    if value.strip() == "":
        raise ValueError(f"{what} must be a non-empty string when present.")
    return value


def _bool_or_null(value: object, what: str) -> bool | None:
    if value is None:
        return None
    if type(value) is bool:
        return value
    raise ValueError(f"{what} must be true, false, or null.")


def _string_list(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() != "" for item in value):
        raise ValueError(f"{what} must be a list of non-empty strings.")
    return tuple(value)


def _role(value: object, what: str) -> CurveRole | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a curve role or null.")
    try:
        return CurveRole(value)
    except ValueError as exc:
        raise ValueError(f"{what} is not a curve role: {value!r}.") from exc


def _no_delimiter(value: str, what: str) -> str:
    if any(mark in value for mark in _EVIDENCE_DELIMITER):
        raise ValueError(f"{what} must not contain ';', '|', or a newline.")
    return value


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return _object(json.load(handle), path.name)


def _versioned(path: Path) -> tuple[str, list[object], dict[str, object]]:
    payload = _read_json(path)
    version = _text(payload.get("version"), f"{path.name} version")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path.name} entries must be a list.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name} metadata must be an object when present.")
    return version, entries, metadata


@dataclass(frozen=True)
class FamilyRecord:
    """One shipped family. D31, D41, D42, D90, F7.

    ``accepted_dimensions`` is a non-empty list. A curve in any of those
    dimensions is compatible with the family. Dimensions are not converted
    between. ``display_log_scale`` is a display attribute. It is not an
    interpolation domain. ``tool_group`` and ``interpolation_domain`` stay
    null where OD-13 and OD-14 are unanswered. A family has no range and no
    unit list.
    """

    id: str
    display_name: str
    accepted_dimensions: tuple[str, ...]
    accepted_dimensions_note: str | None
    tool_group: str | None
    interpolation_domain: str | None
    display_log_scale: bool | None
    legacy_group: str
    source: str
    source_tier: str


@dataclass(frozen=True)
class MnemonicRecord:
    """One mnemonic row, shipped or local. D31, D33, D37.

    ``family_id`` is None only on a local row the user did not tie to a
    shipped family. ``qualifiers`` are those the table states, which is empty
    for every migrated legacy name.
    """

    mnemonic: str
    family_id: str | None
    qualifiers: tuple[str, ...]
    role: CurveRole | None
    source: str
    source_tier: str
    origin: str
    user_label: str | None = None
    unit: str | None = None
    decision_id: str | None = None


@dataclass(frozen=True)
class QualifierToken:
    """A trailing token Stage 2 may split. D36, D39, D40.

    ``role`` is set only when the token itself assigns one (_SYNT).
    """

    token: str
    name: str
    role: CurveRole | None
    source: str
    source_tier: str


@dataclass(frozen=True)
class VendorPrefix:
    """A recognised vendor prefix. D35. Presence is not permission to strip."""

    prefix: str
    source: str
    source_tier: str


@dataclass(frozen=True)
class ShippedTables:
    """The four shipped tables and the indexes Stage 2 matches on."""

    families_version: str
    mnemonics_version: str
    qualifiers_version: str
    prefixes_version: str
    review_note: str
    families: tuple[FamilyRecord, ...]
    mnemonics: tuple[MnemonicRecord, ...]
    qualifiers: tuple[QualifierToken, ...]
    prefixes: tuple[VendorPrefix, ...]

    def family(self, family_id: str) -> FamilyRecord | None:
        """The shipped family with this id, or None."""

        for record in self.families:
            if record.id == family_id:
                return record
        return None

    def mnemonic(self, spelling: str) -> MnemonicRecord | None:
        """Shipped mnemonic, compared without regard to case. Whole token."""

        key = spelling.upper()
        for record in self.mnemonics:
            if record.mnemonic.upper() == key:
                return record
        return None


def _dimension_list(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{what} must be a non-empty list of dimension names.")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or item.strip() == "":
            raise ValueError(f"{what} must contain only non-empty strings.")
        names.append(item)
    if len(set(names)) != len(names):
        raise ValueError(f"{what} contains a duplicate dimension name.")
    return tuple(names)


def _family_record(entry: object, seen: set[str]) -> FamilyRecord:
    row = _object(entry, "family entry")
    family_id = _no_delimiter(_text(row.get("id"), "family id"), "family id")
    if family_id in seen:
        raise ValueError(f"Duplicate family id {family_id!r}.")
    seen.add(family_id)
    if "range" in row or "units" in row or "unit" in row:
        raise ValueError(f"Family {family_id} carries a range or a unit list.")
    if "unit_dimension" in row:
        raise ValueError(f"Family {family_id} still carries unit_dimension.")
    if "accepted_dimensions" not in row:
        raise ValueError(f"{family_id} must state accepted_dimensions.")
    if "accepted_dimensions_note" not in row:
        raise ValueError(f"{family_id} must state accepted_dimensions_note, using null when there is none.")
    tool_group = row.get("tool_group")
    if tool_group is not None:
        tool_group = _text(tool_group, f"{family_id} tool_group")
    domain = row.get("interpolation_domain")
    if domain is not None:
        domain = _text(domain, f"{family_id} interpolation_domain")
        if domain not in _INTERPOLATION_DOMAINS:
            raise ValueError(f"{family_id} interpolation_domain {domain!r} is not LINEAR or LOG.")
    if "display_log_scale" not in row:
        raise ValueError(f"{family_id} must state display_log_scale, using null when it was not stated.")
    return FamilyRecord(
        id=family_id,
        display_name=_text(row.get("display_name"), f"{family_id} display_name"),
        accepted_dimensions=_dimension_list(row.get("accepted_dimensions"), f"{family_id} accepted_dimensions"),
        accepted_dimensions_note=_optional_text(
            row.get("accepted_dimensions_note"), f"{family_id} accepted_dimensions_note"
        ),
        tool_group=tool_group,
        interpolation_domain=domain,
        display_log_scale=_bool_or_null(row.get("display_log_scale"), f"{family_id} display_log_scale"),
        legacy_group=_text(row.get("legacy_group"), f"{family_id} legacy_group"),
        source=_text(row.get("source"), f"{family_id} source"),
        source_tier=_text(row.get("source_tier"), f"{family_id} source_tier"),
    )


def _mnemonic_record(entry: object, family_ids: set[str], seen: set[str]) -> MnemonicRecord:
    row = _object(entry, "mnemonic entry")
    mnemonic = _no_delimiter(_text(row.get("mnemonic"), "mnemonic"), "mnemonic")
    key = mnemonic.upper()
    if key in seen:
        raise ValueError(f"Duplicate shipped mnemonic {mnemonic!r}.")
    seen.add(key)
    family_id = _text(row.get("family"), f"{mnemonic} family")
    if family_id not in family_ids:
        raise ValueError(f"{mnemonic} names unknown family {family_id!r}.")
    role = _role(row.get("role"), f"{mnemonic} role")
    if role is None:
        raise ValueError(f"Shipped mnemonic {mnemonic!r} must state a role.")
    return MnemonicRecord(
        mnemonic=mnemonic,
        family_id=family_id,
        qualifiers=_string_list(row.get("qualifiers"), f"{mnemonic} qualifiers"),
        role=role,
        source=_text(row.get("source"), f"{mnemonic} source"),
        source_tier=_text(row.get("source_tier"), f"{mnemonic} source_tier"),
        origin="shipped",
    )


def _qualifier(entry: object, seen: set[str]) -> QualifierToken:
    row = _object(entry, "qualifier entry")
    token = _text(row.get("token"), "qualifier token").upper()
    if not token.startswith("_"):
        raise ValueError(f"Qualifier token {token!r} must include the leading underscore.")
    if token in seen:
        raise ValueError(f"Duplicate qualifier token {token!r}.")
    seen.add(token)
    return QualifierToken(
        token=token,
        name=_text(row.get("name"), f"{token} name"),
        role=_role(row.get("role"), f"{token} role"),
        source=_text(row.get("source"), f"{token} source"),
        source_tier=_text(row.get("source_tier"), f"{token} source_tier"),
    )


def _prefix(entry: object, seen: set[str]) -> VendorPrefix:
    row = _object(entry, "vendor prefix entry")
    prefix = _no_delimiter(_text(row.get("prefix"), "vendor prefix"), "vendor prefix")
    key = prefix.upper()
    if key in seen:
        raise ValueError(f"Duplicate vendor prefix {prefix!r}.")
    seen.add(key)
    return VendorPrefix(
        prefix=prefix,
        source=_text(row.get("source"), f"{prefix} source"),
        source_tier=_text(row.get("source_tier"), f"{prefix} source_tier"),
    )


def load_shipped_tables() -> ShippedTables:
    """Read the shipped tables. D31, D35, D37, D39, D41, D42.

    The files are opened for reading only. A missing key is an error. A
    missing scientific value is left null; it is not replaced.
    """

    families_version, family_entries, family_meta = _versioned(_TABLES / "families.json")
    families: list[FamilyRecord] = []
    family_ids: set[str] = set()
    for entry in family_entries:
        families.append(_family_record(entry, family_ids))
    mnemonics_version, mnemonic_entries, mnemonic_meta = _versioned(_TABLES / "mnemonics.json")
    mnemonics: list[MnemonicRecord] = []
    seen_mnemonics: set[str] = set()
    for entry in mnemonic_entries:
        mnemonics.append(_mnemonic_record(entry, family_ids, seen_mnemonics))
    qualifiers_version, qualifier_entries, _qualifier_meta = _versioned(_TABLES / "qualifier_tokens.json")
    qualifiers: list[QualifierToken] = []
    seen_tokens: set[str] = set()
    for entry in qualifier_entries:
        qualifiers.append(_qualifier(entry, seen_tokens))
    prefixes_version, prefix_entries, _prefix_meta = _versioned(_TABLES / "vendor_prefixes.json")
    prefixes: list[VendorPrefix] = []
    seen_prefixes: set[str] = set()
    for entry in prefix_entries:
        prefixes.append(_prefix(entry, seen_prefixes))
    review = family_meta.get("review")
    if not isinstance(review, str) or review.strip() == "":
        raise ValueError("families.json metadata.review must be a non-empty string.")
    other = mnemonic_meta.get("review")
    if other != review:
        raise ValueError("Mnemonic and family review notes differ.")
    return ShippedTables(
        families_version=families_version,
        mnemonics_version=mnemonics_version,
        qualifiers_version=qualifiers_version,
        prefixes_version=prefixes_version,
        review_note=review,
        families=tuple(families),
        mnemonics=tuple(mnemonics),
        qualifiers=tuple(qualifiers),
        prefixes=tuple(prefixes),
    )


def default_local_tables_directory() -> Path | None:
    """Return %APPDATA%\\POLISH\\tables when APPDATA is set. D38.

    Does not create the directory. Returns None when APPDATA is unset, which
    is treated as a missing local table.
    """

    appdata = os.environ.get("APPDATA")
    if not isinstance(appdata, str) or appdata.strip() == "":
        return None
    return Path(appdata) / "POLISH" / "tables"


def load_local_mnemonics(directory: Path | None, family_ids: frozenset[str]) -> tuple[MnemonicRecord, ...]:
    """Load local mnemonic rows if ``directory`` already exists. D37, D38.

    A missing directory or a missing mnemonics.json is an empty tuple. This
    function does not create the directory. A stated family must be a shipped
    family id. The recorded tier is USER_DEFINED. A file that claims another
    tier is rejected.

    Parameters
    ----------
    directory:
        The local tables directory, or None when there is no app-data path.
    family_ids:
        Shipped family ids. A local family value outside this set is an error.
    """

    if directory is None or not directory.is_dir():
        return ()
    path = directory / _LOCAL_FILE_NAME
    if not path.is_file():
        return ()
    payload = _read_json(path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path.name} entries must be a list.")
    rows: list[MnemonicRecord] = []
    seen: set[str] = set()
    for entry in entries:
        row = _object(entry, "local mnemonic entry")
        unknown = set(row) - _LOCAL_KEYS
        if unknown:
            raise ValueError(f"Local mnemonic entry has unknown keys: {sorted(unknown)}.")
        mnemonic = _no_delimiter(_text(row.get("mnemonic"), "local mnemonic"), "local mnemonic")
        key = mnemonic.upper()
        if key in seen:
            raise ValueError(f"Duplicate local mnemonic {mnemonic!r}.")
        seen.add(key)
        family_value = row.get("family")
        family_id = _optional_text(family_value, f"{mnemonic} family")
        if family_id is not None and family_id not in family_ids:
            raise ValueError(f"Local mnemonic {mnemonic!r} names unknown family {family_id!r}.")
        tier = row.get("tier", "USER_DEFINED")
        if tier != "USER_DEFINED":
            raise ValueError(f"Local mnemonic {mnemonic!r} must be tier USER_DEFINED.")
        rows.append(
            MnemonicRecord(
                mnemonic=mnemonic,
                family_id=family_id,
                qualifiers=(),
                role=_role(row.get("role"), f"{mnemonic} role"),
                source=f"Local mnemonic table {path}",
                source_tier="USER_DEFINED",
                origin="local",
                user_label=_optional_text(row.get("user_label"), f"{mnemonic} user_label"),
                unit=_optional_text(row.get("unit"), f"{mnemonic} unit"),
                decision_id=_optional_text(row.get("decision_id"), f"{mnemonic} decision_id"),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class LocalDecisionRecord:
    """One rule saved for all wells that is not a mnemonic mapping. D92, OD-22.

    ``question_kind`` is a rule id, or ``pass_selection``. ``match_key`` is
    a family id. The row carries the decision that wrote it.
    """

    decision_id: str
    question_kind: str
    match_key: str
    chosen_option: str
    reason: str
    created: str


def load_local_decisions(directory: Path | None) -> tuple[LocalDecisionRecord, ...]:
    """Load saved all-wells decisions if ``directory`` already exists. D47, D92, OD-22.

    A missing directory or a missing local_decisions.json is an empty tuple.
    This function does not create the directory. Unknown keys are rejected.
    A repeated decision id, or a repeated question kind and match key, is
    rejected. Nothing is filled in when a field is absent.

    Parameters
    ----------
    directory:
        The local tables directory, or None when there is no app-data path.

    Returns
    -------
    tuple[LocalDecisionRecord, ...]
        The rows in file order.
    """

    if directory is None or not directory.is_dir():
        return ()
    path = directory / _LOCAL_DECISIONS_NAME
    if not path.is_file():
        return ()
    payload = _read_json(path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path.name} entries must be a list.")
    rows: list[LocalDecisionRecord] = []
    seen_ids: set[str] = set()
    seen_questions: set[tuple[str, str]] = set()
    for entry in entries:
        row = _object(entry, "local decision entry")
        unknown = set(row) - _LOCAL_DECISION_KEYS
        if unknown:
            raise ValueError(f"Local decision entry has unknown keys: {sorted(unknown)}.")
        decision_id = _text(row.get("decision_id"), "local decision_id")
        if decision_id in seen_ids:
            raise ValueError(f"Duplicate local decision {decision_id!r}.")
        seen_ids.add(decision_id)
        question_kind = _text(row.get("question_kind"), "local question_kind")
        match_key = _text(row.get("match_key"), "local match_key")
        question = (question_kind, match_key)
        if question in seen_questions:
            raise ValueError(
                f"Duplicate local decision for {question_kind!r} on {match_key!r}."
            )
        seen_questions.add(question)
        rows.append(
            LocalDecisionRecord(
                decision_id=decision_id,
                question_kind=question_kind,
                match_key=match_key,
                chosen_option=_text(row.get("chosen_option"), "local chosen_option"),
                reason=_text(row.get("reason"), "local reason"),
                created=_text(row.get("created"), "local created"),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class SpellingRecord:
    """One spelling of a unit. C1, D87.

    The same spelling may appear twice when it has two meanings. Those rows
    differ in ``dimension``. A conversion, when one exists, is a separate
    ``ConversionRecord`` and stays inside one dimension.
    """

    spelling: str
    canonical_unit: str
    dimension: str
    source: str
    source_tier: str


@dataclass(frozen=True)
class ConversionRecord:
    """Exact re-expression y = factor * x + offset. OD-05, C1.

    ``factor`` and ``offset`` are rational. The source names where the
    numbers come from. This record does not change stored samples.
    """

    from_unit: str
    to_unit: str
    factor: Fraction
    offset: Fraction
    dimension: str
    source: str
    source_tier: str


@dataclass(frozen=True)
class UnitSpellings:
    """The shipped spelling table and the conversions inside one dimension. C1, D87.

    ``case_conflicts`` lists pairs of spellings that differ only by case and
    do not mean the same units. Lookup keeps case only for those spellings.
    Everywhere else the match ignores case.
    """

    version: str
    spellings: tuple[SpellingRecord, ...]
    conversions: tuple[ConversionRecord, ...]
    case_conflicts: tuple[tuple[str, str], ...]

    def meanings(self, text: str) -> tuple[SpellingRecord, ...]:
        """Spellings that match ``text``. One entry per canonical unit and dimension.

        Case is ignored unless ``text`` is one of a pair that differs only by
        case and means different units.
        """

        if not isinstance(text, str):
            raise TypeError("A unit spelling must be a str.")
        conflicted = {name for pair in self.case_conflicts for name in pair}
        folded = text.casefold()
        if text in conflicted or any(row.spelling in conflicted and row.spelling.casefold() == folded for row in self.spellings):
            chosen = [row for row in self.spellings if row.spelling == text]
        else:
            chosen = [row for row in self.spellings if row.spelling.casefold() == folded]
        seen: set[tuple[str, str]] = set()
        unique: list[SpellingRecord] = []
        for row in chosen:
            key = (row.canonical_unit, row.dimension)
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return tuple(unique)

    def conversion(self, from_unit: str, to_unit: str, dimension: str) -> ConversionRecord | None:
        """The recorded factor from one canonical unit to another, or None.

        The inverse of a stored conversion is included. A missing conversion
        is not invented.
        """

        for record in self.conversions:
            if record.dimension == dimension and record.from_unit == from_unit and record.to_unit == to_unit:
                return record
        return None


def _fraction(value: object, what: str) -> Fraction:
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError(f"{what} must be an exact fraction written as numerator/denominator.")
    numerator, denominator = value.split("/")
    if numerator in {"", "+", "-"} or not _integer_text(numerator) or not _integer_text(denominator):
        raise ValueError(f"{what} must be an exact integer fraction.")
    try:
        return Fraction(int(numerator), int(denominator))
    except ZeroDivisionError as exc:
        raise ValueError(f"{what} has a zero denominator.") from exc


def _integer_text(text: str) -> bool:
    body = text[1:] if text.startswith(("+", "-")) else text
    return body.isdigit() and body != ""


def _case_conflicts(rows: tuple[SpellingRecord, ...]) -> tuple[tuple[str, str], ...]:
    grouped: dict[str, dict[str, set[tuple[str, str]]]] = {}
    for row in rows:
        written = grouped.setdefault(row.spelling.casefold(), {})
        written.setdefault(row.spelling, set()).add((row.canonical_unit, row.dimension))
    conflicts: list[tuple[str, str]] = []
    for written in grouped.values():
        names = sorted(written)
        if len(names) < 2:
            continue
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if written[left] != written[right]:
                    conflicts.append((left, right))
    return tuple(conflicts)


def _inverse(record: ConversionRecord) -> ConversionRecord:
    if record.factor == 0:
        raise ValueError(f"Conversion {record.from_unit} to {record.to_unit} has a zero factor.")
    return ConversionRecord(
        from_unit=record.to_unit,
        to_unit=record.from_unit,
        factor=1 / record.factor,
        offset=-(record.offset / record.factor),
        dimension=record.dimension,
        source=f"Inverse of the recorded conversion from {record.from_unit} to {record.to_unit}. {record.source}",
        source_tier=record.source_tier,
    )


def load_unit_spellings() -> UnitSpellings:
    """Read the shipped unit spellings. C1, D87, OD-05.

    Opened for reading only. A spelling without a source is rejected. A
    conversion that leaves its dimension, or that names a unit the table
    does not carry in that dimension, is rejected. The inverse of each
    stored conversion is added so a round trip uses the same exact factor.
    """

    path = _TABLES / "unit_spellings.json"
    payload = _read_json(path)
    version = _text(payload.get("version"), f"{path.name} version")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path.name} metadata must be an object when present.")
    spelling_rows = payload.get("spellings")
    conversion_rows = payload.get("conversions")
    if not isinstance(spelling_rows, list) or len(spelling_rows) == 0:
        raise ValueError("unit_spellings.json spellings must be a non-empty list.")
    if not isinstance(conversion_rows, list):
        raise ValueError("unit_spellings.json conversions must be a list.")
    if not isinstance(metadata.get("review"), str) or str(metadata.get("review")).strip() == "":
        raise ValueError("unit_spellings.json metadata.review must be a non-empty string.")
    spellings: list[SpellingRecord] = []
    seen_meaning: set[tuple[str, str, str]] = set()
    for entry in spelling_rows:
        row = _object(entry, "spelling entry")
        text = _text(row.get("spelling"), "spelling")
        canonical = _text(row.get("canonical_unit"), f"{text} canonical_unit")
        dimension = _text(row.get("dimension"), f"{text} dimension")
        key = (text, canonical, dimension)
        if key in seen_meaning:
            raise ValueError(f"Duplicate spelling {text!r} for {canonical} {dimension}.")
        seen_meaning.add(key)
        spellings.append(
            SpellingRecord(
                spelling=text,
                canonical_unit=canonical,
                dimension=dimension,
                source=_text(row.get("source"), f"{text} source"),
                source_tier=_text(row.get("source_tier"), f"{text} source_tier"),
            )
        )
    known = {(row.canonical_unit, row.dimension) for row in spellings}
    conversions: list[ConversionRecord] = []
    seen_edge: set[tuple[str, str, str]] = set()
    for entry in conversion_rows:
        row = _object(entry, "conversion entry")
        frm = _text(row.get("from_unit"), "conversion from_unit")
        to = _text(row.get("to_unit"), "conversion to_unit")
        dimension = _text(row.get("dimension"), "conversion dimension")
        if frm == to:
            raise ValueError(f"Conversion {frm} to itself is not a re-expression.")
        if (frm, dimension) not in known or (to, dimension) not in known:
            raise ValueError(f"Conversion {frm} to {to} is not inside dimension {dimension}.")
        edge = (frm, to, dimension)
        if edge in seen_edge:
            raise ValueError(f"Duplicate conversion {frm} to {to} in {dimension}.")
        seen_edge.add(edge)
        conversions.append(
            ConversionRecord(
                from_unit=frm,
                to_unit=to,
                factor=_fraction(row.get("factor"), f"{frm} to {to} factor"),
                offset=_fraction(row.get("offset"), f"{frm} to {to} offset"),
                dimension=dimension,
                source=_text(row.get("source"), f"{frm} to {to} source"),
                source_tier=_text(row.get("source_tier"), f"{frm} to {to} source_tier"),
            )
        )
    completed = list(conversions)
    for record in conversions:
        back = (record.to_unit, record.from_unit, record.dimension)
        if back in seen_edge:
            continue
        completed.append(_inverse(record))
        seen_edge.add(back)
    rows = tuple(spellings)
    return UnitSpellings(
        version=version,
        spellings=rows,
        conversions=tuple(completed),
        case_conflicts=_case_conflicts(rows),
    )
