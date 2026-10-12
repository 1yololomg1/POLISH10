"""Load the shipped and local mnemonic tables. D31, D35, D37, D38, D39, D41, D42.

Structure
---------
FamilyRecord, MnemonicRecord, QualifierToken, VendorPrefix, ShippedTables
    The rows as read. No scientific field is filled in when the file omits it.
load_shipped_tables
    Read the four JSON files under polish/tables. Read-only.
default_local_tables_directory
    %APPDATA%\\POLISH\\tables . This function does not create it.
load_local_mnemonics
    Read mnemonics.json from a directory that already exists. Missing is empty.

A local row is tier USER_DEFINED. Saving local rows is not this module (P13).
Nothing here is sent off the machine (D38).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from polish.model.states import CurveRole

_TABLES = Path(__file__).resolve().parent
_LOCAL_FILE_NAME = "mnemonics.json"
_INTERPOLATION_DOMAINS = frozenset({"LINEAR", "LOG"})
_LOCAL_KEYS = frozenset({"mnemonic", "family", "user_label", "unit", "role", "tier"})
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
    """One shipped family. D31, D41, D42, F7.

    ``display_log_scale`` is a display attribute. It is not an interpolation
    domain. ``tool_group`` and ``interpolation_domain`` stay null where OD-13
    and OD-14 are unanswered. A family has no range and no unit list.
    """

    id: str
    display_name: str
    unit_dimension: str | None
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


def _family_record(entry: object, seen: set[str]) -> FamilyRecord:
    row = _object(entry, "family entry")
    family_id = _no_delimiter(_text(row.get("id"), "family id"), "family id")
    if family_id in seen:
        raise ValueError(f"Duplicate family id {family_id!r}.")
    seen.add(family_id)
    if "range" in row or "units" in row or "unit" in row:
        raise ValueError(f"Family {family_id} carries a range or a unit list.")
    dimension = row.get("unit_dimension")
    if dimension is not None:
        dimension = _text(dimension, f"{family_id} unit_dimension")
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
        unit_dimension=dimension,
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
            )
        )
    return tuple(rows)
