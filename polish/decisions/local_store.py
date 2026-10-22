"""Write a local table row without rewriting the others. D38, D91, D92, OD-22.

Structure
---------
MnemonicWrite
    Whether a mnemonic row was inserted, replaced, or left as it was.
update_mnemonic
    One USER_DEFINED row in mnemonics.json, keyed by mnemonic.
append_local_decision
    One row in local_decisions.json.
tables_directory
    %APPDATA%\\POLISH\\tables, or the injected root's POLISH\\tables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from polish.tables.loader import LOCAL_DECISIONS_FILE, LOCAL_MNEMONICS_FILE


@dataclass(frozen=True)
class MnemonicWrite:
    """What ``update_mnemonic`` did to the one row. D91.

    ``previous_decision_id`` is the decision that wrote the row being
    replaced. It is None when nothing was replaced.
    """

    status: str
    previous_decision_id: str | None


def tables_directory(app_data: Path) -> Path:
    """Return the local tables directory under an app-data root. D38.

    Does not create it. ``app_data`` is the injected root, the stand-in for
    %APPDATA%. The POLISH folder is a child of that root.
    """

    if not isinstance(app_data, Path):
        raise TypeError("app_data must be a Path")
    return app_data / "POLISH" / "tables"


def saved_mnemonic(directory: Path, mnemonic: str) -> dict[str, object] | None:
    """The saved row for this mnemonic, compared without regard to case. D91.

    None when the file or the row is absent. The file is not created.
    """

    path = directory / LOCAL_MNEMONICS_FILE
    if not path.is_file():
        return None
    entries = _entries(path.read_text(encoding="utf-8"), path.name)
    key = mnemonic.upper()
    for entry in entries:
        saved = entry.get("mnemonic")
        if isinstance(saved, str) and saved.upper() == key:
            return entry
    return None


def update_mnemonic(directory: Path, row: dict[str, str], *, confirm: bool) -> MnemonicWrite:
    """Insert or replace one mnemonic row. D91, OD-22.

    The file is a JSON object with an ``entries`` list. Other top-level keys
    and every other row are left byte for byte. A mnemonic already saved with
    the same family and the same role is not written. A different family, or
    the same family with a different role, is refused unless ``confirm`` is
    true, and then that one row is replaced.

    Parameters
    ----------
    directory:
        The local tables directory. Created when a row is written.
    row:
        The new row. ``mnemonic`` and ``family`` are required.
    confirm:
        True only when the caller explicitly confirms replacing a saved family.

    Returns
    -------
    MnemonicWrite
        ``unchanged``, ``inserted``, or ``replaced``.

    Raises
    ------
    ValueError
        The saved family differs and ``confirm`` is false, or the file is
        not the expected object.
    """

    mnemonic = row.get("mnemonic")
    family = row.get("family")
    if not isinstance(mnemonic, str) or mnemonic.strip() == "":
        raise ValueError("A local mnemonic row needs the mnemonic as spelled in the file.")
    if not isinstance(family, str) or family.strip() == "":
        raise ValueError("A local mnemonic row needs a family.")
    path = directory / LOCAL_MNEMONICS_FILE
    if not path.is_file():
        _write_new(directory, path, row)
        return MnemonicWrite("inserted", None)
    text = path.read_text(encoding="utf-8")
    entries = _entries(text, path.name)
    key = mnemonic.upper()
    match = None
    for index, entry in enumerate(entries):
        saved = entry.get("mnemonic")
        if isinstance(saved, str) and saved.upper() == key:
            match = index
            break
    if match is None:
        _splice(path, text, None, row)
        return MnemonicWrite("inserted", None)
    saved_family = entries[match].get("family")
    saved_role = entries[match].get("role")
    if saved_family == family and saved_role == row.get("role"):
        return MnemonicWrite("unchanged", _decision_id(entries[match]))
    if not confirm:
        if saved_family != family:
            shown = saved_family if isinstance(saved_family, str) and saved_family.strip() else "no family"
            raise ValueError(
                f"{mnemonic} is already saved for all wells as {shown}. Change that rule for all wells?"
            )
        shown_role = saved_role if isinstance(saved_role, str) and saved_role.strip() else "no role"
        raise ValueError(
            f"{mnemonic} is already saved for all wells as {family} with role {shown_role}. "
            "Change that rule for all wells?"
        )
    previous = _decision_id(entries[match])
    _splice(path, text, match, row)
    return MnemonicWrite("replaced", previous)


def append_local_decision(directory: Path, row: dict[str, str], *, confirm: bool = False) -> MnemonicWrite:
    """Insert or replace one all-wells decision. D92, OD-22.

    The same question kind and match key with the same chosen option is not
    written. A different chosen option is refused unless ``confirm`` is true,
    and then that one row is replaced. Every other row stays byte for byte.
    A repeated decision id is refused. This does not open a session file.

    Parameters
    ----------
    directory:
        The local tables directory. Created when a row is written.
    row:
        decision_id, question_kind, match_key, chosen_option, reason, created.
    confirm:
        True only when the caller explicitly confirms replacing a saved answer.

    Returns
    -------
    MnemonicWrite
        ``unchanged``, ``inserted``, or ``replaced``.

    Raises
    ------
    ValueError
        A field is missing, the decision id is repeated, or the saved answer
        differs and ``confirm`` is false.
    """

    for key in ("decision_id", "question_kind", "match_key", "chosen_option", "reason", "created"):
        value = row.get(key)
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"A local decision needs {key}.")
    path = directory / LOCAL_DECISIONS_FILE
    if not path.is_file():
        _write_new(directory, path, row)
        return MnemonicWrite("inserted", None)
    text = path.read_text(encoding="utf-8")
    entries = _entries(text, path.name)
    match = None
    for index, entry in enumerate(entries):
        if entry.get("decision_id") == row["decision_id"]:
            raise ValueError(f"Duplicate local decision {row['decision_id']!r}.")
        if entry.get("question_kind") == row["question_kind"] and entry.get("match_key") == row["match_key"]:
            match = index
    if match is None:
        _splice(path, text, None, row)
        return MnemonicWrite("inserted", None)
    previous = _decision_id(entries[match])
    if entries[match].get("chosen_option") == row["chosen_option"]:
        return MnemonicWrite("unchanged", previous)
    if not confirm:
        raise ValueError(
            f"A rule for {row['question_kind']} on {row['match_key']} is already saved for all wells "
            f"as {entries[match].get('chosen_option')}. Change that rule for all wells?"
        )
    _splice(path, text, match, row)
    return MnemonicWrite("replaced", previous)


def _decision_id(entry: dict[str, object]) -> str | None:
    value = entry.get("decision_id")
    if isinstance(value, str) and value.strip() != "":
        return value
    return None


def _entries(text: str, name: str) -> list[dict[str, object]]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object.")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{name} entries must be a list.")
    rows: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{name} entries must be objects.")
        rows.append(entry)
    return rows


def _write_new(directory: Path, path: Path, row: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"entries": [row]}, indent=2, ensure_ascii=False) + "\n"
    _commit(path, text)


def _splice(path: Path, text: str, index: int | None, row: dict[str, str]) -> None:
    bracket, spans = _entry_spans(text)
    encoded = json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
    if index is None:
        if not spans:
            updated = text[: bracket + 1] + encoded + text[bracket + 1 :]
        else:
            end = spans[-1][1]
            updated = text[:end] + ", " + encoded + text[end:]
    else:
        start, end = spans[index]
        updated = text[:start] + encoded + text[end:]
    json.loads(updated)
    _commit(path, updated)


def _entry_spans(text: str) -> tuple[int, list[tuple[int, int]]]:
    """The entries array bracket and each element's exact slice. D91."""

    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length and text[index] in " \t\r\n":
        index += 1
    if index >= length or text[index] != "{":
        raise ValueError("A local table must be a JSON object.")
    index += 1
    bracket: int | None = None
    while index < length:
        while index < length and text[index] in " \t\r\n,":
            index += 1
        if index < length and text[index] == "}":
            break
        key, index = decoder.raw_decode(text, index)
        while index < length and text[index] in " \t\r\n":
            index += 1
        if index >= length or text[index] != ":":
            raise ValueError("A local table key has no value.")
        index += 1
        while index < length and text[index] in " \t\r\n":
            index += 1
        if key == "entries":
            if index >= length or text[index] != "[":
                raise ValueError("entries must be a list.")
            bracket = index
            break
        _value, index = decoder.raw_decode(text, index)
    if bracket is None:
        raise ValueError("A local table has no entries list.")
    spans: list[tuple[int, int]] = []
    cursor = bracket + 1
    while cursor < length:
        while cursor < length and text[cursor] in " \t\r\n,":
            cursor += 1
        if cursor < length and text[cursor] == "]":
            return bracket, spans
        _row, end = decoder.raw_decode(text, cursor)
        spans.append((cursor, end))
        cursor = end
    raise ValueError("The entries list is not closed.")


def _commit(path: Path, text: str) -> None:
    """Replace ``path`` only after the new text is valid JSON. D91."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    try:
        os.replace(temporary, path)
    except OSError:
        if temporary.exists():
            temporary.unlink()
        raise
