"""Render finding evidence for the queue. UXC2, supplement section 4.1.

A curve reference, and a list of them, is shown as "col N MNEMONIC" from the
loaded dataset. DatasetId.parse reads the dataset part of a reference; there
is no CurveRef parser, so the column is the integer after the last "#", which
is the form CurveRef.__str__ writes. Lists show at most 20 items, then
"and N more". A source hash is never printed.

Structure
---------
render_evidence
    One line per evidence field, ready to escape into the queue detail.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from polish.model.ids import CurveRef, DatasetId

# How many list items are written before "and N more". A display limit, not a
# scientific threshold.
_LIST_LIMIT = 20
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HASH_IN_TEXT = re.compile(r"[0-9a-f]{64}")
_OMITTED_HASH = "source hash not shown"


def render_evidence(fields: tuple[tuple[str, object], ...], mnemonics: Mapping[int, str]) -> str:
    """Evidence lines for one finding. UXC2, supplement 4.1.

    Parameters
    ----------
    fields:
        The finding's evidence pairs, in stored order.
    mnemonics:
        Column index to the mnemonic stored on the loaded dataset.

    Returns
    -------
    str
        One "key: value" line per field, separated by newlines.
    """

    return "\n".join(f"{key}: {_format(value, mnemonics)}" for key, value in fields)


def _format(value: object, mnemonics: Mapping[int, str]) -> str:
    """One evidence value, with curve references and long lists reduced."""

    if isinstance(value, str):
        return _format_text(value, mnemonics)
    if isinstance(value, CurveRef):
        return _column_label(value, mnemonics)
    if isinstance(value, list):
        shown = [_format(item, mnemonics) for item in value[:_LIST_LIMIT]]
        extra = len(value) - len(shown)
        if extra:
            shown.append(f"and {extra} more")
        return ", ".join(shown)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_format(item, mnemonics)}" for key, item in value.items())
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_text(text: str, mnemonics: Mapping[int, str]) -> str:
    """A string value: a curve ref, a JSON list or object, or plain text."""

    sample = _sample_label(text, mnemonics)
    if sample is not None:
        return sample
    ref = _curve_ref(text)
    if ref is not None:
        return _column_label(ref, mnemonics)
    stripped = text.strip()
    if stripped[:1] in "[{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (list, dict)):
            return _format(parsed, mnemonics)
    if _HEX64.fullmatch(text):
        return _OMITTED_HASH
    return _HASH_IN_TEXT.sub(_OMITTED_HASH, _replace_curve_refs(text, mnemonics))


def _column_label(ref: CurveRef, mnemonics: Mapping[int, str]) -> str:
    """``col N MNEMONIC`` when this dataset has that column."""

    mnemonic = mnemonics.get(ref.column_index)
    if mnemonic is None:
        return f"col {ref.column_index}"
    return f"col {ref.column_index} {mnemonic}"


def _curve_ref(text: str) -> CurveRef | None:
    """Parse ``curve:{dataset}#{column}`` using DatasetId.parse. None if it is not one."""

    if not text.startswith("curve:"):
        return None
    dataset_text, mark, column_text = text[len("curve:") :].rpartition("#")
    if mark != "#" or column_text == "" or not column_text.isdecimal():
        return None
    try:
        dataset_id = DatasetId.parse(dataset_text)
    except (TypeError, ValueError):
        return None
    return CurveRef(dataset_id, int(column_text))


def _sample_label(text: str, mnemonics: Mapping[int, str]) -> str | None:
    """``sample:{curve ref}@{index}`` as written by SampleRef.__str__.

    There is no SampleRef parser. The curve half is parsed by ``_curve_ref``.
    """

    if not text.startswith("sample:"):
        return None
    curve_text, mark, index_text = text[len("sample:") :].rpartition("@")
    if mark != "@" or index_text == "" or not index_text.isdecimal():
        return None
    ref = _curve_ref(curve_text)
    if ref is None:
        return None
    return f"{_column_label(ref, mnemonics)} sample {int(index_text)}"


def _replace_curve_refs(text: str, mnemonics: Mapping[int, str]) -> str:
    """Replace curve references embedded in a longer string. The hash is not kept."""

    pieces: list[str] = []
    cursor = 0
    token = "curve:"
    while True:
        start = text.find(token, cursor)
        if start < 0:
            pieces.append(text[cursor:])
            return "".join(pieces)
        pieces.append(text[cursor:start])
        rest = text[start + len(token) :]
        split = rest.find("#")
        if split < 0:
            pieces.append(text[start:])
            return "".join(pieces)
        column_end = split + 1
        while column_end < len(rest) and rest[column_end].isdecimal():
            column_end += 1
        if column_end == split + 1:
            pieces.append(token)
            cursor = start + len(token)
            continue
        ref = _curve_ref(token + rest[:column_end])
        if ref is None:
            pieces.append(token)
            cursor = start + len(token)
            continue
        pieces.append(_column_label(ref, mnemonics))
        cursor = start + len(token) + column_end
