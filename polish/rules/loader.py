"""Load one versioned rule entry from polish/rules. P09 scaffold.

P12 extends the library. This loader does not supply a missing scientific
value. A missing key or an unknown entry id is an error.

Structure
---------
load_rule
    Read the JSON file registered for an entry id and check its identity.
"""

from __future__ import annotations

import json
from pathlib import Path

# Entry id to file. Adding an entry is a new file plus a row here, not a default.
_ENTRY_FILES = {
    "depth.grid_rounding": "depth_grid_rounding.json",
    "depth.standard_steps": "depth_standard_steps.json",
}

_REQUIRED = ("id", "source", "source_tier")


def load_rule(entry_id: str) -> dict[str, object]:
    """Return the JSON object for ``entry_id``.

    Parameters
    ----------
    entry_id:
        ``depth.grid_rounding`` or ``depth.standard_steps``.

    Raises
    ------
    KeyError
        The id is not registered.
    ValueError
        The file's id, source, or source tier is missing or does not match.
    """

    if entry_id not in _ENTRY_FILES:
        raise KeyError(f"No rule file is registered for {entry_id!r}.")
    path = Path(__file__).resolve().parent / _ENTRY_FILES[entry_id]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object.")
    for key in _REQUIRED:
        if key not in payload:
            raise ValueError(f"{path.name} is missing {key!r}.")
        value = payload[key]
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"{path.name} {key} must be a non-empty string.")
    if payload["id"] != entry_id:
        raise ValueError(f"{path.name} id is {payload['id']!r}, not {entry_id!r}.")
    return payload
