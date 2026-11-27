"""Append-only snapshot of the state-schema string values. D61.

Adding a value is allowed only by updating tests/model/states_snapshot.json in
the same commit. Removing or renaming an existing value fails with a message
that cites D61. This test writes nothing.
"""

import inspect
import json
from pathlib import Path

from polish.model import states

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "states_snapshot.json"


def _current_pairs():
    pairs = set()
    for name, obj in inspect.getmembers(states, inspect.isclass):
        if obj.__module__ != states.__name__ or not issubclass(obj, states.Enum):
            continue
        for member in obj:
            pairs.add((name, member.value))
    return pairs


def _snapshot_pairs():
    payload = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return {tuple(pair) for pair in payload["pairs"]}


def test_enum_values_match_the_committed_snapshot():
    current = _current_pairs()
    snapshot = _snapshot_pairs()
    removed = snapshot - current
    added = current - snapshot
    assert not removed, (
        "Existing state-schema value removed or renamed. "
        "D61: the code table is append-only; removing or renaming a value is not allowed. "
        f"Missing from code: {sorted(removed)}"
    )
    assert not added, (
        "New state-schema values are not in tests/model/states_snapshot.json. "
        "Adding values is allowed only by updating the snapshot in the same commit. "
        f"Not in snapshot: {sorted(added)}"
    )
