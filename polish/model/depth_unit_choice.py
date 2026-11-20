"""A this-well depth-unit choice. D86.

This module imports neither polish.decisions nor polish.stages. Stage 1 and
the decisions layer both import it, so neither has to import the other to
share the choice record.

Structure
---------
DepthUnitChoice
    The canonical depth unit the user chose, and the decision that recorded it.
"""

from __future__ import annotations

from dataclasses import dataclass


def _nonempty(text: str | None) -> bool:
    return isinstance(text, str) and text.strip() != ""


@dataclass(frozen=True)
class DepthUnitChoice:
    """One THIS_ITEM choice for a missing or unrecognised depth unit. D86.

    Stage 1 applies this only when the caller passes it. Re-running Stage 1
    is that call. ``unit`` is a canonical spelling the depth-unit question
    listed. ``decision_id`` is the decision that recorded it.

    Parameters
    ----------
    unit:
        The canonical feet or metres spelling the question listed.
    decision_id:
        The decision that recorded the choice.
    """

    unit: str
    decision_id: str

    def __post_init__(self) -> None:
        if not _nonempty(self.unit):
            raise ValueError("A depth-unit choice requires a unit.")
        if not _nonempty(self.decision_id):
            raise ValueError("A depth-unit choice requires a decision id.")
