"""A this-well unit choice. D93.

This module imports neither polish.decisions nor polish.stages. Stage 3 and
the decisions layer both import it, so neither has to import the other to
share the choice record.

Structure
---------
UnitChoice
    One curve, the canonical unit the user chose, and the decision that
    recorded it.
"""

from __future__ import annotations

from dataclasses import dataclass

from polish.model.ids import CurveRef


def _nonempty(text: str | None) -> bool:
    return isinstance(text, str) and text.strip() != ""


@dataclass(frozen=True)
class UnitChoice:
    """One THIS_ITEM unit choice already recorded for a curve. D93.

    Stage 3 applies this only when the caller passes it. Re-running Stage 3
    is that call. ``canonical_unit`` is the option the unit question listed.
    ``decision_id`` is the decision that recorded it.

    Parameters
    ----------
    curve:
        The curve the decision was about.
    canonical_unit:
        A canonical unit from that curve's unit question.
    decision_id:
        The decision that recorded the choice.
    """

    curve: CurveRef
    canonical_unit: str
    decision_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.curve, CurveRef):
            raise TypeError("curve must be a CurveRef")
        if not _nonempty(self.canonical_unit):
            raise ValueError("A unit choice requires a canonical unit.")
        if not _nonempty(self.decision_id):
            raise ValueError("A unit choice requires a decision id.")
