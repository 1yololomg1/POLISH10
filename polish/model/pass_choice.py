"""A recorded pass choice. D94.

This module imports neither polish.decisions nor polish.stages. Stage 2 and
the decisions layer both import it, so neither has to import the other to
share the choice record.

Structure
---------
PassChoice
    One family, the pass the user chose, and the decision that recorded it.
"""

from __future__ import annotations

from dataclasses import dataclass


def _nonempty(text: str | None) -> bool:
    return isinstance(text, str) and text.strip() != ""


@dataclass(frozen=True)
class PassChoice:
    """One recorded pass choice for a family. D94.

    Stage 2 applies this only when the caller passes it. Re-running Stage 2
    is that call. ``pass_name`` is an option the pass question listed, such
    as "main pass" or "repeat pass". ``decision_id`` is the decision that
    recorded it. Scope is not stored here: a this-well answer and an
    all-wells answer are the same choice once they have been recorded.

    Parameters
    ----------
    family_id:
        The family the pass question was about.
    pass_name:
        The pass named by that question's options.
    decision_id:
        The decision that recorded the choice.
    """

    family_id: str
    pass_name: str
    decision_id: str

    def __post_init__(self) -> None:
        if not _nonempty(self.family_id):
            raise ValueError("A pass choice requires a family id.")
        if not _nonempty(self.pass_name):
            raise ValueError("A pass choice requires a pass name.")
        if not _nonempty(self.decision_id):
            raise ValueError("A pass choice requires a decision id.")
