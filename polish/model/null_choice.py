"""A this-well NULL choice. D66.

This module imports neither polish.decisions nor polish.stages. Stage 0 and
the decisions layer both import it, so neither has to import the other to
share the choice record.

Structure
---------
NullChoice
    NONE, or one sentinel text the NULL question listed, and the decision
    that recorded it.
"""

from __future__ import annotations

from dataclasses import dataclass

# The option that means the file has no NULL. It is not a sentinel value.
NONE_OPTION = "NONE"


def _nonempty(text: str | None) -> bool:
    return isinstance(text, str) and text.strip() != ""


@dataclass(frozen=True)
class NullChoice:
    """One THIS_ITEM choice for a missing or non-finite NULL. D66.

    Stage 0 applies this only when the caller passes it. Re-running Stage 0
    is that call. ``option`` is NONE or a sentinel text the NULL question
    listed. ``decision_id`` is the decision that recorded it.

    Parameters
    ----------
    option:
        ``NONE``, or the first text of a sentinel value the question listed.
    decision_id:
        The decision that recorded the choice.
    """

    option: str
    decision_id: str

    def __post_init__(self) -> None:
        if not _nonempty(self.option):
            raise ValueError("A NULL choice requires an option.")
        if not _nonempty(self.decision_id):
            raise ValueError("A NULL choice requires a decision id.")

    @property
    def converts_nothing(self) -> bool:
        """True when the user said the file has no NULL."""

        return self.option == NONE_OPTION
