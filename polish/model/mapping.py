"""A this-well mapping, and the roles a family's shipped names state. D91.

This module imports neither polish.decisions nor polish.stages. Stage 2 and
the decisions layer both import it, so neither has to import the other to
share the mapping record.

Structure
---------
ExplicitMapping
    One curve, the family it behaves as, the role of that family, and the
    optional label.
roles_stated_by_family
    The roles that family's shipped mnemonic rows state, in table order.
    The caller passes the shipped tables. This function does not load them.
"""

from __future__ import annotations

from dataclasses import dataclass

from polish.model.ids import CurveRef
from polish.model.states import CurveRole
from polish.tables.loader import ShippedTables


def _nonempty(text: str | None) -> bool:
    return isinstance(text, str) and text.strip() != ""


def roles_stated_by_family(shipped: ShippedTables, family_id: str) -> tuple[CurveRole, ...]:
    """Roles stated by this family's rows in the shipped mnemonic table. D91.

    Table order. A missing role and UNKNOWN are skipped. Nothing here names
    a family, and nothing here reads the table files. The caller passes the
    shipped tables it already loaded.

    Parameters
    ----------
    shipped:
        The shipped tables, including their mnemonic rows.
    family_id:
        A family id, compared as the table stores it.

    Returns
    -------
    tuple[CurveRole, ...]
        The distinct roles, in the order those rows state them.
    """

    if not isinstance(shipped, ShippedTables):
        raise TypeError("shipped must be the shipped tables")
    stated: list[CurveRole] = []
    for row in shipped.mnemonics:
        if row.family_id != family_id or row.role is None or row.role is CurveRole.UNKNOWN:
            continue
        if row.role not in stated:
            stated.append(row.role)
    return tuple(stated)


@dataclass(frozen=True)
class ExplicitMapping:
    """One THIS_ITEM mapping already recorded for a curve. D91, UXC4.

    Stage 2 applies this only when the caller passes it. Re-running Stage 2
    is that call. The curve keeps its mnemonic. ``user_label`` is the
    optional name. The family is required. ``role`` is the one role that
    family's shipped mnemonics state, or the role the user chose when they
    do not share one. It is never UNKNOWN.

    Parameters
    ----------
    curve:
        The curve the decision was about.
    family_id:
        A shipped family id.
    role:
        A curve role stated by that family's shipped mnemonics.
    user_label:
        The user's name for the curve, or None.
    decision_id:
        The decision that recorded this mapping, or None when the caller
        built the mapping without one. D98.
    """

    curve: CurveRef
    family_id: str
    role: CurveRole
    user_label: str | None = None
    decision_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.curve, CurveRef):
            raise TypeError("curve must be a CurveRef")
        if not _nonempty(self.family_id):
            raise ValueError("An explicit mapping requires a family id.")
        if not isinstance(self.role, CurveRole):
            raise TypeError("role must be a CurveRole")
        if self.role is CurveRole.UNKNOWN:
            raise ValueError("A mapped curve cannot have role UNKNOWN.")
        if self.user_label is not None and not _nonempty(self.user_label):
            raise ValueError("user_label must be non-empty text or None")
        if self.decision_id is not None and not _nonempty(self.decision_id):
            raise ValueError("decision_id must be a non-empty string or None")
