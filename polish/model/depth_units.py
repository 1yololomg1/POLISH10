"""Depth-index spellings shared by Stages 1 and 3. D12.

Both stages read polish/rules/depth_standard_steps.json. That file is the
only list of which spellings are feet and which are metres for the depth
index. The general unit table is not a second list for this question.

Structure
---------
DepthSpelling
    One written spelling, its canonical unit, and the source of that spelling.
DepthUnitSpellings
    The feet and metre spellings from the rule file.
depth_unit_spellings
    Load that list. A missing source or a missing spelling is an error.
"""

from __future__ import annotations

from dataclasses import dataclass

from polish.rules.loader import load_rule

_RULE_ID = "depth.standard_steps"


@dataclass(frozen=True)
class DepthSpelling:
    """One depth-index spelling. D12.

    ``text`` is the spelling as written in the rule. ``canonical`` is the
    feet or metres spelling the question offers. ``family`` is ``feet`` or
    ``metres``. ``source`` names where this spelling comes from.
    """

    text: str
    canonical: str
    family: str
    source: str
    source_tier: str


@dataclass(frozen=True)
class DepthUnitSpellings:
    """The depth-index spellings. D12.

    ``feet_canonical`` and ``metre_canonical`` are the two options a missing
    depth unit offers. ``match`` returns one spelling, or None when the text
    is absent or belongs to both families.
    """

    spellings: tuple[DepthSpelling, ...]
    feet_canonical: str
    metre_canonical: str

    def match(self, unit: str) -> DepthSpelling | None:
        """The one spelling of ``unit``, or None when it is not exactly one family. D12."""

        if not isinstance(unit, str):
            raise TypeError("A depth unit must be a str.")
        folded = unit.strip().casefold()
        if folded == "":
            return None
        found = tuple(row for row in self.spellings if row.text.casefold() == folded)
        families = {row.family for row in found}
        if len(families) != 1:
            return None
        return found[0]

    def family_name(self, unit: str) -> str | None:
        """``feet`` or ``metres`` when ``unit`` is one depth spelling. D12."""

        found = self.match(unit)
        if found is None:
            return None
        return found.family

    def options(self) -> tuple[str, ...]:
        """The canonical feet spelling, then the canonical metres spelling. D12."""

        return (self.feet_canonical, self.metre_canonical)


def depth_unit_spellings() -> DepthUnitSpellings:
    """Load the depth-index spellings. D12.

    Opened through the rule loader. Each spelling carries its own source.
    A spelling without a source, or a canonical unit that is not one of its
    own spellings, is an error. Nothing is filled in.

    Returns
    -------
    DepthUnitSpellings
        Feet and metre spellings from ``depth.standard_steps``.
    """

    payload = load_rule(_RULE_ID)
    feet = _group(payload, "feet")
    metres = _group(payload, "metres")
    if feet[0] == metres[0]:
        raise ValueError(f"{_RULE_ID} feet and metres name the same canonical unit.")
    return DepthUnitSpellings(feet[1] + metres[1], feet[0], metres[0])


def _group(payload: dict[str, object], key: str) -> tuple[str, tuple[DepthSpelling, ...]]:
    if key not in payload:
        raise ValueError(f"{_RULE_ID} is missing {key!r}.")
    group = payload[key]
    if not isinstance(group, dict):
        raise ValueError(f"{_RULE_ID} {key} must be an object.")
    canonical = group.get("canonical")
    rows = group.get("spellings")
    if not isinstance(canonical, str) or canonical.strip() == "":
        raise ValueError(f"{_RULE_ID} {key}.canonical must be a non-empty string.")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{_RULE_ID} {key}.spellings must be a non-empty list.")
    family = key
    built: list[DepthSpelling] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError(f"{_RULE_ID} {key} spellings must be objects.")
        text = item.get("text")
        source = item.get("source")
        tier = item.get("source_tier")
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError(f"{_RULE_ID} {key} spelling text must be a non-empty string.")
        if not isinstance(source, str) or source.strip() == "":
            raise ValueError(f"{_RULE_ID} spelling {text!r} is missing its source.")
        if not isinstance(tier, str) or tier.strip() == "":
            raise ValueError(f"{_RULE_ID} spelling {text!r} is missing its source tier.")
        folded = text.casefold()
        if folded in seen:
            raise ValueError(f"{_RULE_ID} {key} repeats spelling {text!r}.")
        seen.add(folded)
        built.append(DepthSpelling(text, canonical, family, source, tier))
    if canonical.casefold() not in seen:
        raise ValueError(f"{_RULE_ID} {key}.canonical {canonical!r} is not one of its spellings.")
    return canonical, tuple(built)
