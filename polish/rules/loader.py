"""Load one versioned rule entry from polish/rules. P09 scaffold, P12 library.

This loader does not supply a missing scientific value. A missing key or an
unknown entry id is an error. Validity rules are checked for the D87 fields
when ``validity.rules`` is loaded. No field is filled in.

Structure
---------
load_rule
    Read the JSON file registered for an entry id and check its identity.
validity_rules
    The Stage 4 library, already checked.
validity_rule
    One rule by id. An unknown id is an error.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# Entry id to file. Adding an entry is a new file plus a row here, not a default.
_ENTRY_FILES = {
    "depth.grid_rounding": "depth_grid_rounding.json",
    "depth.standard_steps": "depth_standard_steps.json",
    "validity.rules": "validity_rules.json",
}

_REQUIRED = ("id", "source", "source_tier")

# D87 fields on every validity rule. ``id`` is the loader's entry key.
_RULE_KEYS = (
    "id",
    "family",
    "accepted_units",
    "canonical_unit",
    "rule_type",
    "bound_values",
    "environmental_qualifiers",
    "source",
    "source_tier",
    "library_version",
    "violation_effect",
    "overridable",
    "override_recorded_as",
)

_RULE_TYPES = frozenset({"DOMAIN_CONSTRAINT", "HARD_IMPOSSIBILITY", "PLAUSIBILITY", "CONSISTENCY"})
_EFFECTS = frozenset({"INVALID", "WARNING", "REFUSAL"})
# Stated in P12. A different pairing is a bad file, not a guess the loader repairs.
_EFFECT_FOR_TYPE = {
    "DOMAIN_CONSTRAINT": "INVALID",
    "HARD_IMPOSSIBILITY": "INVALID",
    "PLAUSIBILITY": "WARNING",
}
_LIBRARY_VERSION = "1"
_RANGE_BOUND_KEYS = frozenset({"minimum", "maximum", "inclusivity"})
_LOG_BOUND_KEYS = frozenset({"minimum", "maximum", "inclusivity", "outside_fraction_more_than"})
_NONFINITE_BOUND_KEYS = frozenset({"condition"})


def load_rule(entry_id: str) -> dict[str, object]:
    """Return the JSON object for ``entry_id``.

    Parameters
    ----------
    entry_id:
        ``depth.grid_rounding``, ``depth.standard_steps``, or ``validity.rules``.

    Raises
    ------
    KeyError
        The id is not registered.
    ValueError
        The file's id, source, or source tier is missing or does not match.
        A validity rule is missing a D87 field, or a field has the wrong shape.
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
    if entry_id == "validity.rules":
        _validate_validity_library(payload)
    return payload


@dataclass(frozen=True)
class BoundValues:
    """Bounds as written in the rule file. D87.

    ``outside_fraction_more_than`` is set only on the log-level SP rule.
    It is a rational, so "more than 20%" is ``outside / non_absent > 1/5``
    with no float cutoff. ``condition`` is set only on the non-finite rule.
    """

    minimum: float | None
    maximum: float | None
    inclusivity: str | None
    outside_fraction_more_than: Fraction | None
    condition: str | None


@dataclass(frozen=True)
class ValidityRule:
    """One D87 rule. Strings are the library tokens, not a second state enum."""

    id: str
    family: str | None
    accepted_units: tuple[str, ...]
    canonical_unit: str | None
    rule_type: str
    bound_values: BoundValues
    environmental_qualifiers: tuple[str, ...]
    source: str
    source_tier: str
    library_version: str
    violation_effect: str
    overridable: bool
    override_recorded_as: str

    @property
    def log_level(self) -> bool:
        """True when the rule flags a log by a fraction, not each sample."""

        return self.bound_values.outside_fraction_more_than is not None


def validity_rules() -> tuple[ValidityRule, ...]:
    """Every rule in ``validity.rules``, in file order. D87, D11, P12.

    Raises
    ------
    ValueError
        The library is missing a D87 field or a field has the wrong shape.
    """

    payload = load_rule("validity.rules")
    rows = payload["rules"]
    if not isinstance(rows, list):
        raise ValueError("validity.rules rules must be a list.")
    return tuple(_validity_rule(row) for row in rows)


def validity_rule(rule_id: str) -> ValidityRule:
    """One rule by id.

    Raises
    ------
    KeyError
        No rule has this id.
    """

    if not isinstance(rule_id, str) or rule_id.strip() == "":
        raise KeyError(f"No validity rule is registered for {rule_id!r}.")
    for rule in validity_rules():
        if rule.id == rule_id:
            return rule
    raise KeyError(f"No validity rule is registered for {rule_id!r}.")


def _validate_validity_library(payload: dict[str, object]) -> None:
    """Reject a validity file that omits a D87 field. Nothing is filled in."""

    if "library_version" not in payload:
        raise ValueError("validity_rules.json is missing 'library_version'.")
    if payload["library_version"] != _LIBRARY_VERSION:
        raise ValueError("validity_rules.json library_version must be '1'.")
    if "rules" not in payload:
        raise ValueError("validity_rules.json is missing 'rules'.")
    rows = payload["rules"]
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("validity_rules.json rules must be a non-empty list.")
    seen: set[str] = set()
    for row in rows:
        rule = _validity_rule(row)
        if rule.id in seen:
            raise ValueError(f"Duplicate validity rule id {rule.id!r}.")
        seen.add(rule.id)


def _validity_rule(row: object) -> ValidityRule:
    if not isinstance(row, dict):
        raise ValueError("A validity rule must be a JSON object.")
    extra = set(row) - set(_RULE_KEYS)
    if extra:
        raise ValueError(f"Validity rule has unexpected keys: {sorted(extra)}.")
    for key in _RULE_KEYS:
        if key not in row:
            raise ValueError(f"Validity rule is missing {key!r}.")
    rule_id = _text(row["id"], "validity rule id")
    family = _optional_text(row["family"], f"{rule_id} family")
    accepted = _unit_list(row["accepted_units"], f"{rule_id} accepted_units")
    canonical = _optional_text(row["canonical_unit"], f"{rule_id} canonical_unit")
    rule_type = _choice(row["rule_type"], _RULE_TYPES, f"{rule_id} rule_type")
    effect = _choice(row["violation_effect"], _EFFECTS, f"{rule_id} violation_effect")
    expected = _EFFECT_FOR_TYPE.get(rule_type)
    if expected is not None and effect != expected:
        raise ValueError(f"{rule_id} violation_effect must be {expected} for {rule_type}.")
    if rule_type == "CONSISTENCY":
        raise ValueError(f"{rule_id} is CONSISTENCY. No consistency rule is shipped, and none is applied.")
    bounds = _bounds(row["bound_values"], rule_id, family is None)
    qualifiers = _text_list(row["environmental_qualifiers"], f"{rule_id} environmental_qualifiers")
    source = _text(row["source"], f"{rule_id} source")
    source_tier = _text(row["source_tier"], f"{rule_id} source_tier")
    version = _text(row["library_version"], f"{rule_id} library_version")
    if version != _LIBRARY_VERSION:
        raise ValueError(f"{rule_id} library_version must be '1'.")
    overridable = row["overridable"]
    if type(overridable) is not bool:
        raise ValueError(f"{rule_id} overridable must be a bool.")
    recorded = _text(row["override_recorded_as"], f"{rule_id} override_recorded_as")
    if family is None:
        if canonical is not None or accepted:
            raise ValueError(f"{rule_id} has no family, so it has no unit.")
        if bounds.condition != "non_finite":
            raise ValueError(f"{rule_id} has no family and is not the non-finite rule.")
    else:
        if canonical is None:
            raise ValueError(f"{rule_id} needs a canonical unit.")
        if canonical not in accepted:
            raise ValueError(f"{rule_id} accepted_units must include {canonical}.")
        if bounds.condition is not None:
            raise ValueError(f"{rule_id} is a family rule and cannot use condition.")
    return ValidityRule(
        id=rule_id,
        family=family,
        accepted_units=accepted,
        canonical_unit=canonical,
        rule_type=rule_type,
        bound_values=bounds,
        environmental_qualifiers=qualifiers,
        source=source,
        source_tier=source_tier,
        library_version=version,
        violation_effect=effect,
        overridable=overridable,
        override_recorded_as=recorded,
    )


def _bounds(value: object, rule_id: str, non_finite: bool) -> BoundValues:
    if not isinstance(value, dict):
        raise ValueError(f"{rule_id} bound_values must be a JSON object.")
    keys = set(value)
    if non_finite:
        if keys != _NONFINITE_BOUND_KEYS:
            raise ValueError(f"{rule_id} bound_values must be exactly condition for a non-finite rule.")
        if value["condition"] != "non_finite":
            raise ValueError(f"{rule_id} condition must be 'non_finite'.")
        return BoundValues(None, None, None, None, "non_finite")
    fraction_key = "outside_fraction_more_than"
    if fraction_key in keys:
        if keys != _LOG_BOUND_KEYS:
            raise ValueError(f"{rule_id} log-level bound_values has unexpected or missing keys.")
        fraction: Fraction | None = _ratio(value[fraction_key], f"{rule_id} outside_fraction_more_than")
    else:
        if keys != _RANGE_BOUND_KEYS:
            raise ValueError(f"{rule_id} bound_values must state minimum, maximum, and inclusivity.")
        fraction = None
    if value["inclusivity"] != "inclusive":
        raise ValueError(f"{rule_id} inclusivity must be 'inclusive'. Exclusive bounds are not shipped.")
    minimum = _real(value["minimum"], f"{rule_id} minimum")
    maximum = _real(value["maximum"], f"{rule_id} maximum")
    if minimum > maximum:
        raise ValueError(f"{rule_id} minimum is above maximum.")
    return BoundValues(minimum, maximum, "inclusive", fraction, None)


def _ratio(value: object, what: str) -> Fraction:
    """A rational ``n/d``. The SP threshold is stored this way so 20% is exact."""

    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError(f"{what} must be a rational written as n/d.")
    left, right = value.split("/")
    if left in ("", "-", "+") or not left.lstrip("+-").isdigit() or not right.isdigit() or int(right) == 0:
        raise ValueError(f"{what} must be a rational written as n/d.")
    fraction = Fraction(int(left), int(right))
    if fraction <= 0 or fraction >= 1:
        raise ValueError(f"{what} must be greater than 0 and less than 1.")
    return fraction


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite.")
    return number


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{what} must be a non-empty string.")
    return value


def _optional_text(value: object, what: str) -> str | None:
    if value is None:
        return None
    return _text(value, what)


def _choice(value: object, allowed: frozenset[str], what: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{what} must be one of {sorted(allowed)}.")
    return value


def _unit_list(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{what} must be a list.")
    units: list[str] = []
    for item in value:
        units.append(_text(item, what))
    if len(set(units)) != len(units):
        raise ValueError(f"{what} contains a duplicate unit.")
    return tuple(units)


def _text_list(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{what} must be a list.")
    return tuple(_text(item, what) for item in value)
