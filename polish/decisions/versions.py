"""Versions of the rule library and the shipped tables. D63, UXC4.

Structure
---------
versions_in_force
    What the loaders report, as (name, version) pairs.
"""

from __future__ import annotations

from polish.rules.loader import validity_rules
from polish.tables.loader import load_shipped_tables, load_unit_spellings

# Names are the loader's table, not a second versioning scheme.
_VERSION_NAMES = (
    "families",
    "mnemonics",
    "qualifier_tokens",
    "unit_spellings",
    "validity_rules",
    "vendor_prefixes",
)


def versions_in_force() -> tuple[tuple[str, str], ...]:
    """Return the rule-library version and every shipped table version. D63.

    The pairs are the versions ``load_shipped_tables``, ``load_unit_spellings``,
    and ``validity_rules`` report. A local file has no loader version and is
    not included. Nothing is filled in when a loader omits a version: the
    loader raises instead.

    Returns
    -------
    tuple[tuple[str, str], ...]
        ``(name, version)`` in a stable order.
    """

    shipped = load_shipped_tables()
    spellings = load_unit_spellings()
    rules = validity_rules()
    if not rules:
        raise ValueError("The validity library has no rules.")
    library = rules[0].library_version
    if any(rule.library_version != library for rule in rules):
        raise ValueError("Validity rules do not share one library version.")
    pairs = (
        ("families", shipped.families_version),
        ("mnemonics", shipped.mnemonics_version),
        ("qualifier_tokens", shipped.qualifiers_version),
        ("unit_spellings", spellings.version),
        ("validity_rules", library),
        ("vendor_prefixes", shipped.prefixes_version),
    )
    if tuple(name for name, _version in pairs) != _VERSION_NAMES:
        raise ValueError("Version names are not the loader set.")
    return pairs
