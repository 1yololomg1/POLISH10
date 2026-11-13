"""Curve colour by family. Colour encodes which curve and nothing else. D72.

The colours are the legacy table's ``industry_color`` values, copied one for one
from ``core/curve_identification.py`` (12-13 Nov 2026). The legacy family keys
are the family ids in ``polish/tables/families.json`` (P10), so every shipped
family has a colour. They agree with the design brief, section 4.3: gamma ray
green, bulk density red, neutron porosity blue, resistivity red.

``families.json`` leaves ``industry_color`` out on purpose (its
``legacy_fields_not_carried`` list): colour is presentation, so it lives here,
in the interface package, and the engine never reads it.

A curve with no family is drawn in the legacy default, black (``CurveInfo``
default in the same file). Provenance and validity must never be shown by
colour alone (D73); nothing here does that.
"""

from __future__ import annotations

SOURCE = "core/curve_identification.py industry_color (legacy LAS Visualizer table)"

NO_FAMILY_COLOUR = "#000000"

FAMILY_COLOURS: dict[str, str] = {
    "RESISTIVITY_DEEP": "#FF0000",
    "RESISTIVITY_MEDIUM": "#FF4444",
    "RESISTIVITY_SHALLOW": "#FF8888",
    "RESISTIVITY_MICRO": "#FFAAAA",
    "RESISTIVITY_LATEROLOG": "#FF0000",
    "GAMMA_RAY_TOTAL": "#008000",
    "GAMMA_RAY_SPECTRAL": "#228B22",
    "THORIUM": "#228B22",
    "URANIUM": "#228B22",
    "POTASSIUM": "#228B22",
    "NEUTRON_POROSITY": "#0000FF",
    "NEUTRON_COMPENSATED": "#0000FF",
    "NEUTRON_EPITHERMAL": "#0000FF",
    "NEUTRON_DUAL": "#0000FF",
    "BULK_DENSITY": "#FF0000",
    "PHOTOELECTRIC_FACTOR": "#FF00FF",
    "DENSITY_CORRECTION": "#AA0000",
    "SONIC_COMPRESSIONAL": "#800080",
    "SONIC_SHEAR": "#9932CC",
    "SONIC_STONELEY": "#8B008B",
    "SPONTANEOUS_POTENTIAL": "#FFA500",
    "CALIPER": "#000000",
    "DEPTH": "#000000",
    "NMR_POROSITY": "#4169E1",
    "NMR_PERMEABILITY": "#4169E1",
    "FORMATION_PRESSURE": "#FF4500",
    "FORMATION_TEMPERATURE": "#FF6347",
    "BOREHOLE_AZIMUTH": "#2F4F4F",
    "BOREHOLE_DEVIATION": "#2F4F4F",
    "FORMATION_RESISTIVITY_IMAGING": "#2F4F4F",
    "ACOUSTIC_IMAGING": "#2F4F4F",
    "CARBON_OXYGEN_RATIO": "#8B4513",
    "SILICON_CALCIUM_RATIO": "#8B4513",
}


def curve_colour(family_id: str | None) -> str:
    """The colour a curve is drawn in: its family's colour, or black when it has no
    family or a family this table does not list. D72."""

    if family_id is None:
        return NO_FAMILY_COLOUR
    return FAMILY_COLOURS.get(family_id, NO_FAMILY_COLOUR)
