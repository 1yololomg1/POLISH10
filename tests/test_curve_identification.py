"""
Regression tests for the unified CurveIdentificationEngine.

Guards:
- Single mnemonic database (no split engines)
- No mnemonic string appears under more than one curve family
- Provenance `source` field present on every entry
- Demo LAS mnemonics resolve correctly
"""

from __future__ import annotations

from collections import defaultdict

from core.curve_identification import (
    CurveIdentificationEngine,
    ComprehensiveCurveManager,
    ComprehensiveMnemonicLibrary,
    build_mnemonic_database,
)


def test_aliases_resolve_to_same_engine_class():
    assert ComprehensiveMnemonicLibrary is CurveIdentificationEngine
    assert ComprehensiveCurveManager is CurveIdentificationEngine


def test_no_mnemonic_appears_under_more_than_one_curve_family():
    """Regression guard: a mnemonic must map to exactly one curve family."""
    db = build_mnemonic_database()
    mnemonic_to_families = defaultdict(set)

    for curve_type, entry in db.items():
        family = entry.get('curve_family', 'unknown')
        for mnemonic in entry.get('mnemonics', []):
            mnemonic_to_families[mnemonic.upper()].add(family)

    conflicts = {
        mnemonic: sorted(families)
        for mnemonic, families in mnemonic_to_families.items()
        if len(families) > 1
    }
    assert conflicts == {}, f"Mnemonics mapped to multiple families: {conflicts}"


def test_every_entry_has_nonempty_source():
    db = build_mnemonic_database()
    missing = [
        curve_type
        for curve_type, entry in db.items()
        if not str(entry.get('source', '')).strip()
    ]
    assert missing == [], f"Entries missing source provenance: {missing}"


def test_no_concatenated_combo_mnemonics_in_database():
    """Combo strings like RLLD_HRLA_AT90_AHT90 must not re-enter the table."""
    db = build_mnemonic_database()
    banned_substrings = [
        'RLLD_HRLA_AT90_AHT90',
        'RLLM_HRLA_AT60_AHT60',
        'RLLS_HRLA_AT30_AHT30',
        'AT90_HRLA_AHT90',
    ]
    all_mnemonics = {
        m.upper()
        for entry in db.values()
        for m in entry.get('mnemonics', [])
    }
    present = [b for b in banned_substrings if b in all_mnemonics]
    assert present == [], f"Invented combo mnemonics still present: {present}"


def test_demo_las_mnemonics_identify():
    engine = CurveIdentificationEngine()
    expected = {
        'DEPT': 'DEPTH',
        'GR': 'GAMMA_RAY_TOTAL',
        'RHOB': 'BULK_DENSITY',
        'NPHI': 'NEUTRON_POROSITY',
        'RT': 'RESISTIVITY_DEEP',
        'DT': 'SONIC_COMPRESSIONAL',
        'CALI': 'CALIPER',
    }
    for mnemonic, curve_type in expected.items():
        identified, confidence, _ = engine.identify_curve(mnemonic)
        assert identified == curve_type, f"{mnemonic} -> {identified}, expected {curve_type}"
        assert confidence >= 0.9


def test_metadata_apis_available():
    engine = CurveIdentificationEngine()
    assert engine.get_industry_color_for_curve('GR') == '#008000'
    assert engine.is_log_scale_curve('RT') is True
    assert engine.get_optimal_wavelet_for_curve('RHOB') == 'db4'
    lo, hi = engine.get_track_scale_for_curve('NPHI')
    assert lo > hi  # reversed neutron scale
