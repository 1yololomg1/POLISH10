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


def test_exact_short_spectral_mnemonics_identify():
    """Full-mnemonic K / U / TH must still resolve to spectral components."""
    engine = CurveIdentificationEngine()
    expected = {
        'K': 'POTASSIUM',
        'U': 'URANIUM',
        'TH': 'THORIUM',
    }
    for mnemonic, curve_type in expected.items():
        identified, confidence, _ = engine.identify_curve(mnemonic)
        assert identified == curve_type, f"{mnemonic} -> {identified}, expected {curve_type}"
        assert confidence >= 0.9


def test_short_spectral_mnemonics_do_not_substring_false_positive():
    """
    Longer names that merely contain K / U / TH characters or substrings must
    not identify as spectral potassium/uranium/thorium solely due to short
    mnemonic overlap.
    """
    engine = CurveIdentificationEngine()
    spectral = {'POTASSIUM', 'URANIUM', 'THORIUM'}
    probes = [
        'ROCK',
        'BULK',
        'DEPTH',
        'UNIT',
        'UNKNOWN',
        'THICK',
        'THETA',
        'AUTH',
        'SOUTH',
        'PATH',
        'MYTH',
        'GR_K',
        'SGRK',
        'CALK',
        'RHOBK',
        'STH',
        'GTH',
        'DTH',
        'UK',
        'KU',
        'KT',
        'THU',
        'KTH',
        'XK',
        'XU',
        'XTH',
    ]
    for name in probes:
        identified, confidence, _ = engine.identify_curve(name)
        assert identified not in spectral, (
            f"{name} incorrectly identified as {identified} (conf={confidence:.3f}) "
            f"via short-mnemonic overlap"
        )


def test_fuzzy_refuses_long_query_onto_short_mnemonic():
    """Levenshtein path must not map long strings onto very short DB mnemonics."""
    engine = CurveIdentificationEngine()
    for query, short in [
        ('ROCK', 'K'),
        ('BULK', 'K'),
        ('UNIT', 'U'),
        ('DEPTH', 'TH'),
        ('THICK', 'TH'),
        ('CALK', 'K'),
        ('MYTH', 'TH'),
    ]:
        matched, _ = engine._fuzzy_match_mnemonic(query, short, threshold=0.7)
        assert matched is False, f"fuzzy unexpectedly matched {query!r} ~ {short!r}"


def test_longer_spectral_aliases_still_identify():
    """Non-short spectral aliases (POTA/URAN/THOR/HTHO/…) remain valid."""
    engine = CurveIdentificationEngine()
    expected = {
        'POTA': 'POTASSIUM',
        'HPOT': 'POTASSIUM',
        'URAN': 'URANIUM',
        'HURA': 'URANIUM',
        'THOR': 'THORIUM',
        'HTHO': 'THORIUM',
    }
    for mnemonic, curve_type in expected.items():
        identified, confidence, _ = engine.identify_curve(mnemonic)
        assert identified == curve_type, f"{mnemonic} -> {identified}, expected {curve_type}"
        assert confidence >= 0.9
