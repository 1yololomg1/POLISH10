"""DLIS support surface checks without launching the GUI."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / 'advanced_preprocessing_system10.py').read_text(encoding='utf-8', errors='ignore')
LOADERS = (ROOT / 'core' / 'well_loaders.py').read_text(encoding='utf-8', errors='ignore')


def test_dlis_extensions_allowed_in_safe_handler():
    assert "'.dlis'" in MAIN and "'.lis'" in MAIN
    assert 'ALLOWED_READ_EXTENSIONS' in MAIN


def test_load_dlis_and_load_data_methods_defined():
    # Loader implementation lives in WellLoadingMixin after Step 3 extraction
    assert 'def load_dlis_file(' in LOADERS
    assert 'dlisio' in LOADERS
    # App still owns the dispatch entry point
    assert 'def load_data(' in MAIN
    assert 'WellLoadingMixin' in MAIN
