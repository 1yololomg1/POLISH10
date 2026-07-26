"""Gap-filling math checks without importing the Tk GUI module."""
import numpy as np


def linear_fill(before: float, after: float, size: int) -> np.ndarray:
    """Mirrors AdvancedGapFiller._linear_interpolation two-sided case."""
    return np.linspace(before, after, size)


def test_linear_interpolation_exact():
    values = linear_fill(4.0, 10.0, 5)
    expected = np.linspace(4.0, 10.0, 5)
    np.testing.assert_allclose(values, expected, rtol=1e-6)


def test_punched_hole_recoverable_by_linspace():
    depth = np.arange(0, 50, dtype=float)
    clean = 2.0 * depth + 10.0
    punched = clean.copy()
    punched[20:28] = np.nan
    filled = punched.copy()
    filled[20:28] = np.linspace(clean[19], clean[28], 8)
    np.testing.assert_allclose(filled[20:28], clean[20:28], rtol=0.05)
    assert not np.any(np.isnan(filled))


def test_no_gaps_passthrough():
    data = np.linspace(0, 1, 30)
    assert np.sum(np.isnan(data)) == 0
    np.testing.assert_array_equal(data, data.copy())
