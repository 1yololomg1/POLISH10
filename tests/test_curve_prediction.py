from __future__ import annotations

import numpy as np

from core.petrophysical_models import RelativeRockPropertiesModel


def test_predict_missing_curve_full_length():
    x = np.linspace(1, 20, 100)
    y = 1.5 * x + 2.0
    model = RelativeRockPropertiesModel()
    model.train({'GR': x, 'DT': y})

    # Entire DT missing — predict from GR
    result = model.predict_missing_curve('DT', {'GR': x}, existing_target=None)
    assert result is not None
    pred = result['predicted']
    assert result['points_predicted'] > 50
    np.testing.assert_allclose(pred[10:90], y[10:90], rtol=0.1)


def test_predict_missing_curve_preserves_known_samples():
    x = np.linspace(1, 20, 80)
    y = 2.0 * x + 1.0
    model = RelativeRockPropertiesModel()
    model.train({'GR': x, 'RHOB': y})

    sparse = y.copy()
    sparse[20:40] = np.nan
    result = model.predict_missing_curve('RHOB', {'GR': x}, existing_target=sparse)
    assert result is not None
    # Known points preserved
    np.testing.assert_allclose(result['predicted'][:20], y[:20], rtol=1e-6)
    # Gap filled
    assert not np.any(np.isnan(result['predicted'][20:40]))
