import numpy as np

from core.petrophysical_models import ArchieEquationCalculator, RelativeRockPropertiesModel


def test_archie_formation_factor():
    calc = ArchieEquationCalculator()
    phi = np.array([0.20])
    rt = np.array([10.0])
    result = calc.calculate_water_saturation_archie(phi, rt, a=1.0, m=2.0, n=2.0, rw=0.05)
    f = 1.0 / (0.20 ** 2.0)
    assert abs(f - 25.0) < 1e-6
    sw_n = f * 0.05 / 10.0
    assert abs(result['water_saturation'][0] - np.sqrt(sw_n)) < 1e-6


def test_rrp_swap_direction_with_underscored_names():
    x = np.linspace(1, 10, 50)
    y = 2.0 * x + 3.0
    data = {'DEEP_RES': x, 'NPHI_CORR': y}
    model = RelativeRockPropertiesModel()
    model.train(data)

    gap_start, gap_end = 10, 20
    aux = {'NPHI_CORR': y.copy()}
    target_data = x.copy()
    target_data[gap_start:gap_end] = np.nan

    result = model.fill_large_gap('DEEP_RES', gap_start, gap_end, target_data, aux)
    assert result is not None
    filled = result['values']
    assert np.sum(~np.isnan(filled)) >= len(filled) * 0.5
    expected = (y[gap_start:gap_end] - 3.0) / 2.0
    valid = ~np.isnan(filled)
    np.testing.assert_allclose(filled[valid], expected[valid], rtol=0.05)

    model2 = RelativeRockPropertiesModel()
    model2.train(data)
    aux2 = {'DEEP_RES': x.copy()}
    y_target = y.copy()
    y_target[gap_start:gap_end] = np.nan
    result2 = model2.fill_large_gap('NPHI_CORR', gap_start, gap_end, y_target, aux2)
    assert result2 is not None
    filled2 = result2['values']
    expected2 = 2.0 * x[gap_start:gap_end] + 3.0
    valid2 = ~np.isnan(filled2)
    np.testing.assert_allclose(filled2[valid2], expected2[valid2], rtol=0.05)
