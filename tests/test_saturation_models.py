import numpy as np
import pytest

from petrophysics.saturation_models import ShalySandSaturationModels


@pytest.fixture
def models():
    return ShalySandSaturationModels()


def test_archie_spot_calculation(models):
    calc = __import__('core.petrophysical_models', fromlist=['ArchieEquationCalculator']).ArchieEquationCalculator()
    phi = np.array([0.20])
    rt = np.array([10.0])
    result = calc.calculate_water_saturation_archie(phi, rt, a=1.0, m=2.0, n=2.0, rw=0.05)
    expected = 0.3535533905932738
    assert abs(result['water_saturation'][0] - expected) < 1e-4


def test_simandoux_n2_hand_check(models):
    phi = np.array([0.20])
    rt = np.array([10.0])
    vsh = np.array([0.30])
    result = models.simandoux_saturation(phi, rt, vsh, rsh=2.0, a=1.0, m=2.0, n=2.0, rw=0.05)
    expected = 0.27202
    assert abs(result['sw'][0] - expected) < 1e-4


def test_simandoux_reduces_to_archie_when_clean(models):
    phi = np.array([0.20, 0.25])
    rt = np.array([10.0, 8.0])
    vsh = np.zeros(2)
    sim = models.simandoux_saturation(phi, rt, vsh, n=2.0)['sw']
    archie_sw = np.power((1.0 * 0.05) / (rt * phi ** 2.0), 0.5)
    np.testing.assert_allclose(sim, archie_sw, rtol=1e-4)


def test_indonesia_reduces_to_archie_when_clean(models):
    phi = np.array([0.20])
    rt = np.array([10.0])
    vsh = np.array([0.0])
    sw = models.indonesia_saturation(phi, rt, vsh, n=2.0)['sw'][0]
    expected = 0.3535533905932738
    assert abs(sw - expected) < 1e-4


def test_indonesia_monotonic_with_resistivity(models):
    phi = np.full(5, 0.20)
    vsh = np.full(5, 0.15)
    rt = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
    sw = models.indonesia_saturation(phi, rt, vsh)['sw']
    assert np.all(np.diff(sw) < 0)


def test_dual_water_converges_and_bounded(models):
    phi = np.array([0.22])
    rt = np.array([12.0])
    vsh = np.array([0.45])
    result = models.dual_water_saturation(phi, rt, vsh)
    assert result['convergence'][0]
    assert 0.0 <= result['sw_total'][0] <= 1.0


def test_dual_water_approaches_archie_when_no_clay_water(models):
    phi = np.array([0.20])
    rt = np.array([10.0])
    vsh = np.array([0.05])
    swb = np.array([0.0])
    dw = models.dual_water_saturation(phi, rt, vsh, swb=swb)['sw_total'][0]
    archie = np.power((0.05) / (10.0 * 0.20 ** 2.0), 0.5)
    assert abs(dw - archie) < 0.15


def test_auto_select_high_vsh_uses_dual_water(models):
    gr = np.array([100.0])
    phi = np.array([0.22])
    rt = np.array([8.0])
    result = models.auto_select_model(phi, rt, gr, gr_clean=20.0, gr_shale=120.0)
    assert result['model_used'][0] == 'dual_water_highVsh'
