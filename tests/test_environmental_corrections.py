import numpy as np

from core.environmental_corrections import EnvironmentalCorrectionsManager


def test_sandstone_matrix_correction_positive():
    mgr = EnvironmentalCorrectionsManager()
    nphi = np.array([0.15])
    caliper = np.array([8.5])
    result = mgr.correct_neutron_borehole(nphi, caliper, bit_size=8.5, matrix_type='sandstone')
    assert result['nphi_corrected'][0] > nphi[0]
    assert abs(result['matrix_correction'] - 0.04) < 1e-6


def test_dolomite_matrix_correction_negative():
    mgr = EnvironmentalCorrectionsManager()
    nphi = np.array([0.15])
    caliper = np.array([8.5])
    result = mgr.correct_neutron_borehole(nphi, caliper, bit_size=8.5, matrix_type='dolomite')
    assert result['nphi_corrected'][0] < nphi[0]
    assert abs(result['matrix_correction'] + 0.06) < 1e-6


def test_limestone_matrix_correction_zero():
    mgr = EnvironmentalCorrectionsManager()
    nphi = np.array([0.15])
    caliper = np.array([8.5])
    result = mgr.correct_neutron_borehole(nphi, caliper, bit_size=8.5, matrix_type='limestone')
    assert result['nphi_corrected'][0] == nphi[0]


def test_arp_temperature_correction():
    mgr = EnvironmentalCorrectionsManager()
    r = np.array([10.0])
    temp = np.array([150.0])
    result = mgr.correct_temperature_drift(r, temp, reference_temp=75.0, curve_type='resistivity')
    expected = 10.0 * (150.0 + 6.77) / (75.0 + 6.77)
    assert abs(result['corrected_data'][0] - expected) < 1e-2


def test_oversized_hole_increases_neutron():
    mgr = EnvironmentalCorrectionsManager()
    nphi = np.array([0.15, 0.15])
    caliper = np.array([8.5, 10.5])
    result = mgr.correct_neutron_borehole(nphi, caliper, bit_size=8.5, matrix_type='limestone')
    assert result['nphi_corrected'][1] > result['nphi_corrected'][0]
