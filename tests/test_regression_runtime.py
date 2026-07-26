from core.environmental_corrections import EnvironmentalCorrectionsManager


def test_pipeline_environmental_manager_api():
    mgr = EnvironmentalCorrectionsManager()
    required = [
        'apply_environmental_corrections',
        'apply_all_corrections',
        'correct_density_borehole',
        'correct_neutron_borehole',
        'correct_temperature_drift',
        'get_correction_summary',
    ]
    for name in required:
        assert hasattr(mgr, name), f"Missing method: {name}"
        assert callable(getattr(mgr, name))
