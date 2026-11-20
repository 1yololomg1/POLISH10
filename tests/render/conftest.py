"""Register the performance mark used by the bin timing test. P17.6."""


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "performance: reports a timing and does not apply a pass threshold (P17.6)",
    )
