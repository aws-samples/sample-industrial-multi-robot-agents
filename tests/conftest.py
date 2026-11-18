"""Shared pytest configuration for the test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: exercises the real planner and cell geometry (minutes, not "
        "seconds); needs Pinocchio and OMPL. Deselect with -m 'not slow'.",
    )
