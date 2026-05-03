"""pytest configuration and shared fixtures."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow-running")
