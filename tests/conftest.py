"""
Configuration file for pytest.

This file configures pytest for the ExactCIs project, including:
- Test markers for categorizing tests
- Timeout settings to prevent tests from running indefinitely
- Verbose output configuration
- Logging setup
"""

import pytest
import logging
import time
from _pytest.logging import LogCaptureFixture
from exactcis.methods.midp import _cache as midp_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("exactcis.tests")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "fast: mark a test as fast running")
    config.addinivalue_line("markers", "slow: mark a test as slow running")
    config.addinivalue_line("markers", "unconditional: tests for the unconditional method")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "edge: tests edge cases")
    config.addinivalue_line("markers", "mock: tests that use mocking")
    config.addinivalue_line("markers", "core: tests of core functionality")
    config.addinivalue_line("markers", "methods: tests for specific CI methods")
    config.addinivalue_line("markers", "utils: tests for utility functions")


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-all", action="store_true", default=False, help="run all tests including skipped ones"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is specified."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="function")
def timer():
    """Fixture to time test execution."""
    start = time.time()
    logger.info("Test starting")
    yield
    duration = time.time() - start
    logger.info(f"Test completed in {duration:.6f} seconds")


@pytest.fixture(scope="function")
def caplog_custom(caplog: LogCaptureFixture):
    """Enhanced log capture fixture with automatic level setting."""
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture(autouse=True)
def clear_midp_cache_globally():
    """Clears the midp method's internal cache before and after each test run globally."""
    midp_cache.clear()
    yield
    midp_cache.clear()


@pytest.fixture(scope="session", autouse=True)
def session_setup_teardown():
    """Setup and teardown for the entire test session."""
    logger.info("Starting test session")

    # Setup code here
    yield
    # Teardown code here

    logger.info("Test session completed")

    # Note: The following code was causing errors because config and items are not defined
    # Unskipping tests should be handled in pytest_collection_modifyitems instead
    # if config.getoption("--run-all"):
    #     for item in items:
    #         if item.get_closest_marker("skip"):
    #             item.add_marker(pytest.mark.skip(reason=None))
