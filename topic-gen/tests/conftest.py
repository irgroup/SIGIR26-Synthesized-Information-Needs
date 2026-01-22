from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Returns the absolute path to the test data directory.
    'scope="session"' means this calculation only happens once.
    """
    return Path(__file__).parent / "data"
