"""
Pytest configuration and shared fixtures for OpenElectricity tests.

This module provides common fixtures and configuration for all tests.
"""

import os
from pathlib import Path
from typing import Optional

import pytest
from dotenv import load_dotenv


def load_env_file() -> None:
    """Load environment variables from .env file in project root."""
    # Get the project root directory (parent of tests directory)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Also try loading from current directory
        load_dotenv()


@pytest.fixture(scope="session")
def openelectricity_api_key() -> Optional[str]:
    """
    Fixture to provide the OpenElectricity API key.
    
    Loads the API key from:
    1. OPENELECTRICITY_API_KEY environment variable
    2. .env file in project root
    3. Returns None if not found
    
    Returns:
        str: The API key if found, None otherwise
    """
    # Load environment variables from .env file
    load_env_file()
    
    # Get API key from environment
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    
    if not api_key:
        pytest.skip("OPENELECTRICITY_API_KEY not found in environment or .env file")
    
    return api_key


@pytest.fixture(scope="session")
def openelectricity_client():
    """
    Fixture to provide an OEClient instance with API key.
    
    Automatically loads API key from .env file and creates a client.
    Skips tests if API key is not available.
    
    Returns:
        OEClient: Configured client instance
    """
    from openelectricity import OEClient
    
    # Load environment variables from .env file
    load_env_file()
    
    # Get API key from environment
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    
    if not api_key:
        pytest.skip("OPENELECTRICITY_API_KEY not found in environment or .env file")
    
    return OEClient(api_key=api_key)


@pytest.fixture(scope="session")
def openelectricity_async_client():
    """
    Fixture to provide an AsyncOEClient instance with API key.
    
    Automatically loads API key from .env file and creates an async client.
    Skips tests if API key is not available.
    
    Returns:
        AsyncOEClient: Configured async client instance
    """
    from openelectricity import AsyncOEClient
    
    # Load environment variables from .env file
    load_env_file()
    
    # Get API key from environment
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    
    if not api_key:
        pytest.skip("OPENELECTRICITY_API_KEY not found in environment or .env file")
    
    return AsyncOEClient(api_key=api_key)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Auto-use fixture to set up test environment.
    
    This fixture runs automatically for every test and ensures
    the .env file is loaded.
    """
    load_env_file()


# Optional: Add markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )
    config.addinivalue_line(
        "markers", "pyspark: marks tests that require PySpark"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )

"""
Pytest configuration and shared fixtures for the OpenElectricity test suite.
"""

import pytest
import os
from unittest.mock import Mock


@pytest.fixture(scope="session")
def test_api_key():
    """Provide a test API key for testing."""
    return "test-api-key-12345"


@pytest.fixture(scope="session")
def test_base_url():
    """Provide a test base URL for testing."""
    return "https://test.api.openelectricity.org.au"


@pytest.fixture
def mock_requests_session():
    """Mock requests.Session for testing HTTP requests."""
    with pytest.MonkeyPatch.context() as m:
        mock_session = Mock()
        mock_session.headers = {}
        mock_session.mount = Mock()

        # Mock the session class
        m.setattr("openelectricity.client.requests.Session", Mock(return_value=mock_session))

        yield mock_session


@pytest.fixture
def mock_response_success():
    """Mock successful HTTP response."""
    mock_response = Mock()
    mock_response.ok = True
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [], "success": True, "version": "v4.0", "created_at": "2024-01-01T00:00:00Z"}
    return mock_response


@pytest.fixture
def mock_response_error():
    """Mock error HTTP response."""
    mock_response = Mock()
    mock_response.ok = False
    mock_response.status_code = 404
    mock_response.reason = "Not Found"
    mock_response.json.return_value = {"detail": "Resource not found", "success": False}
    return mock_response


@pytest.fixture
def mock_facility_response():
    """Mock facility response data."""
    return {
        "data": [
            {
                "code": "TEST_FACILITY_1",
                "name": "Test Solar Facility",
                "network_id": "NEM",
                "network_region": "QLD1",
                "status": "operating",
                "fueltech": "solar_utility",
            },
            {
                "code": "TEST_FACILITY_2",
                "name": "Test Wind Facility",
                "network_id": "NEM",
                "network_region": "SA1",
                "status": "operating",
                "fueltech": "wind",
            },
        ],
        "success": True,
        "version": "v4.0",
        "created_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def mock_timeseries_response():
    """Mock timeseries response data."""
    return {
        "data": [
            {
                "metric": "energy",
                "unit": "MWh",
                "results": [
                    {
                        "name": "Solar",
                        "data": [
                            {"timestamp": "2024-01-01T00:00:00Z", "value": 100.5},
                            {"timestamp": "2024-01-01T00:05:00Z", "value": 120.3},
                        ],
                    }
                ],
            }
        ],
        "success": True,
        "version": "v4.0",
        "created_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def mock_user_response():
    """Mock user response data."""
    return {
        "data": {"id": "user123", "email": "test@example.com", "full_name": "Test User", "plan": "pro", "roles": ["user"]},
        "success": True,
        "version": "v4.0",
        "created_at": "2024-01-01T00:00:00Z",
    }



def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests that make real HTTP requests as integration tests
        if "real_api" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark tests that might be slow
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)