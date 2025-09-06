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


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests that make real HTTP requests as integration tests
        if "real_api" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark tests that might be slow
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
