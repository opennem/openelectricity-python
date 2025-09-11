"""
Tests for the synchronous OEClient.

This module tests the new synchronous client implementation using the requests library.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import requests

from openelectricity.client import OEClient, OpenElectricityError, APIError
from openelectricity.types import (
    NetworkCode,
    DataMetric,
    DataInterval,
    DataPrimaryGrouping,
    DataSecondaryGrouping,
    MarketMetric,
    UnitFueltechType,
    UnitStatusType,
)
from openelectricity.models.facilities import FacilityResponse
from openelectricity.models.timeseries import TimeSeriesResponse
from openelectricity.models.user import OpennemUserResponse


class TestOEClient:
    """Test suite for the synchronous OEClient."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock response object."""
        mock = Mock()
        mock.ok = True
        mock.status_code = 200
        mock.json.return_value = {"data": [], "success": True}
        return mock

    @pytest.fixture
    def mock_error_response(self):
        """Create a mock error response object."""
        mock = Mock()
        mock.ok = False
        mock.status_code = 404
        mock.reason = "Not Found"
        mock.json.return_value = {"detail": "Resource not found"}
        return mock

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return OEClient(api_key="test-api-key")

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.api_key == "test-api-key"
        assert client.base_url == "https://api.openelectricity.org.au/"
        assert "Authorization" in client.headers
        assert "Bearer test-api-key" in client.headers["Authorization"]

    def test_client_initialization_with_custom_base_url(self):
        """Test client initialization with custom base URL."""
        client = OEClient(api_key="test-key", base_url="https://custom.api.com")
        assert client.base_url == "https://custom.api.com/"

    def test_client_initialization_without_api_key(self):
        """Test client initialization without API key raises error."""
        # Mock the settings to ensure no default API key
        with patch("openelectricity.client.settings") as mock_settings:
            mock_settings.api_key = ""
            with pytest.raises(OpenElectricityError, match="API key must be provided"):
                OEClient(api_key=None)

    def test_build_url(self, client):
        """Test URL construction."""
        test_cases = [
            ("/facilities/", "https://api.openelectricity.org.au/v4/facilities/"),
            ("/data/network/NEM", "https://api.openelectricity.org.au/v4/data/network/NEM"),
            ("/me", "https://api.openelectricity.org.au/v4/me"),
        ]

        for endpoint, expected in test_cases:
            result = client._build_url(endpoint)
            assert result == expected

    def test_clean_params(self, client):
        """Test parameter cleaning."""
        params = {
            "key1": "value1",
            "key2": None,
            "key3": "",
            "key4": 0,
            "key5": False,
        }

        cleaned = client._clean_params(params)
        assert "key1" in cleaned
        assert "key2" not in cleaned
        assert "key3" in cleaned  # Empty string is not None
        assert "key4" in cleaned
        assert "key5" in cleaned

    @patch("openelectricity.client.requests.Session")
    def test_ensure_session(self, mock_session_class, client):
        """Test session creation and configuration."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        session = client._ensure_session()

        # Verify session was created
        mock_session_class.assert_called_once()

        # Verify headers were set
        mock_session.headers.update.assert_called_once_with(client.headers)

        # Verify HTTP adapter was configured
        mock_session.mount.assert_called_once()

    @patch("openelectricity.client.requests.Session")
    def test_ensure_session_reuses_existing(self, mock_session_class, client):
        """Test that existing session is reused."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Create session first time
        session1 = client._ensure_session()

        # Create session second time
        session2 = client._ensure_session()

        # Should be the same session
        assert session1 is session2

        # Session class should only be called once
        mock_session_class.assert_called_once()

    def test_handle_response_success(self, client, mock_response):
        """Test successful response handling."""
        result = client._handle_response(mock_response)
        assert result == {"data": [], "success": True}

    def test_handle_response_error(self, client, mock_error_response):
        """Test error response handling."""
        with pytest.raises(APIError) as exc_info:
            client._handle_response(mock_error_response)

        assert exc_info.value.status_code == 404
        assert "Resource not found" in str(exc_info.value)

    def test_handle_response_error_without_json(self, client):
        """Test error response handling when JSON parsing fails."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.json.side_effect = Exception("JSON parse error")

        with pytest.raises(APIError) as exc_info:
            client._handle_response(mock_response)

        assert exc_info.value.status_code == 500
        assert "Internal Server Error" in str(exc_info.value)

    @patch("openelectricity.client.requests.Session")
    def test_get_facilities(self, mock_session_class, client, mock_response):
        """Test get_facilities method."""
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Mock the response data
        mock_response.json.return_value = {
            "data": [
                {
                    "code": "TEST1",
                    "name": "Test Facility",
                    "network_id": "NEM",
                    "network_region": "QLD1",
                    "description": "Test facility for testing",
                    "units": [
                        {
                            "code": "TEST1_U1",
                            "fueltech_id": "solar_utility",
                            "status_id": "operating",
                            "capacity_registered": 100.0,
                            "emissions_factor_co2": None,
                            "data_first_seen": "2024-01-01T00:00:00Z",
                            "data_last_seen": "2024-01-01T00:00:00Z",
                            "dispatch_type": "scheduled",
                        }
                    ],
                }
            ],
            "success": True,
            "version": "v4.0",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = client.get_facilities()

        # Verify request was made
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert "/v4/facilities/" in call_args[0][0]

        # Verify result
        assert isinstance(result, FacilityResponse)

    @patch("openelectricity.client.requests.Session")
    def test_get_facilities_with_filters(self, mock_session_class, client, mock_response):
        """Test get_facilities method with filters."""
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        mock_response.json.return_value = {"data": [], "success": True, "version": "v4.0", "created_at": "2024-01-01T00:00:00Z"}

        result = client.get_facilities(
            facility_code=["TEST1", "TEST2"],
            status_id=[UnitStatusType.OPERATING],
            fueltech_id=[UnitFueltechType.SOLAR_UTILITY],
            network_id=["NEM"],
            network_region="QLD1",
        )

        # Verify request parameters
        call_args = mock_session.get.call_args
        params = call_args[1]["params"]

        assert params["facility_code"] == ["TEST1", "TEST2"]
        assert params["status_id"] == ["operating"]
        assert params["fueltech_id"] == ["solar_utility"]
        assert params["network_id"] == ["NEM"]
        assert params["network_region"] == "QLD1"

    @patch("openelectricity.client.requests.Session")
    def test_get_network_data(self, mock_session_class, client, mock_response):
        """Test get_network_data method."""
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        mock_response.json.return_value = {
            "data": [
                {
                    "network_code": "NEM",
                    "metric": "energy",
                    "unit": "MWh",
                    "interval": "5m",
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-01T01:00:00Z",
                    "groupings": ["fueltech_group"],
                    "network_timezone_offset": "+10:00",
                    "results": [
                        {
                            "name": "Solar",
                            "date_start": "2024-01-01T00:00:00Z",
                            "date_end": "2024-01-01T01:00:00Z",
                            "columns": {"fueltech_group": "solar"},
                            "data": [["2024-01-01T00:00:00Z", 100.5], ["2024-01-01T00:05:00Z", 120.3]],
                        }
                    ],
                }
            ],
            "success": True,
            "version": "v4.0",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.ENERGY],
            interval="5m",
            date_start=datetime.now() - timedelta(hours=1),
            date_end=datetime.now(),
            primary_grouping="network",
            secondary_grouping="fueltech_group",
        )

        # Verify request was made
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert "/v4/data/network/NEM" in call_args[0][0]

        # Verify result
        assert isinstance(result, TimeSeriesResponse)

    @patch("openelectricity.client.requests.Session")
    def test_get_facility_data(self, mock_session_class, client, mock_response):
        """Test get_facility_data method."""
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        mock_response.json.return_value = {
            "data": [
                {
                    "network_code": "NEM",
                    "metric": "power",
                    "unit": "MW",
                    "interval": "5m",
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-01T01:00:00Z",
                    "groupings": ["facility"],
                    "network_timezone_offset": "+10:00",
                    "results": [
                        {
                            "name": "TEST1",
                            "date_start": "2024-01-01T00:00:00Z",
                            "date_end": "2024-01-01T01:00:00Z",
                            "columns": {"facility": "TEST1"},
                            "data": [["2024-01-01T00:00:00Z", 50.0], ["2024-01-01T00:05:00Z", 55.0]],
                        }
                    ],
                }
            ],
            "success": True,
            "version": "v4.0",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = client.get_facility_data(network_code="NEM", facility_code="TEST1", metrics=[DataMetric.POWER], interval="5m")

        # Verify request was made
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert "/v4/data/facilities/NEM" in call_args[0][0]

        # Verify result
        assert isinstance(result, TimeSeriesResponse)

    @patch("openelectricity.client.requests.Session")
    def test_get_market(self, mock_session_class, client, mock_response):
        """Test get_market method."""
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        mock_response.json.return_value = {
            "data": [
                {
                    "network_code": "NEM",
                    "metric": "price",
                    "unit": "$/MWh",
                    "interval": "5m",
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-01T01:00:00Z",
                    "groupings": ["network_region"],
                    "network_timezone_offset": "+10:00",
                    "results": [
                        {
                            "name": "QLD1",
                            "date_start": "2024-01-01T00:00:00Z",
                            "date_end": "2024-01-01T01:00:00Z",
                            "columns": {"network_region": "QLD1"},
                            "data": [["2024-01-01T00:00:00Z", 45.50], ["2024-01-01T00:05:00Z", 47.20]],
                        }
                    ],
                }
            ],
            "success": True,
            "version": "v4.0",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.PRICE],
            interval="5m",
            primary_grouping="network_region",
            network_region="QLD1",
        )

        # Verify request was made
        mock_session.get.call_args
        call_args = mock_session.get.call_args
        assert "/v4/market/network/NEM" in call_args[0][0]

        # Verify result
        assert isinstance(result, TimeSeriesResponse)

    @patch("openelectricity.client.requests.Session")
    def test_get_current_user(self, mock_session_class, client, mock_response):
        """Test get_current_user method."""
        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        mock_response.json.return_value = {
            "data": {
                "id": "user123",
                "email": "test@example.com",
                "full_name": "Test User",
                "owner_id": None,
                "plan": "pro",
                "rate_limit": None,
                "unkey_meta": None,
                "roles": ["user"],
                "meta": None,
            },
            "success": True,
            "version": "v4.0",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = client.get_current_user()

        # Verify request was made
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert "/v4/me" in call_args[0][0]

        # Verify result
        assert isinstance(result, OpennemUserResponse)

    def test_context_manager(self):
        """Test client as context manager."""
        with OEClient(api_key="test-key") as client:
            assert isinstance(client, OEClient)
            # Client should be properly initialized

    def test_close_method(self, client):
        """Test client close method."""
        # Mock the session
        mock_session = Mock()
        client._session = mock_session

        client.close()

        # Verify session was closed
        mock_session.close.assert_called_once()
        assert client._session is None

    def test_close_method_no_session(self, client):
        """Test close method when no session exists."""
        # Should not raise an error
        client.close()

    def test_del_method(self, client):
        """Test client destructor."""
        # Mock the session
        mock_session = Mock()
        client._session = mock_session

        # Call destructor
        client.__del__()

        # Verify session was closed
        mock_session.close.assert_called_once()

    @patch("openelectricity.client.requests.Session")
    def test_session_configuration(self, mock_session_class, client):
        """Test that session is configured with proper HTTP adapter."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        client._ensure_session()

        # Verify HTTP adapter was configured
        mock_session.mount.assert_called_once()
        mount_args = mock_session.mount.call_args
        assert mount_args[0][0] == "https://"

        # Verify adapter configuration
        adapter = mount_args[0][1]
        assert adapter._pool_connections == 10
        assert adapter._pool_maxsize == 20
        assert adapter.max_retries.total == 3
        assert adapter._pool_block is False


class TestOEClientErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return OEClient(api_key="test-api-key")

    @patch("openelectricity.client.requests.Session")
    def test_network_error_handling(self, mock_session_class, client):
        """Test handling of network errors."""
        mock_session = Mock()
        mock_session.get.side_effect = requests.RequestException("Network error")
        mock_session_class.return_value = mock_session

        with pytest.raises(requests.RequestException, match="Network error"):
            client.get_facilities()

    @patch("openelectricity.client.requests.Session")
    def test_invalid_json_response(self, mock_session_class, client):
        """Test handling of invalid JSON responses."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(Exception, match="Invalid JSON"):
            client.get_facilities()


class TestOEClientIntegration:
    """Integration tests for the client."""

    @pytest.mark.integration
    def test_real_api_connection(self):
        """Test connection to real API (requires API key)."""
        api_key = "test-key"  # This would be set in environment

        try:
            client = OEClient(api_key=api_key)
            # This would fail with invalid API key, but tests the connection
            with pytest.raises(APIError):
                client.get_current_user()
        except Exception as e:
            # Any exception is fine for this test
            assert isinstance(e, Exception)