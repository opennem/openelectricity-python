#!/usr/bin/env python3
"""
Tests for the get_network_data function in examples/databricks/openelectricity_etl.py
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Optional PySpark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
    PYSPARK_AVAILABLE = True
except ImportError:
    SparkSession = None
    StructType = None
    StructField = None
    StringType = None
    DoubleType = None
    TimestampType = None
    PYSPARK_AVAILABLE = False

from examples.databricks.openelectricity_etl import get_network_data
from openelectricity.types import NetworkCode, DataInterval, DataPrimaryGrouping, DataSecondaryGrouping, DataMetric


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
class TestGetNetworkData:
    """Test cases for get_network_data function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Load environment variables
        from dotenv import load_dotenv

        load_dotenv()

        # Check if API key is available
        self.api_key = os.getenv("OPENELECTRICITY_API_KEY")
        self.has_api_key = bool(self.api_key)

    @pytest.fixture
    def mock_spark_session(self):
        """Mock Spark session for testing."""
        with patch("examples.databricks.openelectricity_etl._get_spark") as mock_get_spark:
            mock_spark = Mock(spec=SparkSession)
            mock_dataframe = Mock()
            mock_dataframe.withColumnRenamed.return_value = mock_dataframe
            mock_spark.createDataFrame.return_value = mock_dataframe
            mock_get_spark.return_value = mock_spark
            yield mock_spark

    @pytest.fixture
    def mock_client(self):
        """Mock OpenNEM client for testing."""
        mock_client = Mock()
        mock_response = Mock()

        # Mock the response data
        mock_response.to_pandas.return_value = Mock()
        mock_response.get_metric_units.return_value = {"power": "MW", "energy": "MWh", "market_value": "$", "emissions": "t"}

        mock_client.get_network_data.return_value = mock_response

        return mock_client

    @pytest.fixture
    def sample_network_data(self):
        """Sample network data for testing."""
        return {
            "interval": ["2025-09-04T20:25:00+10:00", "2025-09-04T20:30:00+10:00"],
            "network_region": ["QLD1", "QLD1"],
            "fueltech_group": ["solar_utility", "solar_utility"],
            "power": [1000.5, 1050.2],
            "energy": [83.375, 87.517],
            "market_value": [5000.25, 5250.10],
            "emissions": [50.025, 52.510],
        }

    def test_get_network_data_basic_parameters(self, mock_spark_session, mock_client):
        """Test get_network_data with basic parameters."""
        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            # Call the function
            result = get_network_data(network="NEM", interval="5m", days_back=1)

            # Verify the client was called correctly
            mock_client.get_network_data.assert_called_once()
            call_args = mock_client.get_network_data.call_args

            assert call_args[1]["network_code"] == "NEM"
            assert call_args[1]["interval"] == "5m"
            assert DataMetric.POWER in call_args[1]["metrics"]
            assert DataMetric.ENERGY in call_args[1]["metrics"]
            assert DataMetric.MARKET_VALUE in call_args[1]["metrics"]
            assert DataMetric.EMISSIONS in call_args[1]["metrics"]

            # Verify Spark DataFrame was created
            mock_spark_session.createDataFrame.assert_called_once()

    def test_get_network_data_with_custom_groupings(self, mock_spark_session, mock_client):
        """Test get_network_data with custom primary and secondary groupings."""
        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            # Call the function with custom groupings
            result = get_network_data(
                network="NEM", interval="1h", days_back=7, primary_grouping="fueltech_group", secondary_grouping="network_region"
            )

            # Verify the groupings were passed correctly
            call_args = mock_client.get_network_data.call_args
            assert call_args[1]["primary_grouping"] == "fueltech_group"
            assert call_args[1]["secondary_grouping"] == "network_region"

    def test_get_network_data_with_end_date(self, mock_spark_session, mock_client):
        """Test get_network_data with custom end_date."""
        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            end_date = datetime(2025, 9, 4, 12, 0, 0)

            # Call the function with custom end_date
            result = get_network_data(network="NEM", interval="5m", days_back=1, end_date=end_date)

            # Verify the end_date was passed correctly
            call_args = mock_client.get_network_data.call_args
            assert call_args[1]["date_end"] == end_date

    def test_get_network_data_column_renaming(self, mock_spark_session, mock_client):
        """Test that columns are properly renamed with units."""
        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            # Call the function
            result = get_network_data(network="NEM", interval="5m", days_back=1)

            # Verify column renaming was called
            mock_dataframe = mock_spark_session.createDataFrame.return_value
            mock_dataframe.withColumnRenamed.assert_called()

            # Check that the expected renames were called
            rename_calls = [call[0] for call in mock_dataframe.withColumnRenamed.call_args_list]
            assert ("power", "power_MW") in rename_calls
            assert ("energy", "energy_MWh") in rename_calls
            assert ("emissions", "emissions_t") in rename_calls
            assert ("market_value", "market_value_aud") in rename_calls

    def test_get_network_data_with_api_key(self, mock_spark_session, mock_client):
        """Test get_network_data with explicit API key."""
        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            # Call the function with explicit API key
            result = get_network_data(network="NEM", interval="5m", days_back=1, api_key="test-api-key")

            # Verify the client was created with the API key
            mock_get_client.assert_called_once()

    def test_get_network_data_different_networks(self, mock_spark_session, mock_client):
        """Test get_network_data with different network codes."""
        networks = ["NEM", "WEM", "AU"]

        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            for network in networks:
                # Reset mock to clear previous calls
                mock_client.get_network_data.reset_mock()

                # Call the function
                result = get_network_data(network=network, interval="5m", days_back=1)

                # Verify the network was passed correctly
                call_args = mock_client.get_network_data.call_args
                assert call_args[1]["network_code"] == network

    def test_get_network_data_different_intervals(self, mock_spark_session, mock_client):
        """Test get_network_data with different intervals."""
        intervals = ["5m", "1h", "1d"]

        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            for interval in intervals:
                # Reset mock to clear previous calls
                mock_client.get_network_data.reset_mock()

                # Call the function
                result = get_network_data(network="NEM", interval=interval, days_back=1)

                # Verify the interval was passed correctly
                call_args = mock_client.get_network_data.call_args
                assert call_args[1]["interval"] == interval

    def test_get_network_data_error_handling(self, mock_spark_session):
        """Test get_network_data error handling."""
        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            # Mock client to raise an exception
            mock_get_client.return_value.__enter__.side_effect = Exception("API Error")

            # Call the function and expect it to raise an exception
            with pytest.raises(Exception, match="API Error"):
                get_network_data(network="NEM", interval="5m", days_back=1)

    @pytest.mark.skipif(True, reason="Requires real API key and internet connection")
    def test_get_network_data_integration(self):
        """Integration test for get_network_data (requires real API key)."""
        if not self.has_api_key:
            pytest.skip("No API key available for integration test")

        # This test requires a real API key and internet connection
        # It's marked as skip by default but can be enabled for manual testing

        # Test with real API call
        result = get_network_data(network="NEM", interval="5m", days_back=1)

        # Verify the result is a Spark DataFrame
        assert result is not None
        assert hasattr(result, "count")

        # Verify it has the expected columns
        columns = result.columns
        assert "interval" in columns
        assert "network_region" in columns
        assert "power_MW" in columns
        assert "energy_MWh" in columns
        assert "market_value_aud" in columns
        assert "emissions_t" in columns

    def test_get_network_data_date_calculation(self, mock_spark_session, mock_client):
        """Test that date range calculation works correctly."""
        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            # Mock datetime.now() to return a fixed time
            with patch("examples.databricks.openelectricity_etl.datetime") as mock_datetime:
                mock_datetime.now.return_value = datetime(2025, 9, 4, 12, 0, 0)

                # Call the function
                result = get_network_data(network="NEM", interval="5m", days_back=2)

                # Verify the date range was calculated correctly
                call_args = mock_client.get_network_data.call_args
                expected_start = datetime(2025, 9, 2, 12, 0, 0)
                expected_end = datetime(2025, 9, 4, 12, 0, 0)

                assert call_args[1]["date_start"] == expected_start
                assert call_args[1]["date_end"] == expected_end

    def test_get_network_data_metric_units_handling(self, mock_spark_session, mock_client):
        """Test handling of different metric units."""
        with patch("examples.databricks.openelectricity_etl._get_client") as mock_get_client:
            mock_get_client.return_value.__enter__.return_value = mock_client

            # Mock different units
            mock_client.get_network_data.return_value.get_metric_units.return_value = {
                "power": "kW",  # Different unit
                "energy": "kWh",  # Different unit
                "market_value": "USD",  # Different unit
                "emissions": "kg",  # Different unit
            }

            # Call the function
            result = get_network_data(network="NEM", interval="5m", days_back=1)

            # Verify column renaming with different units
            mock_dataframe = mock_spark_session.createDataFrame.return_value
            rename_calls = [call[0] for call in mock_dataframe.withColumnRenamed.call_args_list]
            assert ("power", "power_kW") in rename_calls
            assert ("energy", "energy_kWh") in rename_calls
            assert ("emissions", "emissions_kg") in rename_calls
            assert ("market_value", "market_value_aud") in rename_calls  # Always renamed to aud


if __name__ == "__main__":
    # Run integration test if API key is available
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("OPENELECTRICITY_API_KEY")

    if api_key:
        print("🔍 Running integration test...")
        # Create test instance and run integration test
        test_instance = TestGetNetworkData()
        test_instance.has_api_key = True
        test_instance.test_get_network_data_integration()
        print("✅ Integration test completed successfully!")
    else:
        print("⚠️  Skipping integration test - no API key available")
        print("Set OPENELECTRICITY_API_KEY environment variable to run integration tests")
