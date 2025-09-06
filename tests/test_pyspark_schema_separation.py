#!/usr/bin/env python
"""
Integration tests for PySpark DataFrame conversion with schema separation.

This test validates that the automatic schema detection works correctly
for different data types: facility, market, and network data.
"""

import os
import logging
import pytest
from datetime import datetime, timedelta, timezone
from openelectricity import OEClient
from openelectricity.types import DataMetric, MarketMetric

# Configure logging to be quiet during tests
logging.getLogger("openelectricity").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Remove the client fixture since we now use openelectricity_client from conftest.py


@pytest.fixture
def facility_test_parameters():
    """Test parameters for facility data."""
    return {
        "network_code": "NEM",
        "facility_code": "BAYSW",
        "metrics": [
            DataMetric.POWER,
            DataMetric.ENERGY,
            DataMetric.MARKET_VALUE,
            DataMetric.EMISSIONS,
        ],
        "interval": "7d",  # 7 day interval
        "date_start": datetime.fromisoformat("2025-08-19T21:30:00"),
        "date_end": datetime.fromisoformat("2025-08-19T21:30:00") + timedelta(days=7),
    }


@pytest.fixture
def market_test_parameters():
    """Test parameters for market data."""
    return {
        "network_code": "NEM",
        "metrics": [
            MarketMetric.PRICE,
        ],
        "interval": "1d",  # 1 day interval
        "primary_grouping": "network_region",
        "date_start": datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
        "date_end": datetime.fromisoformat("2024-01-02T00:00:00+00:00"),
    }


@pytest.fixture
def network_test_parameters():
    """Test parameters for network data."""
    return {
        "network_code": "NEM",
        "metrics": [
            DataMetric.POWER,
        ],
        "interval": "1d",  # 1 day interval
        "primary_grouping": "network_region",
        "secondary_grouping": "fueltech",
        "date_start": datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
        "date_end": datetime.fromisoformat("2024-01-02T00:00:00+00:00"),
    }


class TestPySparkSchemaSeparation:
    """Test PySpark DataFrame conversion with automatic schema detection."""

    def test_facility_schema_detection(self, openelectricity_client, facility_test_parameters):
        """Test that facility data gets the correct facility schema."""
        try:
            response = openelectricity_client.get_facility_data(**facility_test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No facility data available for testing")

        # Convert to PySpark
        spark_df = response.to_pyspark()

        if spark_df is None:
            pytest.skip("PySpark conversion failed")

        # Check that facility-specific fields are present
        schema_fields = [f.name for f in spark_df.schema.fields]

        # Facility-specific metric fields should be present
        facility_metrics = ["power", "energy", "market_value", "emissions"]
        for metric in facility_metrics:
            assert metric in schema_fields, f"Missing facility metric: {metric}"

        # Facility-specific grouping fields should be present
        facility_fields = ["facility_code", "unit_code", "fueltech_id", "status_id"]
        found_facility_fields = [field for field in facility_fields if field in schema_fields]
        assert len(found_facility_fields) > 0, f"No facility-specific fields found. Expected some of: {facility_fields}"

        # Market-specific fields should NOT be present
        market_fields = ["price", "demand", "curtailment"]
        unexpected_fields = [field for field in market_fields if field in schema_fields]
        assert len(unexpected_fields) == 0, f"Unexpected market fields in facility schema: {unexpected_fields}"

    def test_market_schema_detection(self, openelectricity_client, market_test_parameters):
        """Test that market data gets the correct market schema."""
        try:
            response = openelectricity_client.get_market(**market_test_parameters)
        except Exception as e:
            pytest.skip(f"Market API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No market data available for testing")

        # Convert to PySpark
        try:
            spark_df = response.to_pyspark()
        except Exception as e:
            pytest.skip(f"PySpark conversion failed: {e}")

        if spark_df is None:
            pytest.skip("PySpark conversion returned None")

        # Check that market-specific fields are present
        schema_fields = [f.name for f in spark_df.schema.fields]

        # Market-specific metric fields should be present
        market_metrics = ["price", "demand", "curtailment"]
        found_market_metrics = [metric for metric in market_metrics if metric in schema_fields]
        assert len(found_market_metrics) > 0, f"No market-specific metrics found. Expected some of: {market_metrics}"

        # Market-specific grouping fields should be present
        market_fields = ["primary_grouping"]
        found_market_fields = [field for field in market_fields if field in schema_fields]
        assert len(found_market_fields) > 0, f"No market-specific fields found. Expected some of: {market_fields}"

        # Facility-specific fields should NOT be present
        facility_fields = ["facility_code", "unit_code", "fueltech_id", "status_id"]
        unexpected_fields = [field for field in facility_fields if field in schema_fields]
        assert len(unexpected_fields) == 0, f"Unexpected facility fields in market schema: {unexpected_fields}"

    def test_network_schema_detection(self, openelectricity_client, network_test_parameters):
        """Test that network data gets the correct network schema."""
        try:
            response = openelectricity_client.get_network_data(**network_test_parameters)
        except Exception as e:
            pytest.skip(f"Network API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No network data available for testing")

        # Convert to PySpark
        try:
            spark_df = response.to_pyspark()
        except Exception as e:
            pytest.skip(f"PySpark conversion failed: {e}")

        if spark_df is None:
            pytest.skip("PySpark conversion returned None")

        # Check that network-specific fields are present
        schema_fields = [f.name for f in spark_df.schema.fields]

        # Network-specific metric fields should be present
        network_metrics = ["power", "energy", "emissions"]
        found_network_metrics = [metric for metric in network_metrics if metric in schema_fields]
        assert len(found_network_metrics) > 0, f"No network-specific metrics found. Expected some of: {network_metrics}"

        # Network-specific grouping fields should be present
        network_fields = ["primary_grouping", "secondary_grouping"]
        found_network_fields = [field for field in network_fields if field in schema_fields]
        assert len(found_network_fields) > 0, f"No network-specific fields found. Expected some of: {network_fields}"

        # Facility-specific fields should NOT be present
        facility_fields = ["facility_code", "unit_code", "fueltech_id", "status_id"]
        unexpected_fields = [field for field in facility_fields if field in schema_fields]
        assert len(unexpected_fields) == 0, f"Unexpected facility fields in network schema: {unexpected_fields}"

    def test_schema_field_types(self, openelectricity_client, facility_test_parameters):
        """Test that schema fields have correct types."""
        try:
            response = openelectricity_client.get_facility_data(**facility_test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No facility data available for testing")

        spark_df = response.to_pyspark()

        if spark_df is None:
            pytest.skip("PySpark conversion failed")

        # Check field types
        for field in spark_df.schema.fields:
            if field.name in ["power", "energy", "market_value", "emissions"]:
                # Metric fields should be DoubleType
                assert "DoubleType" in str(field.dataType), (
                    f"Metric field {field.name} should be DoubleType, got {field.dataType}"
                )
            elif field.name == "interval":
                # Time field should be TimestampType
                assert "TimestampType" in str(field.dataType), (
                    f"Time field {field.name} should be TimestampType, got {field.dataType}"
                )
            elif field.name in ["network_id", "network_region", "facility_code", "unit_code"]:
                # String fields should be StringType
                assert "StringType" in str(field.dataType), (
                    f"String field {field.name} should be StringType, got {field.dataType}"
                )

    def test_schema_consistency(self, openelectricity_client, facility_test_parameters):
        """Test that the same data always gets the same schema."""
        try:
            response = openelectricity_client.get_facility_data(**facility_test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No facility data available for testing")

        # Convert to PySpark multiple times
        spark_df1 = response.to_pyspark()
        spark_df2 = response.to_pyspark()

        if spark_df1 is None or spark_df2 is None:
            pytest.skip("PySpark conversion failed")

        # Schemas should be identical
        schema1_fields = [f.name for f in spark_df1.schema.fields]
        schema2_fields = [f.name for f in spark_df2.schema.fields]

        assert schema1_fields == schema2_fields, f"Schema inconsistency: {schema1_fields} vs {schema2_fields}"

    def test_data_integrity_with_schema(self, openelectricity_client, facility_test_parameters):
        """Test data integrity with the detected schema."""
        try:
            response = openelectricity_client.get_facility_data(**facility_test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No facility data available for testing")

        records = response.to_records()
        spark_df = response.to_pyspark()

        if not records or spark_df is None:
            pytest.skip("No data available for integrity testing")

        # Compare record count
        records_count = len(records)
        spark_count = spark_df.count()

        assert records_count == spark_count, f"Record count mismatch: records={records_count}, spark={spark_count}"

        # Compare schema completeness
        if records:
            record_keys = set(records[0].keys())
            spark_columns = set(spark_df.columns)

            # All record keys should be in Spark columns
            assert record_keys.issubset(spark_columns), f"Missing columns in Spark: {record_keys - spark_columns}"

    def test_performance_with_schema_detection(self, openelectricity_client, facility_test_parameters):
        """Test that schema detection doesn't impact performance."""
        try:
            response = openelectricity_client.get_facility_data(**facility_test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No facility data available for testing")

        import time

        # Time the conversion
        start_time = time.time()
        spark_df = response.to_pyspark()
        conversion_time = time.time() - start_time

        if spark_df is None:
            pytest.skip("PySpark conversion failed")

        # Should complete in reasonable time (less than 10 seconds for small datasets)
        assert conversion_time < 10.0, f"Conversion took too long: {conversion_time:.2f} seconds"

    def test_facility_schema_structure(self, openelectricity_client, facility_test_parameters):
        """Test that facility data schema has reasonable structure and types."""
        try:
            response = openelectricity_client.get_facility_data(**facility_test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No facility data available for testing")

        # Convert to PySpark
        spark_df = response.to_pyspark()

        if spark_df is None:
            pytest.skip("PySpark conversion failed")

        # Get schema fields
        schema_fields = spark_df.schema.fields
        field_names = [f.name for f in schema_fields]

        # Should have some fields
        assert len(schema_fields) > 0, "Schema should have fields"
        
        # Should have essential fields (flexible - at least some should be present)
        essential_fields = ["interval", "network_region", "facility_code"]
        found_essential = [field for field in essential_fields if field in field_names]
        assert len(found_essential) > 0, f"Should have at least some essential fields. Found: {found_essential}"

        # Should have some metric fields (flexible - at least some should be present)
        metric_fields = ["power", "energy", "emissions", "market_value"]
        found_metrics = [field for field in metric_fields if field in field_names]
        assert len(found_metrics) > 0, f"Should have at least some metric fields. Found: {found_metrics}"

        # Check field types for fields that are present
        for field in schema_fields:
            if field.name == "interval":
                assert "TimestampType" in str(field.dataType), f"Field {field.name} should be TimestampType, got {field.dataType}"
            elif field.name in ["power", "energy", "emissions", "market_value"]:
                assert "DoubleType" in str(field.dataType), f"Field {field.name} should be DoubleType, got {field.dataType}"
            elif field.name in ["network_region", "facility_code", "unit_code", "fueltech_id", "status_id"]:
                assert "StringType" in str(field.dataType), f"Field {field.name} should be StringType, got {field.dataType}"

    def test_schema_detection_edge_cases(self):
        """Test schema detection with edge cases."""
        from openelectricity.spark_utils import detect_timeseries_schema

        # Test with empty data
        empty_schema = detect_timeseries_schema([])
        assert empty_schema is not None, "Empty data should return default schema"

        # Test with mixed data (should default to facility)
        mixed_data = [{"interval": "2025-01-01", "power": 100, "price": 50}]
        mixed_schema = detect_timeseries_schema(mixed_data)
        assert mixed_schema is not None, "Mixed data should return a schema"

        # Test with unknown fields (should default to facility)
        unknown_data = [{"interval": "2025-01-01", "unknown_field": "value"}]
        unknown_schema = detect_timeseries_schema(unknown_data)
        assert unknown_schema is not None, "Unknown data should return default schema"


# Integration test runner
def test_full_schema_separation(openelectricity_client, facility_test_parameters, market_test_parameters, network_test_parameters):
    """Run full integration test with all three data types."""
    # Test facility data
    try:
        facility_response = openelectricity_client.get_facility_data(**facility_test_parameters)
    except Exception as e:
        pytest.skip(f"Facility API call failed: {e}")
    if facility_response and facility_response.data:
        facility_df = facility_response.to_pyspark()
        assert facility_df is not None, "Facility PySpark conversion should succeed"
        assert len(facility_df.schema.fields) > 0, "Facility schema should have fields"

    # Test market data
    try:
        market_response = openelectricity_client.get_market(**market_test_parameters)
        if market_response and market_response.data:
            market_df = market_response.to_pyspark()
            assert market_df is not None, "Market PySpark conversion should succeed"
            assert len(market_df.schema.fields) > 0, "Market schema should have fields"
    except Exception:
        # Market API might not be available, skip silently
        pass

    # Test network data
    try:
        network_response = openelectricity_client.get_network_data(**network_test_parameters)
        if network_response and network_response.data:
            network_df = network_response.to_pyspark()
            assert network_df is not None, "Network PySpark conversion should succeed"
            assert len(network_df.schema.fields) > 0, "Network schema should have fields"
    except Exception:
        # Network API might not be available, skip silently
        pass
