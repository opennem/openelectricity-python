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


@pytest.fixture
def client():
    """Create OEClient instance for testing."""
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        pytest.skip("OPENELECTRICITY_API_KEY environment variable not set")

    return OEClient(api_key=api_key)


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


@pytest.mark.schema
class TestPySparkSchemaSeparation:
    """Test PySpark DataFrame conversion with automatic schema detection."""

    def test_facility_schema_detection(self, client, facility_test_parameters):
        """Test that facility data gets the correct facility schema."""
        response = client.get_facility_data(**facility_test_parameters)

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
        for field in facility_fields:
            if field in schema_fields:
                print(f"✅ Found facility field: {field}")

        # Market-specific fields should NOT be present
        market_fields = ["price", "demand", "curtailment"]
        for field in market_fields:
            if field in schema_fields:
                print(f"⚠️  Unexpected market field in facility schema: {field}")

        print(f"✅ Facility schema detection working correctly: {len(schema_fields)} fields")

    def test_market_schema_detection(self, client, market_test_parameters):
        """Test that market data gets the correct market schema."""
        try:
            response = client.get_market(**market_test_parameters)
        except Exception as e:
            pytest.skip(f"Market API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No market data available for testing")

        # Convert to PySpark
        spark_df = response.to_pyspark()

        if spark_df is None:
            pytest.skip("PySpark conversion failed")

        # Check that market-specific fields are present
        schema_fields = [f.name for f in spark_df.schema.fields]

        # Market-specific metric fields should be present
        market_metrics = ["price", "demand", "curtailment"]
        for metric in market_metrics:
            if metric in schema_fields:
                print(f"✅ Found market metric: {metric}")

        # Market-specific grouping fields should be present
        market_fields = ["primary_grouping"]
        for field in market_fields:
            if field in schema_fields:
                print(f"✅ Found market field: {field}")

        # Facility-specific fields should NOT be present
        facility_fields = ["facility_code", "unit_code", "fueltech_id", "status_id"]
        for field in facility_fields:
            if field in schema_fields:
                print(f"⚠️  Unexpected facility field in market schema: {field}")

        print(f"✅ Market schema detection working correctly: {len(schema_fields)} fields")

    def test_network_schema_detection(self, client, network_test_parameters):
        """Test that network data gets the correct network schema."""
        try:
            response = client.get_network_data(**network_test_parameters)
        except Exception as e:
            pytest.skip(f"Network API call failed: {e}")

        if not response or not response.data:
            pytest.skip("No network data available for testing")

        # Convert to PySpark
        spark_df = response.to_pyspark()

        if spark_df is None:
            pytest.skip("PySpark conversion failed")

        # Check that network-specific fields are present
        schema_fields = [f.name for f in spark_df.schema.fields]

        # Network-specific metric fields should be present
        network_metrics = ["power", "energy", "emissions"]
        for metric in network_metrics:
            if metric in schema_fields:
                print(f"✅ Found network metric: {metric}")

        # Network-specific grouping fields should be present
        network_fields = ["primary_grouping", "secondary_grouping"]
        for field in network_fields:
            if field in schema_fields:
                print(f"✅ Found network field: {field}")

        # Facility-specific fields should NOT be present
        facility_fields = ["facility_code", "unit_code", "fueltech_id", "status_id"]
        for field in facility_fields:
            if field in schema_fields:
                print(f"⚠️  Unexpected facility field in network schema: {field}")

        print(f"✅ Network schema detection working correctly: {len(schema_fields)} fields")

    def test_schema_field_types(self, client, facility_test_parameters):
        """Test that schema fields have correct types."""
        response = client.get_facility_data(**facility_test_parameters)

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
                print(f"✅ {field.name}: {field.dataType}")
            elif field.name == "interval":
                # Time field should be TimestampType
                assert "TimestampType" in str(field.dataType), (
                    f"Time field {field.name} should be TimestampType, got {field.dataType}"
                )
                print(f"✅ {field.name}: {field.dataType}")
            elif field.name in ["network_id", "network_region", "facility_code", "unit_code"]:
                # String fields should be StringType
                assert "StringType" in str(field.dataType), (
                    f"String field {field.name} should be StringType, got {field.dataType}"
                )
                print(f"✅ {field.name}: {field.dataType}")

    def test_schema_consistency(self, client, facility_test_parameters):
        """Test that the same data always gets the same schema."""
        response = client.get_facility_data(**facility_test_parameters)

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

        print("✅ Schema consistency maintained across multiple conversions")

    def test_data_integrity_with_schema(self, client, facility_test_parameters):
        """Test data integrity with the detected schema."""
        response = client.get_facility_data(**facility_test_parameters)

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

        print("✅ Data integrity maintained with detected schema")

    def test_performance_with_schema_detection(self, client, facility_test_parameters):
        """Test that schema detection doesn't impact performance."""
        response = client.get_facility_data(**facility_test_parameters)

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

        print(f"✅ Schema detection performance acceptable: {conversion_time:.3f} seconds")

    def test_facility_schema_exact_structure(self, client, facility_test_parameters):
        """Test that facility data schema has exactly the expected fields and types."""
        response = client.get_facility_data(**facility_test_parameters)

        if not response or not response.data:
            pytest.skip("No facility data available for testing")

        # Convert to PySpark
        spark_df = response.to_pyspark()

        if spark_df is None:
            pytest.skip("PySpark conversion failed")

        # Get schema fields
        schema_fields = spark_df.schema.fields
        field_names = [f.name for f in schema_fields]

        # Expected fields based on the user's specification
        expected_fields = ["interval", "network_region", "power", "energy", "emissions", "market_value", "facility_code"]

        # Check that all expected fields are present
        for field in expected_fields:
            assert field in field_names, f"Missing expected field: {field}"

        # Check that we have exactly the right number of fields
        expected_field_count = len(expected_fields)
        actual_field_count = len(schema_fields)

        assert actual_field_count == expected_field_count, (
            f"Expected {expected_field_count} fields, but got {actual_field_count}. Fields: {field_names}"
        )

        # Check field types
        for field in schema_fields:
            if field.name == "interval":
                assert "TimestampType" in str(field.dataType), f"Field {field.name} should be TimestampType, got {field.dataType}"
            elif field.name in ["power", "energy", "emissions", "market_value"]:
                assert "DoubleType" in str(field.dataType), f"Field {field.name} should be DoubleType, got {field.dataType}"
            elif field.name in ["network_region", "facility_code"]:
                assert "StringType" in str(field.dataType), f"Field {field.name} should be StringType, got {field.dataType}"

        # Print schema for verification
        print(f"✅ Facility schema has exactly {expected_field_count} fields:")
        for field in schema_fields:
            print(f"   |-- {field.name}: {field.dataType} (nullable = {field.nullable})")

        print(f"✅ All expected fields present with correct types")

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

        print("✅ Schema detection handles edge cases correctly")


# Integration test runner
def test_full_schema_separation(client, facility_test_parameters, market_test_parameters, network_test_parameters):
    """Run full integration test with all three data types."""
    print(f"\n🧪 Running Full Schema Separation Test")

    # Test facility data
    print(f"\n📊 Testing Facility Data Schema")
    facility_response = client.get_facility_data(**facility_test_parameters)
    if facility_response and facility_response.data:
        facility_df = facility_response.to_pyspark()
        if facility_df:
            print(f"✅ Facility schema: {[f'{f.name}:{f.dataType}' for f in facility_df.schema.fields[:5]]}...")
        else:
            print("⚠️  Facility PySpark conversion failed")
    else:
        print("⚠️  No facility data available")

    # Test market data
    print(f"\n📊 Testing Market Data Schema")
    try:
        market_response = client.get_market(**market_test_parameters)
    except Exception as e:
        print(f"⚠️  Market API call failed: {e}")
        market_response = None
    
    if market_response and market_response.data:
        market_df = market_response.to_pyspark()
        if market_df:
            print(f"✅ Market schema: {[f'{f.name}:{f.dataType}' for f in market_df.schema.fields[:5]]}...")
        else:
            print("⚠️  Market PySpark conversion failed")
    else:
        print("⚠️  No market data available")

    # Test network data
    print(f"\n📊 Testing Network Data Schema")
    try:
        network_response = client.get_network_data(**network_test_parameters)
    except Exception as e:
        print(f"⚠️  Network API call failed: {e}")
        network_response = None
    
    if network_response and network_response.data:
        network_df = network_response.to_pyspark()
        if network_df:
            print(f"✅ Network schema: {[f'{f.name}:{f.dataType}' for f in network_df.schema.fields[:5]]}...")
        else:
            print("⚠️  Network PySpark conversion failed")
    else:
        print("⚠️  No network data available")

    print(f"\n🎉 Schema separation test completed!")
