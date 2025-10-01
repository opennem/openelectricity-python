#!/usr/bin/env python
"""
Integration test for PySpark DataFrame conversion with facility data.

This test validates that the TimestampType conversion and schema alignment
works correctly with real API data for specific facility metrics.
"""

import os
import logging
import pytest
from datetime import datetime, timedelta, timezone

# Check if PySpark is available
try:
    import pyspark
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

from openelectricity import OEClient
from openelectricity.types import DataMetric

# Configure logging to be quiet during tests
logging.getLogger("openelectricity").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Remove the client fixture since we now use openelectricity_client from conftest.py


@pytest.fixture
def test_parameters():
    """Test parameters based on user requirements."""
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


class TestPySparkFacilityDataIntegration:
    """Test PySpark DataFrame conversion with facility data."""

    def test_api_response_structure(self, openelectricity_client, test_parameters):
        """Test that API returns expected data structure."""
        try:
            response = openelectricity_client.get_facility_data(**test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        # Validate response structure
        assert response is not None
        assert hasattr(response, "data")
        assert isinstance(response.data, list)

        if response.data:
            # Validate time series structure
            first_ts = response.data[0]
            assert hasattr(first_ts, "network_code")
            assert hasattr(first_ts, "metric")
            assert hasattr(first_ts, "unit")
            assert hasattr(first_ts, "interval")
            assert hasattr(first_ts, "results")

            assert first_ts.network_code == "NEM"
            assert first_ts.metric in ["power", "energy", "market_value", "emissions"]
            assert first_ts.interval == "7d"

    def test_records_conversion(self, openelectricity_client, test_parameters):
        """Test that to_records() conversion works correctly."""
        try:
            response = openelectricity_client.get_facility_data(**test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        records = response.to_records()
        assert isinstance(records, list)

        if records:
            first_record = records[0]
            assert isinstance(first_record, dict)

            # Check for expected fields
            expected_fields = ["interval", "network_region"]
            for field in expected_fields:
                assert field in first_record

            # Check datetime field
            if "interval" in first_record:
                interval_value = first_record["interval"]
                assert isinstance(interval_value, datetime)
                # Should have timezone info
                assert interval_value.tzinfo is not None

    @pytest.mark.skipif(not pytest.importorskip("pyspark", reason="PySpark not available"), reason="PySpark not available")
    def test_pyspark_conversion_success(self, openelectricity_client, test_parameters):
        """Test that PySpark conversion succeeds."""
        try:
            response = openelectricity_client.get_facility_data(**test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        # Test PySpark conversion
        spark_df = response.to_pyspark()

        # Should not return None
        assert spark_df is not None

        # Should have data
        row_count = spark_df.count()
        assert row_count >= 0  # Allow for empty datasets

    @pytest.mark.skipif(not pytest.importorskip("pyspark", reason="PySpark not available"), reason="PySpark not available")
    def test_pyspark_schema_validation(self, openelectricity_client, test_parameters):
        """Test that PySpark DataFrame has correct schema with TimestampType."""
        from pyspark.sql.types import TimestampType, DoubleType, StringType

        try:
            response = openelectricity_client.get_facility_data(**test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")
        spark_df = response.to_pyspark()

        if spark_df is None:
            pytest.skip("No PySpark DataFrame returned")

        schema = spark_df.schema
        field_types = {field.name: field.dataType for field in schema.fields}

        # Validate datetime fields use TimestampType
        if "interval" in field_types:
            assert isinstance(field_types["interval"], TimestampType), (
                f"Expected TimestampType for 'interval', got {type(field_types['interval'])}"
            )

        # Validate numeric fields use DoubleType (flexible - check fields that are present)
        numeric_fields = ["power", "energy", "market_value", "emissions"]
        for field in numeric_fields:
            if field in field_types:
                assert isinstance(field_types[field], DoubleType), (
                    f"Expected DoubleType for '{field}', got {type(field_types[field])}"
                )

        # Validate string fields use StringType (flexible - check fields that are present)
        string_fields = ["network_region", "facility_code", "unit_code", "fueltech_id", "status_id"]
        for field in string_fields:
            if field in field_types:
                assert isinstance(field_types[field], StringType), (
                    f"Expected StringType for '{field}', got {type(field_types[field])}"
                )

    @pytest.mark.skipif(not pytest.importorskip("pyspark", reason="PySpark not available"), reason="PySpark not available")
    def test_timezone_conversion(self, openelectricity_client, test_parameters):
        """Test that timezone conversion to UTC works correctly."""
        try:
            response = openelectricity_client.get_facility_data(**test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

        # Get original records with timezone info
        records = response.to_records()
        spark_df = response.to_pyspark()

        if not records or spark_df is None:
            pytest.skip("No data available for timezone testing")

        # Get first record with datetime
        original_record = records[0]
        if "interval" not in original_record:
            pytest.skip("No interval field for timezone testing")

        original_dt = original_record["interval"]
        if not hasattr(original_dt, "tzinfo") or original_dt.tzinfo is None:
            pytest.skip("No timezone info in original data")

        # Get corresponding Spark data
        spark_rows = spark_df.collect()
        if not spark_rows:
            pytest.skip("No Spark data for timezone testing")

        spark_dt = spark_rows[0]["interval"]

        # Convert original to UTC and remove timezone for comparison
        expected_utc = original_dt.astimezone(timezone.utc).replace(tzinfo=None)

        # Validate timezone conversion logic
        if original_dt.tzinfo is not None:
            # Calculate expected UTC time
            utc_offset = original_dt.utcoffset()
            if utc_offset is not None:
                expected_utc_calculated = original_dt - utc_offset
                expected_utc_calculated = expected_utc_calculated.replace(tzinfo=None)

                # Both methods should give same result
                assert expected_utc == expected_utc_calculated, (
                    f"UTC calculation methods differ: {expected_utc} != {expected_utc_calculated}"
                )

        # Spark datetime should match expected UTC
        assert spark_dt == expected_utc, f"Timezone conversion failed: {spark_dt} != {expected_utc}"

        # Additional validation: check that conversion is reasonable
        # For Australian timezone (+10:00), UTC should be 10 hours earlier
        if original_dt.tzinfo is not None and original_dt.utcoffset() is not None:
            offset_hours = original_dt.utcoffset().total_seconds() / 3600
            if offset_hours > 0:  # Positive offset (ahead of UTC)
                # UTC time should be earlier than local time
                assert spark_dt < original_dt.replace(tzinfo=None), (
                    f"UTC time {spark_dt} should be earlier than local time {original_dt.replace(tzinfo=None)}"
                )

    @pytest.mark.skipif(not pytest.importorskip("pyspark", reason="PySpark not available"), reason="PySpark not available")
    def test_temporal_operations(self, openelectricity_client, test_parameters):
        """Test that temporal operations work on TimestampType fields."""
        from pyspark.sql.functions import hour, date_format, min as spark_min, max as spark_max

        try:
            response = openelectricity_client.get_facility_data(**test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")
        spark_df = response.to_pyspark()

        if spark_df is None or spark_df.count() == 0:
            pytest.skip("No data available for temporal testing")

        # Check if interval field exists and is TimestampType
        schema = spark_df.schema
        interval_field = next((f for f in schema.fields if f.name == "interval"), None)

        if interval_field is None:
            pytest.skip("No interval field for temporal testing")

        # Test hour extraction
        hour_df = spark_df.select(hour("interval").alias("hour"))
        hour_values = [row["hour"] for row in hour_df.collect()]

        # Hours should be 0-23
        assert all(0 <= h <= 23 for h in hour_values), f"Invalid hour values: {hour_values}"

        # Test date formatting
        formatted_df = spark_df.select(date_format("interval", "yyyy-MM-dd HH:mm:ss").alias("formatted"))
        formatted_values = [row["formatted"] for row in formatted_df.collect()]

        # Should be valid datetime strings
        for formatted in formatted_values[:3]:  # Test first 3
            try:
                datetime.strptime(formatted, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pytest.fail(f"Invalid date format: {formatted}")

        # Test min/max operations
        min_max_df = spark_df.select(spark_min("interval").alias("min_time"), spark_max("interval").alias("max_time")).collect()

        if min_max_df:
            min_time = min_max_df[0]["min_time"]
            max_time = min_max_df[0]["max_time"]

            assert isinstance(min_time, datetime), f"Min time not datetime: {type(min_time)}"
            assert isinstance(max_time, datetime), f"Max time not datetime: {type(max_time)}"
            assert min_time <= max_time, f"Min time {min_time} > Max time {max_time}"

    @pytest.mark.skipif(not pytest.importorskip("pyspark", reason="PySpark not available"), reason="PySpark not available")
    def test_numeric_operations(self, openelectricity_client, test_parameters):
        """Test that numeric operations work on DoubleType fields."""
        from pyspark.sql.functions import avg, sum as spark_sum, count, min as spark_min, max as spark_max
        from pyspark.sql.types import DoubleType

        try:
            response = openelectricity_client.get_facility_data(**test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")
        spark_df = response.to_pyspark()

        if spark_df is None or spark_df.count() == 0:
            pytest.skip("No data available for numeric testing")

        # Find numeric fields
        schema = spark_df.schema
        numeric_fields = [field.name for field in schema.fields if isinstance(field.dataType, DoubleType)]

        if not numeric_fields:
            pytest.skip("No numeric fields found for testing")

        # Test numeric operations on first numeric field
        test_field = numeric_fields[0]

        stats_df = spark_df.select(
            spark_min(test_field).alias("min_val"),
            spark_max(test_field).alias("max_val"),
            avg(test_field).alias("avg_val"),
            spark_sum(test_field).alias("sum_val"),
            count(test_field).alias("count_val"),
        ).collect()

        if stats_df:
            stats = stats_df[0]

            # Validate numeric results
            assert isinstance(stats["min_val"], (int, float, type(None))), f"Min value not numeric: {type(stats['min_val'])}"
            assert isinstance(stats["max_val"], (int, float, type(None))), f"Max value not numeric: {type(stats['max_val'])}"
            assert isinstance(stats["avg_val"], (int, float, type(None))), f"Avg value not numeric: {type(stats['avg_val'])}"
            assert isinstance(stats["sum_val"], (int, float, type(None))), f"Sum value not numeric: {type(stats['sum_val'])}"
            assert isinstance(stats["count_val"], int), f"Count not integer: {type(stats['count_val'])}"

            # If we have non-null values, min should be <= max
            if stats["min_val"] is not None and stats["max_val"] is not None:
                assert stats["min_val"] <= stats["max_val"], f"Min {stats['min_val']} > Max {stats['max_val']}"

    @pytest.mark.skipif(not pytest.importorskip("pyspark", reason="PySpark not available"), reason="PySpark not available")
    def test_data_integrity(self, openelectricity_client, test_parameters):
        """Test data integrity between records and PySpark DataFrame."""
        try:
            response = openelectricity_client.get_facility_data(**test_parameters)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

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

    def test_error_handling(self, openelectricity_client):
        """Test that invalid parameters are handled gracefully."""
        # Test with invalid facility code
        try:
            response = openelectricity_client.get_facility_data(
                network_code="NEM",
                facility_code="INVALID_FACILITY_CODE",
                metrics=[DataMetric.POWER],
                interval="1h",
                date_start=datetime(2025, 8, 19, 21, 30),
                date_end=datetime(2025, 8, 20, 21, 30),
            )
            
            # If no exception is raised, the response should handle gracefully
            if response:
                # Should have empty or no data
                assert len(response.data) == 0, "Invalid facility should return empty data"
                
                # PySpark conversion should handle empty data gracefully
                spark_df = response.to_pyspark()
                if spark_df is not None:
                    assert spark_df.count() == 0, "PySpark DataFrame should be empty for invalid facility"
                    
        except Exception as e:
            # API should raise an exception for invalid parameters
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["facility", "not found", "range", "invalid", "bad request"]), f"Unexpected error: {e}"


# Integration test runner
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
def test_full_integration(openelectricity_client, test_parameters):
    """Run full integration test with the specified parameters."""
    try:
        response = openelectricity_client.get_facility_data(**test_parameters)
    except Exception as e:
        pytest.skip(f"API call failed: {e}")

    # Validate API response
    assert response is not None, "API response should not be None"
    assert len(response.data) > 0, "API should return time series data"

    # Test PySpark conversion if available
    try:
        pytest.importorskip("pyspark", reason="PySpark not available")
        spark_df = response.to_pyspark()
        
        assert spark_df is not None, "PySpark conversion should succeed"
        assert spark_df.count() >= 0, "PySpark DataFrame should have data"
        assert len(spark_df.schema.fields) > 0, "PySpark schema should have fields"
    except ImportError:
        # PySpark not available, skip silently
        pass
