"""
Tests for facilities PySpark conversion.

This module contains tests for converting facility data to PySpark DataFrames.
"""

import os
import pytest

# Check if PySpark is available
try:
    import pyspark
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

from openelectricity import OEClient


@pytest.fixture
def facilities_response(openelectricity_client):
    """Get facilities response for testing."""
    try:
        return openelectricity_client.get_facilities(network_region="NSW1")
    except Exception as e:
        pytest.skip(f"API call failed: {e}")


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
def test_facilities_pyspark_conversion(facilities_response):
    """Test that facilities can be converted to PySpark DataFrame."""
    # Test PySpark conversion
    spark_df = facilities_response.to_pyspark()
    
    # Should not return None
    assert spark_df is not None, "PySpark DataFrame should not be None"
    
    # Should have data
    row_count = spark_df.count()
    assert row_count > 0, f"Expected data rows, got {row_count}"
    
    # Should have some essential columns (flexible - at least some should be present)
    essential_columns = ["code", "name", "network_id", "network_region"]
    found_columns = [col for col in essential_columns if col in spark_df.columns]
    assert len(found_columns) > 0, f"Should have at least some essential columns. Found: {found_columns}"


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
def test_facilities_pyspark_schema(facilities_response):
    """Test that PySpark DataFrame has correct schema."""
    from pyspark.sql.types import StringType, DoubleType
    
    spark_df = facilities_response.to_pyspark()
    assert spark_df is not None, "PySpark DataFrame should not be None"
    
    schema = spark_df.schema
    field_types = {field.name: field.dataType for field in schema.fields}
    
    # Check string fields
    string_fields = ["code", "name", "network_id", "network_region"]
    for field in string_fields:
        if field in field_types:
            assert isinstance(field_types[field], StringType), (
                f"Field {field} should be StringType, got {field_types[field]}"
            )
    
    # Check numeric fields if present
    numeric_fields = ["capacity_registered", "emissions_factor_co2"]
    for field in numeric_fields:
        if field in field_types:
            assert isinstance(field_types[field], DoubleType), (
                f"Field {field} should be DoubleType, got {field_types[field]}"
            )


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
def test_facilities_pyspark_operations(facilities_response):
    """Test that PySpark operations work on facilities DataFrame."""
    spark_df = facilities_response.to_pyspark()
    assert spark_df is not None, "PySpark DataFrame should not be None"
    
    # Test basic operations
    total_count = spark_df.count()
    assert total_count > 0, "Should have facilities data"
    
    # Test grouping operations
    if "fueltech_id" in spark_df.columns:
        fueltech_counts = spark_df.groupBy("fueltech_id").count()
        fueltech_count = fueltech_counts.count()
        assert fueltech_count > 0, "Should have fuel technology groups"
    
    # Test filtering
    if "network_id" in spark_df.columns:
        nem_facilities = spark_df.filter(spark_df.network_id == "NEM")
        nem_count = nem_facilities.count()
        assert nem_count > 0, "Should have NEM facilities"


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
def test_facilities_pyspark_data_integrity(facilities_response):
    """Test data integrity between pandas and PySpark DataFrames."""
    # Get pandas DataFrame for comparison
    pandas_df = facilities_response.to_pandas()
    spark_df = facilities_response.to_pyspark()
    
    assert spark_df is not None, "PySpark DataFrame should not be None"
    
    # Compare row counts
    pandas_count = len(pandas_df)
    spark_count = spark_df.count()
    assert pandas_count == spark_count, f"Row count mismatch: pandas={pandas_count}, spark={spark_count}"
    
    # Compare column counts
    pandas_cols = set(pandas_df.columns)
    spark_cols = set(spark_df.columns)
    assert pandas_cols == spark_cols, f"Column mismatch: pandas={pandas_cols}, spark={spark_cols}"


def test_facilities_pandas_conversion(facilities_response):
    """Test that facilities can be converted to pandas DataFrame."""
    pandas_df = facilities_response.to_pandas()
    
    # Should have data
    assert len(pandas_df) > 0, "Pandas DataFrame should have data"
    
    # Should have some essential columns (flexible - at least some should be present)
    essential_columns = ["code", "name", "network_id", "network_region"]
    found_columns = [col for col in essential_columns if col in pandas_df.columns]
    assert len(found_columns) > 0, f"Should have at least some essential columns. Found: {found_columns}"


def test_pyspark_unavailable_handling(facilities_response):
    """Test that PySpark methods handle unavailability gracefully."""
    # This test should work even without PySpark installed
    try:
        spark_df = facilities_response.to_pyspark()
        # If PySpark is available, should return a DataFrame or None
        if spark_df is not None:
            assert hasattr(spark_df, 'count'), "Should return a PySpark DataFrame"
    except ImportError:
        # PySpark not available - this is expected
        pass
    except Exception as e:
        # Other errors should be handled gracefully
        assert "pyspark" in str(e).lower() or "spark" in str(e).lower(), f"Unexpected error: {e}"
