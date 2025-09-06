#!/usr/bin/env python3
"""
Test script to identify which market metrics work with the get_market method.
"""

import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import pytest

from openelectricity import AsyncOEClient
from openelectricity.settings_schema import settings
from openelectricity.types import MarketMetric
from openelectricity.models.timeseries import TimeSeriesResponse

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def market_metric_response():
    yield {
        "version": "4.2.4",
        "created_at": "2025-09-04T21:57:40+10:00",
        "success": True,
        "error": None,
        "data": [
            {
                "network_code": "NEM",
                "metric": "price",
                "unit": "$/MWh",
                "interval": "1d",
                "date_start": "2025-07-29T00:00:00+10:00",
                "date_end": "2025-08-04T00:00:00+10:00",
                "groupings": [],
                "results": [
                    {
                        "name": "price_NSW1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "NSW1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 162.13802],
                            ["2025-07-30T00:00:00+10:00", 125.86097],
                            ["2025-07-31T00:00:00+10:00", 123.49375],
                            ["2025-08-01T00:00:00+10:00", 153.47354],
                            ["2025-08-02T00:00:00+10:00", 115.38187],
                            ["2025-08-03T00:00:00+10:00", 60.647361],
                            ["2025-08-04T00:00:00+10:00", 86.333785],
                        ],
                    },
                    {
                        "name": "price_QLD1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "QLD1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 144.41337],
                            ["2025-07-30T00:00:00+10:00", 100.33552],
                            ["2025-07-31T00:00:00+10:00", 116.79885],
                            ["2025-08-01T00:00:00+10:00", 151.75597],
                            ["2025-08-02T00:00:00+10:00", 99.923819],
                            ["2025-08-03T00:00:00+10:00", 62.70816],
                            ["2025-08-04T00:00:00+10:00", 82.383437],
                        ],
                    },
                    {
                        "name": "price_SA1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "SA1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 221.44729],
                            ["2025-07-30T00:00:00+10:00", 174.19802],
                            ["2025-07-31T00:00:00+10:00", 174.96885],
                            ["2025-08-01T00:00:00+10:00", 169.45569],
                            ["2025-08-02T00:00:00+10:00", 157.55174],
                            ["2025-08-03T00:00:00+10:00", 40.779722],
                            ["2025-08-04T00:00:00+10:00", -0.094375],
                        ],
                    },
                    {
                        "name": "price_TAS1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "TAS1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 133.00122],
                            ["2025-07-30T00:00:00+10:00", 135.4233],
                            ["2025-07-31T00:00:00+10:00", 124.39788],
                            ["2025-08-01T00:00:00+10:00", 141.54302],
                            ["2025-08-02T00:00:00+10:00", 123.75938],
                            ["2025-08-03T00:00:00+10:00", 97.693889],
                            ["2025-08-04T00:00:00+10:00", 89.773403],
                        ],
                    },
                    {
                        "name": "price_VIC1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "VIC1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 197.1866],
                            ["2025-07-30T00:00:00+10:00", 122.69663],
                            ["2025-07-31T00:00:00+10:00", 111.56892],
                            ["2025-08-01T00:00:00+10:00", 152.78285],
                            ["2025-08-02T00:00:00+10:00", 108.57524],
                            ["2025-08-03T00:00:00+10:00", 45.012639],
                            ["2025-08-04T00:00:00+10:00", 18.915451],
                        ],
                    },
                ],
                "network_timezone_offset": "+10:00",
            },
            {
                "network_code": "NEM",
                "metric": "demand",
                "unit": "MW",
                "interval": "1d",
                "date_start": "2025-07-29T00:00:00+10:00",
                "date_end": "2025-08-04T00:00:00+10:00",
                "groupings": [],
                "results": [
                    {
                        "name": "demand_NSW1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "NSW1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 2569802.8],
                            ["2025-07-30T00:00:00+10:00", 2844766.6],
                            ["2025-07-31T00:00:00+10:00", 2722465.0],
                            ["2025-08-01T00:00:00+10:00", 2785864.9],
                            ["2025-08-02T00:00:00+10:00", 2648681.1],
                            ["2025-08-03T00:00:00+10:00", 2383426.8],
                            ["2025-08-04T00:00:00+10:00", 2446638.2],
                        ],
                    },
                    {
                        "name": "demand_QLD1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "QLD1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 1832302.9],
                            ["2025-07-30T00:00:00+10:00", 1808055.1],
                            ["2025-07-31T00:00:00+10:00", 1805833.2],
                            ["2025-08-01T00:00:00+10:00", 1932236.9],
                            ["2025-08-02T00:00:00+10:00", 1707748.3],
                            ["2025-08-03T00:00:00+10:00", 1700550.3],
                            ["2025-08-04T00:00:00+10:00", 1702549.6],
                        ],
                    },
                    {
                        "name": "demand_SA1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "SA1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 523177.3],
                            ["2025-07-30T00:00:00+10:00", 472879.4],
                            ["2025-07-31T00:00:00+10:00", 477207.48],
                            ["2025-08-01T00:00:00+10:00", 475785.94],
                            ["2025-08-02T00:00:00+10:00", 432926.49],
                            ["2025-08-03T00:00:00+10:00", 401433.35],
                            ["2025-08-04T00:00:00+10:00", 533840.3],
                        ],
                    },
                    {
                        "name": "demand_TAS1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "TAS1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 319244.59],
                            ["2025-07-30T00:00:00+10:00", 342140.19],
                            ["2025-07-31T00:00:00+10:00", 358229.05],
                            ["2025-08-01T00:00:00+10:00", 349270.21],
                            ["2025-08-02T00:00:00+10:00", 332999.28],
                            ["2025-08-03T00:00:00+10:00", 324106.83],
                            ["2025-08-04T00:00:00+10:00", 326445.66],
                        ],
                    },
                    {
                        "name": "demand_VIC1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "VIC1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 1814990.9],
                            ["2025-07-30T00:00:00+10:00", 1762580.6],
                            ["2025-07-31T00:00:00+10:00", 1764286.1],
                            ["2025-08-01T00:00:00+10:00", 1802829.4],
                            ["2025-08-02T00:00:00+10:00", 1569819.4],
                            ["2025-08-03T00:00:00+10:00", 1452537.1],
                            ["2025-08-04T00:00:00+10:00", 1577344.5],
                        ],
                    },
                ],
                "network_timezone_offset": "+10:00",
            },
            {
                "network_code": "NEM",
                "metric": "demand_energy",
                "unit": "MWh",
                "interval": "1d",
                "date_start": "2025-07-29T00:00:00+10:00",
                "date_end": "2025-08-04T00:00:00+10:00",
                "groupings": [],
                "results": [
                    {
                        "name": "demand_energy_NSW1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "NSW1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 214.1436],
                            ["2025-07-30T00:00:00+10:00", 237.0518],
                            ["2025-07-31T00:00:00+10:00", 226.8668],
                            ["2025-08-01T00:00:00+10:00", 232.1629],
                            ["2025-08-02T00:00:00+10:00", 220.7399],
                            ["2025-08-03T00:00:00+10:00", 198.6382],
                            ["2025-08-04T00:00:00+10:00", 203.8769],
                        ],
                    },
                    {
                        "name": "demand_energy_QLD1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "QLD1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 152.6928],
                            ["2025-07-30T00:00:00+10:00", 150.6713],
                            ["2025-07-31T00:00:00+10:00", 150.4854],
                            ["2025-08-01T00:00:00+10:00", 161.0182],
                            ["2025-08-02T00:00:00+10:00", 142.3094],
                            ["2025-08-03T00:00:00+10:00", 141.729],
                            ["2025-08-04T00:00:00+10:00", 141.88],
                        ],
                    },
                    {
                        "name": "demand_energy_SA1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "SA1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 43.5989],
                            ["2025-07-30T00:00:00+10:00", 39.4028],
                            ["2025-07-31T00:00:00+10:00", 39.7701],
                            ["2025-08-01T00:00:00+10:00", 39.6449],
                            ["2025-08-02T00:00:00+10:00", 36.0792],
                            ["2025-08-03T00:00:00+10:00", 33.454],
                            ["2025-08-04T00:00:00+10:00", 44.487],
                        ],
                    },
                    {
                        "name": "demand_energy_TAS1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "TAS1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 26.6025],
                            ["2025-07-30T00:00:00+10:00", 28.5114],
                            ["2025-07-31T00:00:00+10:00", 29.8515],
                            ["2025-08-01T00:00:00+10:00", 29.1059],
                            ["2025-08-02T00:00:00+10:00", 27.7517],
                            ["2025-08-03T00:00:00+10:00", 27.0102],
                            ["2025-08-04T00:00:00+10:00", 27.2047],
                        ],
                    },
                    {
                        "name": "demand_energy_VIC1",
                        "date_start": "2025-07-29T00:00:00+10:00",
                        "date_end": "2025-08-04T00:00:00+10:00",
                        "columns": {"region": "VIC1"},
                        "data": [
                            ["2025-07-29T00:00:00+10:00", 151.2486],
                            ["2025-07-30T00:00:00+10:00", 146.8844],
                            ["2025-07-31T00:00:00+10:00", 147.0138],
                            ["2025-08-01T00:00:00+10:00", 150.2425],
                            ["2025-08-02T00:00:00+10:00", 130.8149],
                            ["2025-08-03T00:00:00+10:00", 121.0728],
                            ["2025-08-04T00:00:00+10:00", 131.4554],
                        ],
                    },
                ],
                "network_timezone_offset": "+10:00",
            },
        ],
        "total_records": None,
    }


def test_to_records_to_pandas(market_metric_response):
    """Test that to_records properly parses the market_metric_response fixture."""
    
    # Parse the response into a TimeSeriesResponse object
    response = TimeSeriesResponse.model_validate(market_metric_response)
    
    # Convert to records
    records = response.to_records()
    


def test_to_records_parses_market_metric_response(market_metric_response):
    """Test that to_records properly parses the market_metric_response fixture."""
    
    # Parse the response into a TimeSeriesResponse object
    response = TimeSeriesResponse.model_validate(market_metric_response)
    
    # Convert to records
    records = response.to_records()
    
    # Basic validation
    assert isinstance(records, list)
    assert len(records) > 0
    
    # Expected number of records: 3 metrics × 5 regions × 7 days = 105 records
    # But due to the way to_records works (combining metrics for same timestamp/region),
    # we should get 5 regions × 7 days = 35 records
    expected_records = 35
    assert len(records) == expected_records
    
    # Check first record structure
    first_record = records[0]
    assert isinstance(first_record, dict)
    
    # Required fields that should be present
    required_fields = {"interval", "network_region"}
    assert all(field in first_record for field in required_fields)
    
    # Check that all three metrics are present in the first record
    metric_fields = {"price", "demand", "demand_energy"}
    assert all(field in first_record for field in metric_fields)
    
    # Validate interval field
    assert isinstance(first_record["interval"], datetime)
    assert first_record["interval"].tzinfo is not None  # Should have timezone info
    
    # Validate network_region field
    assert isinstance(first_record["network_region"], str)
    assert first_record["network_region"] in ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
    
    # Validate metric values are numeric
    for metric in metric_fields:
        assert isinstance(first_record[metric], (int, float))
    
    # Check that we have records for all expected regions
    regions_found = set(record["network_region"] for record in records)
    expected_regions = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}
    assert regions_found == expected_regions
    
    # Check that we have records for all expected dates (7 days)
    dates_found = set(record["interval"].date() for record in records)
    assert len(dates_found) == 7
    
    # Verify specific data points
    # Find NSW1 record for 2025-07-29
    nsw1_record = next(
        (r for r in records if r["network_region"] == "NSW1" and r["interval"].date().isoformat() == "2025-07-29"),
        None
    )
    assert nsw1_record is not None
    assert abs(nsw1_record["price"] - 162.13802) < 0.001
    assert abs(nsw1_record["demand"] - 2569802.8) < 0.001
    assert abs(nsw1_record["demand_energy"] - 214.1436) < 0.001
    
    # Verify timezone handling
    # All intervals should be in +10:00 timezone
    for record in records:
        interval = record["interval"]
        assert interval.tzinfo is not None
        # Check that it's in the correct timezone (+10:00)
        offset = interval.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 10 * 3600  # +10:00 in seconds
    
    # Verify data consistency
    # Each region should have exactly 7 records (one per day)
    for region in expected_regions:
        region_records = [r for r in records if r["network_region"] == region]
        assert len(region_records) == 7
        
        # All records for this region should have the same region value
        assert all(r["network_region"] == region for r in region_records)
        
        # All records should have all three metrics
        assert all(all(metric in r for metric in metric_fields) for r in region_records)


def test_market_metric_response_to_pandas_dataframe(market_metric_response):
    """Test that market_metric_response converts to pandas DataFrame with proper columns and region matching."""
    
    # Parse the response into a TimeSeriesResponse object
    response = TimeSeriesResponse.model_validate(market_metric_response)
    
    # Convert to pandas DataFrame
    df = response.to_pandas()
    
    # Basic DataFrame validation
    assert df is not None
    assert len(df) == 35  # 5 regions × 7 days
    
    # Check that DataFrame has the expected columns
    expected_columns = {"interval", "network_region", "price", "demand", "demand_energy"}
    actual_columns = set(df.columns)
    assert actual_columns == expected_columns, f"Expected columns {expected_columns}, got {actual_columns}"
    
    # Validate data types
    assert "datetime" in str(df["interval"].dtype)  # datetime objects
    assert df["network_region"].dtype == "object"  # string
    assert df["price"].dtype in ["float64", "float32"]
    assert df["demand"].dtype in ["float64", "float32"]
    assert df["demand_energy"].dtype in ["float64", "float32"]
    
    # Check that all regions are present
    regions_in_df = set(df["network_region"].unique())
    expected_regions = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}
    assert regions_in_df == expected_regions, f"Expected regions {expected_regions}, got {regions_in_df}"
    
    # Check that each region has exactly 7 rows (one per day)
    for region in expected_regions:
        region_count = len(df[df["network_region"] == region])
        assert region_count == 7, f"Region {region} has {region_count} rows, expected 7"
    
    # Check that all dates are present (7 unique dates)
    unique_dates = df["interval"].dt.date.unique()
    assert len(unique_dates) == 7, f"Expected 7 unique dates, got {len(unique_dates)}"
    
    # Verify specific data points match the original JSON
    # NSW1 on 2025-07-29
    nsw1_row = df[
        (df["network_region"] == "NSW1") & 
        (df["interval"].dt.date == datetime(2025, 7, 29).date())
    ]
    assert len(nsw1_row) == 1, "Should have exactly one row for NSW1 on 2025-07-29"
    
    nsw1_data = nsw1_row.iloc[0]
    assert abs(nsw1_data["price"] - 162.13802) < 0.001
    assert abs(nsw1_data["demand"] - 2569802.8) < 0.001
    assert abs(nsw1_data["demand_energy"] - 214.1436) < 0.001
    
    # QLD1 on 2025-07-30
    qld1_row = df[
        (df["network_region"] == "QLD1") & 
        (df["interval"].dt.date == datetime(2025, 7, 30).date())
    ]
    assert len(qld1_row) == 1, "Should have exactly one row for QLD1 on 2025-07-30"
    
    qld1_data = qld1_row.iloc[0]
    assert abs(qld1_data["price"] - 100.33552) < 0.001
    assert abs(qld1_data["demand"] - 1808055.1) < 0.001
    assert abs(qld1_data["demand_energy"] - 150.6713) < 0.001
    
    # Check that there are no missing values in metric columns
    assert not df["price"].isna().any(), "Price column should not have missing values"
    assert not df["demand"].isna().any(), "Demand column should not have missing values"
    assert not df["demand_energy"].isna().any(), "Demand_energy column should not have missing values"
    
    # Verify timezone information is preserved
    # All intervals should have timezone info
    assert all(interval.tzinfo is not None for interval in df["interval"])
    
    # Check that all intervals are in +10:00 timezone
    for interval in df["interval"]:
        offset = interval.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 10 * 3600  # +10:00 in seconds
    
    # Test DataFrame operations
    # Group by region and verify counts
    region_counts = df.groupby("network_region").size()
    for region in expected_regions:
        assert region_counts[region] == 7, f"Region {region} should have 7 rows"
    
    # Test filtering by region
    nsw1_df = df[df["network_region"] == "NSW1"]
    assert len(nsw1_df) == 7
    assert all(region == "NSW1" for region in nsw1_df["network_region"])
    
    # Test sorting
    sorted_df = df.sort_values(["network_region", "interval"])
    assert len(sorted_df) == len(df)
    
    # Verify the DataFrame can be used for analysis
    # Check summary statistics
    assert df["price"].mean() > 0
    assert df["demand"].mean() > 0
    assert df["demand_energy"].mean() > 0
    
    # Check that each metric has reasonable value ranges
    assert df["price"].min() >= -1  # Allow for negative prices (like SA1 on 2025-08-04)
    assert df["price"].max() < 1000
    assert df["demand"].min() > 0
    assert df["demand"].max() < 10000000
    assert df["demand_energy"].min() > 0
    assert df["demand_energy"].max() < 1000


def demonstrate_market_metric_dataframe(market_metric_response):
    """Demonstrate what the pandas DataFrame looks like when created from market_metric_response."""
    
    # Parse the response into a TimeSeriesResponse object
    response = TimeSeriesResponse.model_validate(market_metric_response)
    
    # Convert to pandas DataFrame
    df = response.to_pandas()
    
    print("=== Market Metric DataFrame Demonstration ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nUnique regions: {sorted(df['network_region'].unique())}")
    print(f"Date range: {df['interval'].min()} to {df['interval'].max()}")
    
    print("\n=== Sample Data (first 10 rows) ===")
    print(df.head(10).to_string(index=False))
    
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    print("\n=== Data by Region ===")
    for region in sorted(df['network_region'].unique()):
        region_df = df[df['network_region'] == region]
        print(f"{region}: {len(region_df)} rows, price range: ${region_df['price'].min():.2f} - ${region_df['price'].max():.2f}")
    
    print("\n=== Verification: NSW1 on 2025-07-29 ===")
    nsw1_row = df[
        (df["network_region"] == "NSW1") & 
        (df["interval"].dt.date == datetime(2025, 7, 29).date())
    ]
    if len(nsw1_row) == 1:
        data = nsw1_row.iloc[0]
        print(f"Price: ${data['price']:.2f}/MWh")
        print(f"Demand: {data['demand']:.1f} MW")
        print(f"Demand Energy: {data['demand_energy']:.1f} MWh")
    else:
        print("❌ Expected exactly one row for NSW1 on 2025-07-29")
    
    return df


@pytest.mark.asyncio
async def test_market_metric_combinations():
    """Test different combinations of market metrics to identify which ones work."""

    # Get API key from environment
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        print("❌ OPENELECTRICITY_API_KEY environment variable not set")
        print("Please create a .env file with your API key:")
        print("OPENELECTRICITY_API_KEY=your_api_key_here")
        return

    client = AsyncOEClient(api_key=api_key)

    # Test date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    # Individual market metrics to test
    all_market_metrics = [
            MarketMetric.PRICE,
            MarketMetric.DEMAND,
            MarketMetric.DEMAND_ENERGY,
            MarketMetric.CURTAILMENT_SOLAR_UTILITY,
            MarketMetric.CURTAILMENT_WIND,
        ]

    print("🔍 Testing individual market metrics...")

    # Test each market metric individually
    for metric in all_market_metrics:
        try:
            print(f"  Testing {metric.value}...", end=" ")
            response = await client.get_market(
                network_code="NEM",
                metrics=[metric],
                interval="5m",
                date_start=start_date,
                date_end=end_date,
            )
            print("✅ SUCCESS")
        except Exception as e:
            print(f"❌ FAILED: {e}")

    print("\n🔍 Testing market metric combinations...")

    # Test combinations of 2 market metrics
    for i, metric1 in enumerate(all_market_metrics):
        for metric2 in all_market_metrics[i + 1 :]:
            try:
                print(f"  Testing {metric1.value} + {metric2.value}...", end=" ")
                response = await client.get_market(
                    network_code="NEM",
                    metrics=[metric1, metric2],
                    interval="5m",
                    date_start=start_date,
                    date_end=end_date,
                )
                print("✅ SUCCESS")
            except Exception as e:
                print(f"❌ FAILED: {e}")

    print("\n🔍 Testing all market metrics together...")

    # Test all market metrics together
    try:
        print("  Testing all market metrics...", end=" ")
        response = await client.get_market(
            network_code="NEM",
            metrics=all_market_metrics,
            interval="5m",
            date_start=start_date,
            date_end=end_date,
        )
        print("✅ SUCCESS")
    except Exception as e:
        print(f"❌ FAILED: {e}")

        # Try removing one metric at a time
        print("\n🔍 Testing all market metrics minus one at a time...")
        for i, metric in enumerate(all_market_metrics):
            test_metrics = all_market_metrics[:i] + all_market_metrics[i + 1 :]
            try:
                print(f"  Testing without {metric.value}...", end=" ")
                response = await client.get_market(
                    network_code="NEM",
                    metrics=test_metrics,
                    interval="5m",
                    date_start=start_date,
                    date_end=end_date,
                )
                print("✅ SUCCESS - This metric was the problem!")
                print(f"  ❌ Problematic metric: {metric.value}")
                break
            except Exception as e:
                print(f"❌ Still fails: {e}")

    print("\n🔍 Testing different intervals...")

    # Test different intervals with working metrics
    intervals = ["5m", "1h", "1d"]
    working_metrics = [MarketMetric.PRICE]  # Start with a metric that likely works

    for interval in intervals:
        try:
            print(f"  Testing {interval} interval...", end=" ")
            response = await client.get_market(
                network_code="NEM",
                metrics=working_metrics,
                interval=interval,
                date_start=start_date,
                date_end=end_date,
            )
            print("✅ SUCCESS")
        except Exception as e:
            print(f"❌ FAILED: {e}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_market_metric_combinations())
