#!/usr/bin/env python3
"""
Test script to verify that the network_region fix in to_pandas() works properly.

This script tests the OpenElectricity SDK to ensure that network_region information
is properly extracted from the result name when it's None in TimeSeriesColumns.
"""

import os
from datetime import datetime, timedelta
from openelectricity import OEClient
from openelectricity.types import DataMetric


def test_network_region_extraction():
    """Test that network_region is properly extracted in to_pandas()."""

    print("🧪 Testing network_region extraction fix...")
    print("=" * 60)

    # Set up the client
    api_key = "REDACTED_API_KEY"

    try:
        import logging
        import pandas as pd

        from datetime import datetime, timedelta
        from openelectricity import OEClient
        from openelectricity.settings_schema import settings
        from openelectricity.types import DataMetric, UnitStatusType, UnitFueltechType, MarketMetric

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        console = logging.getLogger("OpenElectricityLogger")

        # Calculate date range for last day
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(2))

        # Get data from API
        with OEClient(api_key=api_key) as client:
            console.info("Fetching data from API...")

            # Get network data
            console.info("Fetching network data...")
            response = client.get_network_data(
                network_code="NEM",
                metrics=[
                    DataMetric.POWER,
                    DataMetric.ENERGY,
                    DataMetric.PRICE,
                    DataMetric.MARKET_VALUE,
                    DataMetric.DEMAND,
                    DataMetric.EMISSIONS,
                    DataMetric.RENEWABLE_PROPORTION,
                ],
                interval="5m",
                date_start=start_date,
                date_end=end_date,
                primary_grouping="network_region",
                secondary_grouping="fueltech_group",
            )

        print(f"✅ API call successful! Got {len(response.data)} time series")

        # Show some raw response structure for debugging
        if response.data:
            first_series = response.data[0]
            print(f"\n📊 Sample data structure:")
            print(f"   Network: {first_series.network_code}")
            print(f"   Metric: {first_series.metric}")
            print(f"   Groupings: {first_series.groupings}")

            if first_series.results:
                first_result = first_series.results[0]
                print(f"   Sample result name: {first_result.name}")
                print(f"   Sample columns: {first_result.columns}")

        # Convert to pandas and check for network_region
        print(f"\n🐼 Converting to pandas DataFrame...")
        df = response.to_pandas()

        print(f"✅ DataFrame created with shape: {df.shape}")
        print(f"📋 DataFrame columns: {list(df.columns)}")

        # Check if network_region is present
        if "network_region" in df.columns:
            print(f"✅ SUCCESS: network_region column is present!")

            # Show unique regions
            unique_regions = df["network_region"].dropna().unique()
            print(f"🗺️  Unique regions found: {list(unique_regions)}")

            # Show sample data
            print(f"\n📝 Sample DataFrame rows:")
            print(df[["interval", "network_region", "fueltech_group", "power"]].head(10))

        else:
            print(f"❌ FAILED: network_region column is missing!")
            print(f"   Available columns: {list(df.columns)}")

        return "network_region" in df.columns

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_raw_response_structure():
    """Test the raw response structure to understand the data format."""

    print("\n🔍 Testing raw response structure...")
    print("=" * 60)

    api_key = "REDACTED_API_KEY"

    try:
        client = OEClient(api_key=api_key)

        # Get a small amount of data
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(hours=2)

        response = client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.POWER],
            interval="1h",
            date_start=start_date,
            date_end=end_date,
            primary_grouping="network_region",
            secondary_grouping="fueltech_group",
        )

        # Examine the structure
        print(f"Response success: {response.success}")
        print(f"Number of time series: {len(response.data)}")

        for i, series in enumerate(response.data[:3]):  # Show first 3 series
            print(f"\n📊 Series {i + 1}:")
            print(f"   Metric: {series.metric}")
            print(f"   Network: {series.network_code}")
            print(f"   Groupings: {series.groupings}")
            print(f"   Number of results: {len(series.results)}")

            for j, result in enumerate(series.results[:2]):  # Show first 2 results
                print(f"   📋 Result {j + 1}:")
                print(f"      Name: {result.name}")
                print(f"      Columns: {result.columns}")
                print(f"      Data points: {len(result.data)}")

                if result.data:
                    print(f"      Sample data point: {result.data[0]}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 OpenElectricity Network Region Fix Test")
    print("=" * 60)

    # Test raw structure first
    test_raw_response_structure()

    # Test the fix
    success = test_network_region_extraction()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The network_region fix is working.")
    else:
        print("💥 Tests failed. The fix may need more work.")
    print("=" * 60)
