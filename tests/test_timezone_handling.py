#!/usr/bin/env python
"""
Test script to examine timezone handling in PySpark conversion.
"""

import os
from dotenv import load_dotenv
from openelectricity import OEClient

# Load environment variables
load_dotenv()


def test_timezone_handling():
    """Test how timezones are handled in PySpark conversion."""
    print("🕐 Testing Timezone Handling in PySpark Conversion")
    print("=" * 60)

    # Initialize the client
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        print("❌ OPENELECTRICITY_API_KEY environment variable not set")
        return

    client = OEClient(api_key=api_key)

    print("\n📊 Fetching market data...")
    try:
        # Get market data
        from openelectricity.types import MarketMetric
        from datetime import datetime, timedelta

        # Get just a few hours of data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=5)

        response = client.get_market(
            network_code="NEM", metrics=[MarketMetric.PRICE], interval="1h", date_start=start_time, date_end=end_time
        )
        print(f"✅ Fetched {len(response.data)} time series")

        # Check the raw records first
        print("\n🔍 Examining raw records...")
        records = response.to_records()
        if records:
            first_record = records[0]
            print(f"First record: {first_record}")
            print(f"Interval type: {type(first_record['interval'])}")
            print(f"Interval value: {first_record['interval']}")
            print(f"Interval repr: {repr(first_record['interval'])}")

            # Check if it has timezone info
            interval_value = first_record["interval"]
            if hasattr(interval_value, "tzinfo"):
                print(f"Timezone info: {interval_value.tzinfo}")
                print(f"UTC offset: {interval_value.utcoffset()}")
                print(f"Timezone name: {interval_value.tzname()}")
                print(f"ISO format: {interval_value.isoformat()}")

        # Check pandas conversion
        print("\n📊 Testing pandas conversion...")
        pandas_df = response.to_pandas()
        if not pandas_df.empty:
            print(f"Pandas DataFrame shape: {pandas_df.shape}")
            print(f"Pandas interval dtype: {pandas_df['interval'].dtype}")
            print(f"First pandas interval: {pandas_df['interval'].iloc[0]}")
            print(f"First pandas interval type: {type(pandas_df['interval'].iloc[0])}")

        # Check PySpark conversion
        print("\n⚡ Testing PySpark conversion...")
        try:
            import pyspark

            print(f"PySpark version: {pyspark.__version__}")

            pyspark_df = response.to_pyspark()
            if pyspark_df is not None:
                print("✅ PySpark DataFrame created successfully!")
                print(f"Schema: {pyspark_df.schema}")

                # Show the data
                print("\n📋 PySpark DataFrame content:")
                pyspark_df.show(5, truncate=False)

                # Check the actual string values
                print("\n🔍 Examining PySpark string values:")
                interval_values = pyspark_df.select("interval").collect()
                for i, row in enumerate(interval_values[:3]):
                    print(f"Row {i}: {row['interval']}")

                # Test if we can parse these back to datetime with timezone
                print("\n🔄 Testing timezone parsing from PySpark strings...")
                from datetime import datetime
                import re

                sample_interval = interval_values[0]["interval"]
                print(f"Sample interval string: {sample_interval}")

                # Check if it contains timezone info
                if "+" in sample_interval or sample_interval.endswith("Z"):
                    print("✅ Timezone information is preserved in the string!")

                    # Try to parse it back
                    try:
                        # Parse ISO format with timezone
                        parsed_dt = datetime.fromisoformat(sample_interval)
                        print(f"✅ Successfully parsed back to datetime: {parsed_dt}")
                        print(f"   Timezone: {parsed_dt.tzinfo}")
                        print(f"   UTC offset: {parsed_dt.utcoffset()}")
                    except Exception as parse_error:
                        print(f"❌ Could not parse back to datetime: {parse_error}")
                else:
                    print("⚠️  No timezone information found in the string")

            else:
                print("❌ PySpark DataFrame creation failed")

        except ImportError:
            print("❌ PySpark not available")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_timezone_handling()
