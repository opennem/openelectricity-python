#!/usr/bin/env python
"""
Test script to specifically test facilities PySpark conversion.
"""

import os
from dotenv import load_dotenv
from openelectricity import OEClient

# Load environment variables
load_dotenv()


def test_facilities_pyspark():
    """Test facilities PySpark conversion."""
    print("🧪 Testing Facilities PySpark Conversion")
    print("=" * 50)

    # Check if PySpark is available
    try:
        import pyspark

        print(f"✅ PySpark {pyspark.__version__} is available")
    except ImportError:
        print("❌ PySpark not available. Install with: uv add pyspark")
        return

    # Initialize the client
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        print("❌ OPENELECTRICITY_API_KEY environment variable not set")
        return

    client = OEClient(api_key=api_key)

    print("\n🏭 Fetching facilities data...")
    try:
        # Get a smaller subset to test
        response = client.get_facilities(network_region="NSW1")
        print(f"✅ Fetched {len(response.data)} facilities")

        # Test pandas conversion first (should work)
        print("\n📊 Testing pandas conversion...")
        pandas_df = response.to_pandas()
        print(f"✅ Pandas DataFrame created: {pandas_df.shape}")
        print(f"   Columns: {', '.join(pandas_df.columns)}")

        # Test PySpark conversion
        print("\n⚡ Testing PySpark conversion...")
        spark_df = response.to_pyspark()

        if spark_df is not None:
            print("✅ PySpark DataFrame created successfully!")
            print(f"   Schema: {spark_df.schema}")
            print(f"   Row count: {spark_df.count()}")
            print(f"   Columns: {', '.join(spark_df.columns)}")

            # Show sample data
            print("\n📋 Sample PySpark data:")
            spark_df.show(5, truncate=False)

            # Test some operations
            print("\n🔍 Testing PySpark operations:")

            # Count by fuel technology
            fueltech_counts = spark_df.groupBy("fueltech_id").count()
            print("⛽ Fuel Technology Counts:")
            fueltech_counts.show()

            print("🎉 All tests passed!")

        else:
            print("❌ PySpark DataFrame creation returned None")
            print("   Check the logs above for error details")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_facilities_pyspark()
