#!/usr/bin/env python
"""
Debug script to understand the facilities data structure.
"""

import os
import json
from dotenv import load_dotenv
from openelectricity import OEClient

# Load environment variables
load_dotenv()


def debug_facilities_data():
    """Debug facilities data structure."""
    print("🔍 Debugging Facilities Data Structure")
    print("=" * 50)

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

        # Look at the first facility in detail
        if response.data:
            facility = response.data[0]
            print(f"\n📋 First facility: {facility.code}")
            print(f"   Type: {type(facility)}")
            print(f"   Name: {facility.name}")
            print(f"   Units count: {len(facility.units)}")

            # Convert to dict and examine
            facility_dict = facility.model_dump()
            print(f"\n🔍 Facility dict keys: {list(facility_dict.keys())}")

            # Look at each field type
            for key, value in facility_dict.items():
                print(f"   {key}: {type(value)} = {str(value)[:100]}...")

            # Look at units in detail
            if facility.units:
                unit = facility.units[0]
                print(f"\n🔧 First unit: {unit.code}")
                print(f"   Type: {type(unit)}")

                unit_dict = unit.model_dump()
                print(f"   Unit dict keys: {list(unit_dict.keys())}")

                for key, value in unit_dict.items():
                    print(f"   {key}: {type(value)} = {str(value)[:100]}...")

            # Try to create a simple record manually
            print(f"\n🧪 Testing simple record creation...")

            # Create a minimal record
            simple_record = {
                "code": facility.code,
                "name": facility.name,
                "network_id": str(facility.network_id),
                "network_region": facility.network_region,
            }

            print(f"Simple record: {simple_record}")

            # Try PySpark with just this simple record
            try:
                import pyspark
                from pyspark.sql import SparkSession

                spark = SparkSession.builder.appName("Debug").getOrCreate()

                print("🔥 Testing PySpark with simple record...")
                simple_df = spark.createDataFrame([simple_record])
                print("✅ Simple PySpark DataFrame created!")
                simple_df.show()

                # Now try with a few more fields
                print("\n🔥 Testing with more fields...")
                extended_record = simple_record.copy()
                if facility.units:
                    unit = facility.units[0]
                    extended_record.update(
                        {
                            "unit_code": unit.code,
                            "fueltech_id": str(unit.fueltech_id),
                            "status_id": str(unit.status_id),
                            "capacity_registered": unit.capacity_registered,
                        }
                    )

                print(f"Extended record: {extended_record}")
                extended_df = spark.createDataFrame([extended_record])
                print("✅ Extended PySpark DataFrame created!")
                extended_df.show()

            except Exception as spark_error:
                print(f"❌ PySpark error: {spark_error}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_facilities_data()
