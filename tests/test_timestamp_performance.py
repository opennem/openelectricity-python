#!/usr/bin/env python
"""
Test script to compare different methods of converting Python datetime to PySpark timestamp.
"""

import time
from datetime import datetime, timezone, timedelta
import random
import pytest


@pytest.fixture
def records():
    """Fixture to provide test records."""
    return generate_test_data(1000)

def generate_test_data(num_records=10000):
    """Generate test datetime data."""
    base_time = datetime.now(timezone.utc)
    records = []

    for i in range(num_records):
        # Random time within last 30 days
        random_hours = random.randint(-720, 0)  # -30 days to now
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)

        timestamp = base_time + timedelta(hours=random_hours, minutes=random_minutes, seconds=random_seconds)

        records.append({"id": i, "timestamp": timestamp, "value": random.random() * 1000})

    return records


def test_method_1_string_conversion(records):
    """Method 1: Convert to ISO string (current approach)."""
    start_time = time.time()

    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            if hasattr(value, "isoformat"):  # Datetime objects
                cleaned_record[key] = str(value)
            else:
                cleaned_record[key] = value
        cleaned_records.append(cleaned_record)

    elapsed = time.time() - start_time
    return cleaned_records, elapsed


def test_method_2_timestamp_conversion(records):
    """Method 2: Convert to Unix timestamp (seconds since epoch)."""
    start_time = time.time()

    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            if hasattr(value, "timestamp"):  # Datetime objects
                cleaned_record[key] = value.timestamp()  # Unix timestamp
            else:
                cleaned_record[key] = value
        cleaned_records.append(cleaned_record)

    elapsed = time.time() - start_time
    return cleaned_records, elapsed


def test_method_3_microseconds_conversion(records):
    """Method 3: Convert to microseconds since epoch (PySpark TimestampType)."""
    start_time = time.time()

    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            if hasattr(value, "timestamp"):  # Datetime objects
                # Convert to microseconds for PySpark TimestampType
                cleaned_record[key] = int(value.timestamp() * 1_000_000)
            else:
                cleaned_record[key] = value
        cleaned_records.append(cleaned_record)

    elapsed = time.time() - start_time
    return cleaned_records, elapsed


def test_method_4_direct_datetime(records):
    """Method 4: Keep as datetime object (let PySpark handle conversion)."""
    start_time = time.time()

    # No conversion needed - pass datetime objects directly
    elapsed = time.time() - start_time
    return records, elapsed


def benchmark_methods():
    """Benchmark all conversion methods."""
    print("🚀 Benchmarking DateTime to PySpark Timestamp Conversion Methods")
    print("=" * 70)

    # Generate test data
    print("📊 Generating test data...")
    test_records = generate_test_data(10000)
    print(f"✅ Generated {len(test_records)} test records")

    # Test each method
    methods = [
        ("String Conversion (ISO)", test_method_1_string_conversion),
        ("Unix Timestamp (seconds)", test_method_2_timestamp_conversion),
        ("Microseconds Timestamp", test_method_3_microseconds_conversion),
        ("Direct DateTime", test_method_4_direct_datetime),
    ]

    results = []

    for method_name, method_func in methods:
        print(f"\n🔍 Testing: {method_name}")

        # Run multiple times for more accurate timing
        times = []
        for _ in range(5):
            _, elapsed = method_func(test_records)
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        results.append({"method": method_name, "avg_time": avg_time, "min_time": min_time, "max_time": max_time})

        print(f"   Average: {avg_time:.4f}s")
        print(f"   Range:   {min_time:.4f}s - {max_time:.4f}s")

    # Sort by performance
    results.sort(key=lambda x: x["avg_time"])

    print(f"\n🏆 Performance Ranking:")
    print("=" * 50)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['method']:<25} {result['avg_time']:.4f}s")

    # Calculate speedup
    fastest = results[0]["avg_time"]
    print(f"\n📈 Speedup vs Fastest Method:")
    print("=" * 50)
    for result in results:
        if result["avg_time"] > 0:
            speedup = fastest / result["avg_time"]
            print(f"{result['method']:<25} {speedup:.2f}x")
        else:
            print(f"{result['method']:<25} ∞ (baseline)")

    return results


def test_pyspark_compatibility():
    """Test which methods work with PySpark."""
    print(f"\n🧪 Testing PySpark Compatibility")
    print("=" * 50)

    try:
        import pyspark
        from pyspark.sql import SparkSession
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, LongType

        # Create Spark session
        spark = SparkSession.builder.appName("TimestampTest").getOrCreate()
        print(f"✅ PySpark {pyspark.__version__} available")

        # Test data
        test_data = [
            {"id": 1, "timestamp": datetime.now(timezone.utc), "value": 100.0},
            {"id": 2, "timestamp": datetime.now(timezone.utc), "value": 200.0},
        ]

        # Test each conversion method
        methods = [
            ("String (ISO)", lambda x: str(x) if hasattr(x, "isoformat") else x),
            ("Unix Timestamp", lambda x: x.timestamp() if hasattr(x, "timestamp") else x),
            ("Microseconds", lambda x: int(x.timestamp() * 1_000_000) if hasattr(x, "timestamp") else x),
            ("Direct DateTime", lambda x: x),  # No conversion
        ]

        for method_name, converter in methods:
            print(f"\n🔍 Testing: {method_name}")

            try:
                # Convert data
                converted_data = []
                for record in test_data:
                    converted_record = {}
                    for key, value in record.items():
                        converted_record[key] = converter(value)
                    converted_data.append(converted_record)

                # Try to create DataFrame
                if method_name == "Direct DateTime":
                    # For direct datetime, we need explicit schema
                    schema = StructType(
                        [
                            StructField("id", LongType(), True),
                            StructField("timestamp", TimestampType(), True),
                            StructField("value", DoubleType(), True),
                        ]
                    )
                    df = spark.createDataFrame(converted_data, schema=schema)
                else:
                    # For other methods, let PySpark infer schema
                    df = spark.createDataFrame(converted_data)

                print(f"   ✅ Success: DataFrame created with {df.count()} rows")
                print(f"   Schema: {df.schema}")

                # Show sample data
                df.show(2, truncate=False)

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        spark.stop()

    except ImportError:
        print("❌ PySpark not available for compatibility testing")


if __name__ == "__main__":
    # Run benchmarks
    results = benchmark_methods()

    # Test PySpark compatibility
    test_pyspark_compatibility()

    print(f"\n💡 Recommendations:")
    print("=" * 50)
    print("1. For pure Python performance: Use Unix timestamps or microseconds")
    print("2. For PySpark compatibility: Use microseconds (TimestampType) or direct datetime with schema")
    print("3. For human readability: Use ISO strings (current approach)")
    print("4. For best of both worlds: Use microseconds with explicit PySpark schema")
