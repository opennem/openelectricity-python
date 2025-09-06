#!/usr/bin/env python3
"""
Performance test for the optimized to_records function.

This script benchmarks the to_records function and verifies correctness.
"""

import time
import random
from datetime import datetime, timedelta
from typing import Any

# Import the models
from openelectricity.models.timeseries import (
    TimeSeriesResponse,
    NetworkTimeSeries,
    TimeSeriesResult,
    TimeSeriesColumns,
    TimeSeriesDataPoint,
)
from openelectricity.types import Network, DataInterval


def create_mock_data(size: int = 1000) -> TimeSeriesResponse:
    """Create mock time series data for testing."""

    # Create base timestamp
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    # Create mock results
    results = []
    for i in range(size // 100):  # Create fewer result groups
        # Create data points for this result
        data_points = []
        for j in range(100):  # 100 data points per result
            timestamp = base_time + timedelta(hours=i * 24 + j)
            value = random.uniform(100, 1000)
            data_points.append(TimeSeriesDataPoint(root=(timestamp, value)))

        # Create result with varying network regions
        regions = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
        region = regions[i % len(regions)]

        result = TimeSeriesResult(
            name=f"power_{region}|battery_charging",
            date_start=base_time + timedelta(hours=i * 24),
            date_end=base_time + timedelta(hours=(i + 1) * 24 - 1),
            columns=TimeSeriesColumns(
                unit_code="MW",
                fueltech_group="battery",
                network_region=region if i % 2 == 0 else None,  # Test the regex fallback
            ),
            data=data_points,
        )
        results.append(result)

    # Create network time series
    network_series = NetworkTimeSeries(
        network_code=Network.NEM,
        metric="energy",
        unit="MWh",
        interval="1h",
        groupings=["network_region", "fueltech_group"],
        results=results,
        network_timezone_offset="+10:00",
    )

    # Create response
    response = TimeSeriesResponse(
        version="1.0", created_at=datetime.now(), success=True, error=None, total_records=size, data=[network_series]
    )

    return response


def benchmark_to_records(response: TimeSeriesResponse, iterations: int = 5) -> dict[str, float]:
    """Benchmark the to_records function."""

    print(
        f"Benchmarking to_records with {len(response.data[0].results)} result groups and {len(response.data[0].results[0].data)} data points per group..."
    )

    # Warm up
    _ = response.to_records()

    # Benchmark
    times = []
    for i in range(iterations):
        start_time = time.perf_counter()
        records = response.to_records()
        end_time = time.perf_counter()

        duration = end_time - start_time
        times.append(duration)

        print(f"  Iteration {i + 1}: {duration:.4f}s ({len(records)} records)")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "total_records": len(response.to_records()),
        "iterations": iterations,
    }


def verify_correctness(response: TimeSeriesResponse) -> bool:
    """Verify that the to_records function produces correct results."""

    print("Verifying correctness...")

    records = response.to_records()

    # Check that we have records
    if not records:
        print("❌ No records returned")
        return False

    # Check that each record has required fields
    required_fields = {"interval", "energy"}
    for i, record in enumerate(records[:5]):  # Check first 5 records
        missing_fields = required_fields - set(record.keys())
        if missing_fields:
            print(f"❌ Record {i} missing fields: {missing_fields}")
            return False

    # Check that network_region is present (either from columns or extracted from name)
    regions_found = set()
    for record in records:
        if "network_region" in record and record["network_region"]:
            regions_found.add(record["network_region"])

    expected_regions = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}
    if not regions_found.issuperset(expected_regions):
        print(f"❌ Missing expected regions. Found: {regions_found}, Expected: {expected_regions}")
        return False

    # Check that intervals are datetime objects
    for record in records[:5]:
        if not isinstance(record["interval"], datetime):
            print(f"❌ Interval is not datetime: {type(record['interval'])}")
            return False

    # Check that energy values are numeric
    for record in records[:5]:
        if not isinstance(record["energy"], (int, float)):
            print(f"❌ Energy value is not numeric: {type(record['energy'])}")
            return False

    print("✅ All correctness checks passed!")
    return True


def test_different_sizes():
    """Test performance with different data sizes."""

    sizes = [100, 1000, 5000, 10000]

    print("=" * 60)
    print("PERFORMANCE TESTING WITH DIFFERENT DATA SIZES")
    print("=" * 60)

    for size in sizes:
        print(f"\nTesting with {size} total data points...")

        # Create test data
        response = create_mock_data(size)

        # Benchmark
        results = benchmark_to_records(response, iterations=3)

        print(f"Results:")
        print(f"  Average time: {results['avg_time']:.4f}s")
        print(f"  Min time: {results['min_time']:.4f}s")
        print(f"  Max time: {results['max_time']:.4f}s")
        print(f"  Total records generated: {results['total_records']}")

        # Verify correctness
        verify_correctness(response)


def test_edge_cases():
    """Test edge cases and error conditions."""

    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    # Test empty response
    print("\n1. Testing empty response...")
    empty_response = TimeSeriesResponse(
        version="1.0", created_at=datetime.now(), success=True, error=None, total_records=0, data=[]
    )

    records = empty_response.to_records()
    if records == []:
        print("✅ Empty response handled correctly")
    else:
        print(f"❌ Empty response returned {len(records)} records")

    # Test response with no data points
    print("\n2. Testing response with no data points...")
    no_data_response = create_mock_data(0)
    records = no_data_response.to_records()
    if records == []:
        print("✅ No data points handled correctly")
    else:
        print(f"❌ No data points returned {len(records)} records")

    # Test response with missing network_region in both columns and name
    print("\n3. Testing missing network_region...")
    response = create_mock_data(100)
    # Modify one result to have no region info
    response.data[0].results[0].columns.network_region = None
    response.data[0].results[0].name = "power_unknown|battery_charging"

    records = response.to_records()
    records_without_region = [r for r in records if "network_region" not in r or r["network_region"] is None]

    if records_without_region:
        print(f"⚠️  Found {len(records_without_region)} records without network_region (expected for fallback case)")
    else:
        print("✅ All records have network_region")


def main():
    """Main test function."""

    print("🚀 Testing optimized to_records function")
    print("=" * 60)

    # Test with medium size first
    print("\nInitial test with 1000 data points...")
    response = create_mock_data(1000)

    # Verify correctness
    if not verify_correctness(response):
        print("❌ Correctness check failed!")
        return

    # Benchmark
    results = benchmark_to_records(response)
    print(f"\nPerformance Summary:")
    print(f"  Average time: {results['avg_time']:.4f}s")
    print(f"  Records per second: {results['total_records'] / results['avg_time']:.0f}")

    # Test different sizes
    test_different_sizes()

    # Test edge cases
    test_edge_cases()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")


if __name__ == "__main__":
    main()
