#!/usr/bin/env python3
"""
Test script to identify which metric is causing the 400 Bad Request error.
"""

import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import pytest

from openelectricity import AsyncOEClient
from openelectricity.settings_schema import settings
from openelectricity.types import DataMetric, UnitStatusType, UnitFueltechType, MarketMetric

# Load environment variables from .env file
load_dotenv()


@pytest.mark.asyncio
async def test_metric_combinations():
    """Test different combinations of metrics to identify the problematic one."""

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

    # Individual metrics to test
    all_metrics = [
        DataMetric.POWER,
        DataMetric.ENERGY,
        DataMetric.MARKET_VALUE,
        DataMetric.EMISSIONS,
        DataMetric.RENEWABLE_PROPORTION,
    ]

    print("🔍 Testing individual metrics...")

    # Test each metric individually
    for metric in all_metrics:
        try:
            print(f"  Testing {metric.value}...", end=" ")
            response = await client.get_network_data(
                network_code="NEM",
                metrics=[metric],
                interval="5m",
                date_start=start_date,
                date_end=end_date,
            )
            print("✅ SUCCESS")
        except Exception as e:
            print(f"❌ FAILED: {e}")

    print("\n🔍 Testing metric combinations...")

    # Test combinations of 2 metrics
    for i, metric1 in enumerate(all_metrics):
        for metric2 in all_metrics[i + 1 :]:
            try:
                print(f"  Testing {metric1.value} + {metric2.value}...", end=" ")
                response = await client.get_network_data(
                    network_code="NEM",
                    metrics=[metric1, metric2],
                    interval="5m",
                    date_start=start_date,
                    date_end=end_date,
                )
                print("✅ SUCCESS")
            except Exception as e:
                print(f"❌ FAILED: {e}")

    print("\n🔍 Testing your original combination...")

    # Test the original combination from your code
    original_metrics = [
        DataMetric.POWER,
        DataMetric.ENERGY,
        DataMetric.MARKET_VALUE,
        DataMetric.EMISSIONS,
        DataMetric.RENEWABLE_PROPORTION,
    ]

    try:
        print("  Testing original combination...", end=" ")
        response = await client.get_network_data(
            network_code="NEM",
            metrics=original_metrics,
            interval="5m",
            date_start=start_date,
            date_end=end_date,
        )
        print("✅ SUCCESS")
    except Exception as e:
        print(f"❌ FAILED: {e}")

        # Try removing one metric at a time
        print("\n🔍 Testing original combination minus one metric at a time...")
        for i, metric in enumerate(original_metrics):
            test_metrics = original_metrics[:i] + original_metrics[i + 1 :]
            try:
                print(f"  Testing without {metric.value}...", end=" ")
                response = await client.get_network_data(
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

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_metric_combinations())
