#!/usr/bin/env python3
import asyncio
import pytest
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from openelectricity import AsyncOEClient, OEClient
from openelectricity.types import DataMetric

load_dotenv()

api_key = os.getenv("OPENELECTRICITY_API_KEY")
client = OEClient(api_key=api_key)
response = client.get_facilities()
print(response)


@pytest.mark.asyncio
async def test_facility_metrics():
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        print("❌ OPENELECTRICITY_API_KEY environment variable not set")
        print("Please create a .env file with your API key:")
        print("OPENELECTRICITY_API_KEY=your_api_key_here")
        return

    client = AsyncOEClient(api_key=api_key)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=2)
    facility_code = "YALLOURN"

    print(f"🔍 Testing facility metrics for {facility_code}...")
    print(f"Date range: {start_date} to {end_date}")
    print()

        # Individual facility metrics to test
    all_facility_metrics = [
        DataMetric.POWER,
        DataMetric.ENERGY,
        DataMetric.MARKET_VALUE,
        DataMetric.EMISSIONS,
        DataMetric.RENEWABLE_PROPORTION,
    ]

    print("🔍 Testing individual facility metrics...")
    working_metrics = []
    failing_metrics = []

    for metric in all_facility_metrics:
        print(f"  Testing {metric.value}...", end=" ")
        try:
            response = await client.get_facility_data(
                network_code="NEM",
                facility_code=facility_code,
                metrics=[metric],
                interval="5m",
                date_start=start_date,
                date_end=end_date,
            )
            print("✅ SUCCESS")
            working_metrics.append(metric)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failing_metrics.append(metric)

    print()
    print("🔍 Testing facility metric combinations...")

    # Test combinations of working metrics
    if len(working_metrics) >= 2:
        for i, metric1 in enumerate(working_metrics):
            for metric2 in working_metrics[i + 1 :]:
                print(f"  Testing {metric1.value} + {metric2.value}...", end=" ")
                try:
                    response = await client.get_facility_data(
                        network_code="NEM",
                        facility_code=facility_code,
                        metrics=[metric1, metric2],
                        interval="5m",
                        date_start=start_date,
                        date_end=end_date,
                    )
                    print("✅ SUCCESS")
                except Exception as e:
                    print(f"❌ FAILED: {e}")

    print()
    print("🔍 Testing all working metrics together...")
    if working_metrics:
        print(f"  Testing all working metrics: {[m.value for m in working_metrics]}...", end=" ")
        try:
            response = await client.get_facility_data(
                network_code="NEM",
                facility_code=facility_code,
                metrics=working_metrics,
                interval="5m",
                date_start=start_date,
                date_end=end_date,
            )
            print("✅ SUCCESS")
        except Exception as e:
            print(f"❌ FAILED: {e}")

    print()
    print("🔍 Testing different intervals...")
    if working_metrics:
        test_metric = working_metrics[0]
        intervals = ["5m", "1h", "1d"]
        for interval in intervals:
            print(f"  Testing {test_metric.value} with {interval} interval...", end=" ")
            try:
                response = await client.get_facility_data(
                    network_code="NEM",
                    facility_code=facility_code,
                    metrics=[test_metric],
                    interval=interval,
                    date_start=start_date,
                    date_end=end_date,
                )
                print("✅ SUCCESS")
            except Exception as e:
                print(f"❌ FAILED: {e}")

    print()
    print("📊 SUMMARY:")
    print(f"✅ Working metrics ({len(working_metrics)}): {[m.value for m in working_metrics]}")
    print(f"❌ Failing metrics ({len(failing_metrics)}): {[m.value for m in failing_metrics]}")

    print()
    print("🔧 ALLOWED COMBINATIONS:")
    print("=" * 50)

    # Individual metrics
    print("📋 Individual Metrics:")
    for metric in working_metrics:
        print(f"  ✅ {metric.value}")

    # Two-metric combinations
    if len(working_metrics) >= 2:
        print()
        print("📋 Two-Metric Combinations:")
        for i, metric1 in enumerate(working_metrics):
            for metric2 in working_metrics[i + 1 :]:
                print(f"  ✅ [{metric1.value}, {metric2.value}]")

    # Three-metric combinations
    if len(working_metrics) >= 3:
        print()
        print("📋 Three-Metric Combinations:")
        for i, metric1 in enumerate(working_metrics):
            for j, metric2 in enumerate(working_metrics[i + 1 :], i + 1):
                for metric3 in working_metrics[j + 1 :]:
                    print(f"  ✅ [{metric1.value}, {metric2.value}, {metric3.value}]")

    # All metrics together
    if working_metrics:
        print()
        print("📋 All Metrics Together:")
        all_metrics_str = ", ".join([m.value for m in working_metrics])
        print(f"  ✅ [{all_metrics_str}]")

    print()
    print("💡 USAGE EXAMPLES:")
    print("=" * 50)
    if working_metrics:
        print("For your Databricks code, you can use:")
        print()
        print("from openelectricity import OEClient")
        print("from openelectricity.types import DataMetric")
        print("from datetime import datetime, timedelta")
        print()
        print("with OEClient(api_key=dbutils.secrets.get('daveok', 'opennem')) as client:")
        print("    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)")
        print("    start_date = end_date - timedelta(days=2)")
        print("    facility_data = client.get_facility_data(")
        print("        network_code='NEM',")
        print("        facility_code='YALLOURN',")
        print("        metrics=[")
        for metric in working_metrics:
            print(f"            DataMetric.{metric.name},")
        print("        ],")
        print("        interval='5m',")
        print("        date_start=start_date,")
        print("        date_end=end_date,")
        print("    )")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_facility_metrics())
