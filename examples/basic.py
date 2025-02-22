"""
Basic example of using the OpenElectricity API client.

This example demonstrates how to:
1. Get network data for the NEM (National Electricity Market)
2. Request daily data for a one month period
3. Use both synchronous and asynchronous clients
"""

import asyncio
from datetime import datetime, timedelta

from openelectricity import AsyncOEClient, OEClient
from openelectricity.types import DataMetric, UnitFueltechType, UnitStatusType


def sync_example():
    """Example using the synchronous client."""
    # Calculate date range for last month
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=7)

    # Initialize client
    with OEClient() as client:
        # Get facilities
        facilities = client.get_facilities(
            network_id=["NEM"],
            status_id=[UnitStatusType.OPERATING],
            fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
        )
        print("\nFacilities:")
        for facility in facilities.data:
            print(f"  {facility.network_code}: {facility.metric}")

        # Get network data for NEM
        response = client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.POWER, DataMetric.ENERGY],
            interval="1d",  # Daily intervals
            date_start=start_date,
            date_end=end_date,
            secondary_grouping="fueltech_group",  # Group by fuel technology
        )

        # Print results
        print("\nSynchronous Results:")
        print(f"Data points: {len(response.data)}")

        # Print each time series
        for series in response.data:
            print(f"\nMetric: {series.metric}")
            print(f"Unit: {series.unit}")
            print(f"Interval: {series.interval}")
            print(f"Start: {series.start}")
            print(f"End: {series.end}")
            print("Results:")

            # Print each result group
            for result in series.results:
                print(f"\n  {result.name}:")
                print(f"  Fuel Tech Group: {result.columns.fueltech_group}")
                print("  Data Points:")
                for point in result.data:
                    print(f"    {point.timestamp}: {point.value:.2f} {series.unit}")


async def async_example():
    """Example using the asynchronous client."""
    # Calculate date range for last week
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=7)

    async with AsyncOEClient() as client:
        # Get facilities
        facilities = await client.get_facilities(
            network_id=["NEM"],
            status_id=[UnitStatusType.OPERATING],
            fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
        )
        print("\nFacilities:")
        for facility in facilities.data:
            print(f"  {facility.network_code}: {facility.metric}")

        # Get network data
        response = await client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.POWER],
            interval="1d",
            date_start=start_date,
            date_end=end_date,
            secondary_grouping="fueltech_group",
        )

        # Print results
        print("\nAsynchronous Results:")
        print(f"Data points: {len(response.data)}")

        # Print each time series
        for series in response.data:
            print(f"\nMetric: {series.metric}")
            print(f"Unit: {series.unit}")
            print(f"Interval: {series.interval}")
            print(f"Start: {series.start}")
            print(f"End: {series.end}")
            print("Results:")

            # Print each result group
            for result in series.results:
                print(f"\n  {result.name}:")
                print(f"  Fuel Tech Group: {result.columns.fueltech_group}")
                print("  Data Points:")
                for point in result.data:
                    print(f"    {point.timestamp}: {point.value:.2f} {series.unit}")


def main():
    """Run both examples."""
    sync_example()
    asyncio.run(async_example())


if __name__ == "__main__":
    main()
