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
from openelectricity.types import DataMetric


def sync_example():
    """Example using the synchronous client."""
    # Calculate date range for last month
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=7)

    # Initialize client
    with OEClient() as client:
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
        for point in response.data:
            print(f"\nInterval: {point.interval}")
            print(f"Start: {point.start}")
            print(f"End: {point.end}")
            print("Data:")
            for key, value in point.data.items():
                print(f"  {key}: {value}")


async def async_example():
    """Example using the asynchronous client."""
    # Calculate date range for last month
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=7)

    # Initialize client
    async with AsyncOEClient() as client:
        # Get network data for NEM
        response = await client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.POWER, DataMetric.ENERGY],
            interval="1d",  # Daily intervals
            date_start=start_date,
            date_end=end_date,
            secondary_grouping="fueltech_group",  # Group by fuel technology
        )

        # Print results
        print("\nAsynchronous Results:")
        print(f"Data points: {len(response.data)}")
        for point in response.data:
            print(f"\nInterval: {point.interval}")
            print(f"Start: {point.start}")
            print(f"End: {point.end}")
            print("Data:")
            for key, value in point.data.items():
                print(f"  {key}: {value}")


def main():
    """Run both sync and async examples."""
    # Run synchronous example
    sync_example()

    # Run async example
    asyncio.run(async_example())


if __name__ == "__main__":
    main()
