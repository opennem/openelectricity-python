"""
Basic example of using the OpenElectricity API client.

This example demonstrates how to:
1. Get network data for the NEM (National Electricity Market)
2. Request daily data for a one month period
3. Use both synchronous and asynchronous clients

Required Environment Variables:
    OPENELECTRICITY_API_KEY: Your OpenElectricity API key
    OPENELECTRICITY_API_URL: (Optional) Override the default API URL
"""

import asyncio
import sys
from datetime import datetime, timedelta

from rich.console import Console

from openelectricity import AsyncOEClient, OEClient
from openelectricity.settings_schema import settings
from openelectricity.types import DataMetric, UnitFueltechType, UnitStatusType


def sync_example(console: Console):
    """Example using the synchronous client."""
    # Calculate date range for last month
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=7)

    try:
        # Initialize client
        with OEClient() as client:
            console.print("[blue]Running synchronous example...[/blue]")

            # Get facilities
            console.print("Fetching facilities...")
            facilities = client.get_facilities(
                network_id=["NEM"],
                status_id=[UnitStatusType.OPERATING],
                fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
            )
            console.print(f"Found {len(facilities.data)} facilities")

            # Get network data for NEM
            console.print("Fetching network data...")
            response = client.get_network_data(
                network_code="NEM",
                metrics=[DataMetric.POWER, DataMetric.ENERGY],
                interval="1d",  # Daily intervals
                date_start=start_date,
                date_end=end_date,
                secondary_grouping="fueltech_group",  # Group by fuel technology
            )

            # Print results
            console.print("\n[green]Synchronous Results:[/green]")
            console.print(f"Data points: {len(response.data)}")

            # Print each time series
            for series in response.data:
                console.print(f"\n[blue]Metric: {series.metric}[/blue]")
                console.print(f"Unit: {series.unit}")
                console.print(f"Interval: {series.interval}")
                start, end = series.date_range
                console.print(f"Start: {start}")
                console.print(f"End: {end}")
                console.print("Results:")

                # Print each result group
                for result in series.results:
                    console.print(f"\n  {result.name}:")
                    console.print(f"  Fuel Tech Group: {result.columns.fueltech_group}")
                    console.print("  Data Points:")
                    for point in result.data:
                        console.print(f"    {point.timestamp}: {point.value:.2f} {series.unit}")

    except Exception as e:
        console.print(f"[red]Error in synchronous example: {str(e)}[/red]")
        return False

    return True


async def async_example(console: Console):
    """Example using the asynchronous client."""
    # Calculate date range for last week
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=7)

    try:
        async with AsyncOEClient() as client:
            console.print("\n[blue]Running asynchronous example...[/blue]")

            # Get facilities
            console.print("Fetching facilities...")
            facilities = await client.get_facilities(
                network_id=["NEM"],
                status_id=[UnitStatusType.OPERATING],
                fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
            )
            console.print(f"Found {len(facilities.data)} facilities")

            # Get network data
            console.print("Fetching network data...")
            response = await client.get_network_data(
                network_code="NEM",
                metrics=[DataMetric.POWER],
                interval="1d",
                date_start=start_date,
                date_end=end_date,
                secondary_grouping="fueltech_group",
            )

            # Print results
            console.print("\n[green]Asynchronous Results:[/green]")
            console.print(f"Data points: {len(response.data)}")

            # Print each time series
            for series in response.data:
                console.print(f"\n[blue]Metric: {series.metric}[/blue]")
                console.print(f"Unit: {series.unit}")
                console.print(f"Interval: {series.interval}")
                start, end = series.date_range
                console.print(f"Start: {start}")
                console.print(f"End: {end}")
                console.print("Results:")

                # Print each result group
                for result in series.results:
                    console.print(f"\n  {result.name}:")
                    console.print(f"  Fuel Tech Group: {result.columns.fueltech_group}")
                    console.print("  Data Points:")
                    for point in result.data:
                        console.print(f"    {point.timestamp}: {point.value:.2f} {series.unit}")

    except Exception as e:
        console.print(f"[red]Error in asynchronous example: {str(e)}[/red]")
        return False

    return True


def main():
    """Run both examples."""
    console = Console()

    # Print settings for debugging
    console.print("\n[bold blue]API Settings:[/bold blue]")
    console.print(f"API URL: {settings.base_url}")
    console.print(f"Environment: {settings.env}")
    console.print(f"API Key: {settings.api_key[:8]}...")

    # Run examples
    sync_success = sync_example(console)
    async_success = asyncio.run(async_example(console))

    if not sync_success or not async_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
