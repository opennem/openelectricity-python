"""
Example of using OpenElectricity API data with Polars.

This example demonstrates how to:
1. Get network data from the API
2. Convert it to a Polars DataFrame
3. Perform basic analysis

Required Environment Variables:
    OPENELECTRICITY_API_KEY: Your OpenElectricity API key
    OPENELECTRICITY_API_URL: (Optional) Override the default API URL

Required Dependencies:
    polars: Data analysis library (install with: uv add "openelectricity[analysis]")
"""

import sys
from datetime import datetime, timedelta

from rich.console import Console

try:
    import polars as pl
except ImportError:
    print("Error: Polars is required for this example.")
    print("Install it with: uv add 'openelectricity[analysis]'")
    sys.exit(1)

from openelectricity import OEClient
from openelectricity.settings_schema import settings
from openelectricity.types import DataMetric, UnitFueltechType, UnitStatusType


def main():
    """Run the example."""
    console = Console()

    # Print settings for debugging
    console.print("\n[bold blue]API Settings:[/bold blue]")
    console.print(f"API URL: {settings.base_url}")
    console.print(f"Environment: {settings.env}")
    console.print(f"API Key: {settings.api_key[:8]}...")

    try:
        # Calculate date range for last week
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=7)

        # Get data from API
        with OEClient() as client:
            console.print("\n[blue]Fetching data from API...[/blue]")

            # Get facilities
            console.print("Fetching facilities...")
            facilities = client.get_facilities(
                network_id=["NEM"],
                status_id=[UnitStatusType.OPERATING],
                fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
            )
            console.print(f"Found {len(facilities.data)} facilities")

            # Get network data
            console.print("Fetching network data...")
            response = client.get_network_data(
                network_code="NEM",
                metrics=[DataMetric.POWER, DataMetric.ENERGY],
                interval="1d",
                date_start=start_date,
                date_end=end_date,
                secondary_grouping="fueltech_group",
            )

        # Convert to Polars DataFrame
        console.print("\n[blue]Converting to Polars DataFrame...[/blue]")
        df = response.to_polars()
        units = response.get_metric_units()

        # Print basic information
        console.print("\n[green]DataFrame Info:[/green]")
        console.print(df.describe())

        # Group by fuel tech and calculate total energy
        console.print("\n[blue]Calculating energy by fuel technology...[/blue]")
        energy_by_fueltech = (
            df.group_by("fueltech_group")
            .agg(
                pl.col("energy").sum().alias("total_energy_mwh"),
                pl.col("power").mean().alias("avg_power_mw"),
            )
            .sort("total_energy_mwh", descending=True)
        )

        console.print("\n[green]Energy by Fuel Technology:[/green]")
        console.print(energy_by_fueltech)

        # Calculate daily totals
        console.print("\n[blue]Calculating daily totals...[/blue]")
        daily_totals = (
            df.group_by("interval")
            .agg(
                pl.col("energy").sum().alias("total_energy_mwh"),
                pl.col("power").sum().alias("total_power_mw"),
            )
            .sort("interval")
        )

        console.print("\n[green]Daily Totals:[/green]")
        console.print(daily_totals)

        # Calculate percentage contribution of each fuel tech
        console.print("\n[blue]Calculating percentage contributions...[/blue]")
        total_energy = df["energy"].sum()
        energy_percentage = (
            df.group_by("fueltech_group")
            .agg(pl.col("energy").sum().alias("total_energy"))
            .with_columns((pl.col("total_energy") / total_energy * 100).alias("percentage"))
            .sort("percentage", descending=True)
        )

        console.print("\n[green]Percentage Contribution by Fuel Technology:[/green]")
        console.print(energy_percentage)

    except ImportError as e:
        console.print(f"[red]Error: Required package not found - {str(e)}[/red]")
        console.print("Install analysis dependencies with: uv add 'openelectricity[analysis]'")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
