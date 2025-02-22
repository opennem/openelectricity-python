"""
Example script to get the latest coal generation data from the OpenElectricity API.

This script retrieves all coal facilities (both black and brown coal) in the
National Electricity Market (NEM) and displays their most recent generation interval.

Required Environment Variables:
    OPENELECTRICITY_API_KEY: Your OpenElectricity API key
    OPENELECTRICITY_API_URL: (Optional) Override the default API URL

To run this example:
1. Set your API key in the environment:
   export OPENELECTRICITY_API_KEY=your_api_key_here
2. Run the script:
   python examples/coal_latest.py
"""

import sys

from rich.console import Console
from rich.table import Table

from openelectricity import OEClient
from openelectricity.settings_schema import settings
from openelectricity.types import DataMetric, UnitFueltechType


def format_power(value: float) -> str:
    """Format power values in MW or GW."""
    if abs(value) >= 1000:
        return f"{value / 1000:.2f} GW"
    return f"{value:.2f} MW"


def main():
    """Run the example."""
    console = Console()

    # Print settings for debugging
    console.print("\n[bold blue]API Settings:[/bold blue]")
    console.print(f"API URL: {settings.base_url}")
    console.print(f"Environment: {settings.env}")
    console.print(f"API Key: {settings.api_key[:8]}...")

    # Create table for output
    table = Table(title="Coal Generation in NEM")
    table.add_column("Facility", style="cyan")
    table.add_column("Unit", style="magenta")
    table.add_column("Type", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Generation", justify="right", style="yellow")
    table.add_column("Time", style="dim")

    try:
        # Get data from API
        with OEClient() as client:
            console.print("\n[bold blue]Making API requests...[/bold blue]")

            # Get all coal facilities
            console.print("Fetching coal facilities...")
            facilities = client.get_facilities(
                network_id=["NEM"],
                fueltech_id=[UnitFueltechType.COAL_BLACK, UnitFueltechType.COAL_BROWN],
            )

            if not facilities.data:
                console.print("[yellow]No coal facilities found[/yellow]")
                return

            # Get facility codes
            facility_codes = [facility.code for facility in facilities.data]
            console.print(f"Found {len(facility_codes)} facilities")

            # Get latest generation data
            console.print("Fetching generation data...")
            response = client.get_facility_data(
                network_code="NEM",
                facility_code=facility_codes,
                metrics=[DataMetric.POWER],
                interval="5m",  # 5-minute intervals
            )

            # Process each facility
            for facility in facilities.data:
                # Find matching time series
                for series in response.data:
                    for result in series.results:
                        if result.columns.unit_code in [unit.code for unit in facility.units]:
                            # Get latest data point
                            if result.data:
                                latest = result.data[-1]
                                unit = next(u for u in facility.units if u.code == result.columns.unit_code)
                                status = "Operating" if latest.value > 0 else "Offline"
                                table.add_row(
                                    facility.name,
                                    unit.code,
                                    unit.fueltech_id.value.replace("_", " ").title(),
                                    status,
                                    format_power(latest.value),
                                    latest.timestamp.strftime("%Y-%m-%d %H:%M"),
                                )

        # Print table
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
