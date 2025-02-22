"""
Example of using OpenElectricity API data with Polars.

This example demonstrates how to:
1. Get network data from the API
2. Convert it to a Polars DataFrame
3. Perform basic analysis
"""

from datetime import datetime, timedelta

import polars as pl

from openelectricity import OEClient
from openelectricity.types import DataMetric, UnitFueltechType, UnitStatusType


def main():
    """Run the example."""
    # Calculate date range for last week
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=7)

    # Get data from API
    with OEClient() as client:
        # Get facilities
        facilities = client.get_facilities(
            network_id=["NEM"],
            status_id=[UnitStatusType.OPERATING],
            fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
        )

        # Get network data
        response = client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.POWER, DataMetric.ENERGY],
            interval="1d",
            date_start=start_date,
            date_end=end_date,
            secondary_grouping="fueltech_group",
        )

    # Convert to Polars DataFrame
    df = response.to_polars()
    units = response.get_metric_units()

    # Print basic information
    print("\nDataFrame Info:")
    print(df.describe())

    # Group by fuel tech and calculate total energy
    energy_by_fueltech = (
        df.group_by("fueltech_group")
        .agg(
            pl.col("energy").sum().alias("total_energy_mwh"),
            pl.col("power").mean().alias("avg_power_mw"),
        )
        .sort("total_energy_mwh", descending=True)
    )

    print("\nEnergy by Fuel Technology:")
    print(energy_by_fueltech)

    # Calculate daily totals
    daily_totals = (
        df.group_by("interval")
        .agg(
            pl.col("energy").sum().alias("total_energy_mwh"),
            pl.col("power").sum().alias("total_power_mw"),
        )
        .sort("interval")
    )

    print("\nDaily Totals:")
    print(daily_totals)

    # Calculate percentage contribution of each fuel tech
    total_energy = df["energy"].sum()
    energy_percentage = (
        df.group_by("fueltech_group")
        .agg(pl.col("energy").sum().alias("total_energy"))
        .with_columns((pl.col("total_energy") / total_energy * 100).alias("percentage"))
        .sort("percentage", descending=True)
    )

    print("\nPercentage Contribution by Fuel Technology:")
    print(energy_percentage)


if __name__ == "__main__":
    main()
