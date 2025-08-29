#!/usr/bin/env python
"""
Example: Curtailment Analysis with Line Chart

This example demonstrates how to:
1. Fetch curtailment data for solar and wind
2. Process data for all NEM regions
3. Create a line chart showing total curtailment per region for each interval in MW
"""

import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from openelectricity import OEClient
from openelectricity.types import MarketMetric

# Load environment variables
load_dotenv()

# Set up the seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def fetch_curtailment_data(client: OEClient, days_back: int = 5) -> pd.DataFrame:
    """
    Fetch curtailment data for all NEM regions in a single API call.

    Args:
        client: OpenElectricity API client
        days_back: Number of days to fetch data for

    Returns:
        DataFrame with curtailment data
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Format dates for API
    date_start = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    date_end = end_date.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"Fetching curtailment data from {date_start} to {date_end}")
    print("  Fetching data for all NEM regions...")

    all_data = []

    try:
        # Fetch curtailment data for ALL regions in one call
        response = client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.CURTAILMENT_SOLAR_UTILITY, MarketMetric.CURTAILMENT_WIND],
            interval="5m",
            date_start=pd.to_datetime(date_start),
            date_end=pd.to_datetime(date_end),
            primary_grouping="network_region",
        )

        # Process each metric
        for timeseries in response.data:
            metric = timeseries.metric

            for result in timeseries.results:
                # Extract region from result name (format: metric_REGION)
                # e.g., "curtailment_solar_utility_NSW1" -> "NSW1"
                name_parts = result.name.split("_")
                region = name_parts[-1] if len(name_parts) > 1 else "Unknown"

                # Extract data points
                for data_point in result.data:
                    # data_point is a TimeSeriesDataPoint object
                    timestamp = data_point.timestamp if hasattr(data_point, "timestamp") else data_point.root[0]
                    value = data_point.value if hasattr(data_point, "value") else data_point.root[1]
                    if value is not None:
                        all_data.append({"region": region, "metric": metric, "date": pd.to_datetime(timestamp), "value": value})

    except Exception as e:
        print(f"    Error fetching data: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Sort by date
    if not df.empty:
        df = df.sort_values("date")
        print(f"  Retrieved data for regions: {', '.join(df['region'].unique())}")

    return df


def create_curtailment_line_chart(df: pd.DataFrame):
    """
    Create line charts showing total curtailment per region for each interval in MW.

    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        print("No data to plot")
        return

    # Calculate total curtailment (solar + wind) per region per interval
    total_curtailment = df.groupby(["date", "region"])["value"].sum().reset_index()

    # Create pivot table for line chart
    pivot_df = total_curtailment.pivot(index="date", columns="region", values="value")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for each region
    region_colors = {
        "NSW1": "#FF6B00",  # Orange
        "QLD1": "#0066CC",  # Blue
        "VIC1": "#228B22",  # Green
        "SA1": "#DC143C",  # Red
        "TAS1": "#8B008B",  # Purple
    }

    # Plot line for each region
    for region in pivot_df.columns:
        color = region_colors.get(region, "#888888")
        pivot_df[region].plot(ax=ax, label=region, color=color, linewidth=2, marker="o", markersize=4, alpha=0.8)

    # Format the chart
    ax.set_title("Total Curtailment per Region by Interval", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date/Time", fontsize=12)
    ax.set_ylabel("Total Curtailment (MW)", fontsize=12)
    ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45)

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    # Save the figure
    output_file = "curtailment_line_chart.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nLine chart saved to {output_file}")

    plt.show()


def create_combined_chart(df: pd.DataFrame):
    """
    Create a combined stacked bar chart showing total curtailment (solar + wind) by region.

    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        print("No data to plot")
        return

    # Convert MW to MWh
    df["value_mwh"] = df["value"] * 24  # Convert daily average MW to MWh per day

    # Create pivot table for combined solar and wind
    combined_pivot = df.pivot_table(values="value_mwh", index="date", columns=["region", "metric"], aggfunc="sum", fill_value=0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for stacked bar chart
    # Reorganize to have solar and wind as separate layers for each region
    dates = combined_pivot.index
    regions = df["region"].unique()

    # Define colors for each region (solar=lighter, wind=darker)
    region_colors = {
        "NSW1": ("#FFB366", "#FF6B00"),  # Orange
        "QLD1": ("#66B3FF", "#0066CC"),  # Blue
        "VIC1": ("#90EE90", "#228B22"),  # Green
        "SA1": ("#FFB6C1", "#DC143C"),  # Pink/Red
        "TAS1": ("#DDA0DD", "#8B008B"),  # Purple
    }

    # Create stacked bars
    bar_width = 0.8
    x_pos = range(len(dates))

    # Track bottom position for stacking
    bottom_solar = [0] * len(dates)
    bottom_wind = [0] * len(dates)

    # Plot each region's data
    for region in regions:
        # Solar curtailment
        if (region, "curtailment_solar_utility") in combined_pivot.columns:
            solar_values = combined_pivot[(region, "curtailment_solar_utility")].values
            ax.bar(
                x_pos,
                solar_values,
                bar_width,
                bottom=bottom_solar,
                label=f"{region} Solar",
                color=region_colors.get(region, ("#888888", "#444444"))[0],
                alpha=0.8,
            )
            bottom_solar = [b + v for b, v in zip(bottom_solar, solar_values, strict=False)]

        # Wind curtailment
        if (region, "curtailment_wind") in combined_pivot.columns:
            wind_values = combined_pivot[(region, "curtailment_wind")].values
            ax.bar(
                x_pos,
                wind_values,
                bar_width,
                bottom=bottom_wind,
                label=f"{region} Wind",
                color=region_colors.get(region, ("#888888", "#444444"))[1],
                alpha=0.8,
                hatch="//",
            )  # Add pattern to distinguish wind from solar
            bottom_wind = [b + v for b, v in zip(bottom_wind, wind_values, strict=False)]

    # Format the chart
    ax.set_title("Total Renewable Curtailment by Region and Type (MWh)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total Curtailment (MWh)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.strftime("%b %d") for d in dates], rotation=45, ha="right")
    ax.legend(title="Region & Type", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    # Save the figure
    output_file = "total_curtailment_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nCombined chart saved to {output_file}")

    plt.show()


def print_summary_statistics(df: pd.DataFrame):
    """
    Print summary statistics for curtailment data.

    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        print("No data to summarize")
        return

    print("\n" + "=" * 60)
    print("CURTAILMENT SUMMARY STATISTICS (MW)")
    print("=" * 60)

    # Overall statistics
    solar_data = df[df["metric"] == "curtailment_solar_utility"]
    wind_data = df[df["metric"] == "curtailment_wind"]

    if not solar_data.empty:
        print("\nSolar Curtailment:")
        print(f"  Total: {solar_data['value'].sum():,.1f} MW")
        print(f"  Average: {solar_data['value'].mean():,.1f} MW")
        print(f"  Maximum: {solar_data['value'].max():,.1f} MW")

    if not wind_data.empty:
        print("\nWind Curtailment:")
        print(f"  Total: {wind_data['value'].sum():,.1f} MW")
        print(f"  Average: {wind_data['value'].mean():,.1f} MW")
        print(f"  Maximum: {wind_data['value'].max():,.1f} MW")

    # By region statistics
    print("\n" + "-" * 40)
    print("BY REGION (Total Curtailment MW):")
    print("-" * 40)

    region_totals = df.groupby("region")["value"].sum().sort_values(ascending=False)
    for region, total in region_totals.items():
        print(f"  {region}: {total:,.1f} MW")

    # By metric and region
    print("\n" + "-" * 40)
    print("BY REGION AND TYPE (MW):")
    print("-" * 40)

    pivot_summary = df.pivot_table(values="value", index="region", columns="metric", aggfunc="sum", fill_value=0)

    print(pivot_summary.round(1))


def main():
    """Main function to run the curtailment analysis."""

    # Initialize the client
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    api_url = os.getenv("OPENELECTRICITY_API_URL", "https://api.openelectricity.org.au/v4")

    if not api_key:
        print("Error: OPENELECTRICITY_API_KEY environment variable not set")
        print("Please set your API key in the .env file or environment")
        return

    client = OEClient(api_key=api_key, base_url=api_url)

    print("OpenElectricity Curtailment Analysis")
    print("=" * 60)

    # Fetch data for the last 7 days (API limit for 5m interval)
    df = fetch_curtailment_data(client, days_back=7)

    if df.empty:
        print("No curtailment data retrieved")
        return

    print(f"\nRetrieved {len(df)} data points")

    # Print summary statistics
    print_summary_statistics(df)

    # Create line chart
    print("\nGenerating line chart...")
    create_curtailment_line_chart(df)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
