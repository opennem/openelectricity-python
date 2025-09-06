#!/usr/bin/env python
"""
Example: Curtailment Energy Chart Analysis (30 days)

This example demonstrates how to fetch and visualize curtailment ENERGY data (MWh)
for renewable energy sources using stacked bar charts showing the last 30 days
of curtailment by region.

Use this for energy accounting and visualization of curtailed energy patterns.
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
plt.rcParams["figure.figsize"] = (16, 10)


def fetch_curtailment_energy(client: OEClient, days_back: int = 30) -> pd.DataFrame:
    """
    Fetch daily curtailment energy data for all NEM regions.
    
    Args:
        client: OpenElectricity API client
        days_back: Number of days to fetch data for
        
    Returns:
        DataFrame with curtailment energy data
    """
    # Calculate date range (omit end_date to get latest)
    start_date = datetime.now() - timedelta(days=days_back)
    
    print(f"Fetching curtailment energy data for the last {days_back} days")
    print("  Fetching data for all NEM regions...")
    
    # Fetch daily curtailment energy data for ALL regions
    response = client.get_market(
        network_code="NEM",
        metrics=[
            MarketMetric.CURTAILMENT_ENERGY,
            MarketMetric.CURTAILMENT_SOLAR_UTILITY_ENERGY,
            MarketMetric.CURTAILMENT_WIND_ENERGY
        ],
        interval="1d",
        date_start=pd.to_datetime(start_date),
        # date_end omitted to get latest data
        primary_grouping="network_region",
    )
    
    # Process the response
    data = []
    for timeseries in response.data:
        metric = timeseries.metric
        unit = timeseries.unit
        
        for result in timeseries.results:
            # Extract region from result name (format: metric_REGION)
            name_parts = result.name.split("_")
            region = name_parts[-1] if len(name_parts) > 1 else "Unknown"
            
            for data_point in result.data:
                timestamp = data_point.timestamp if hasattr(data_point, "timestamp") else data_point.root[0]
                value = data_point.value if hasattr(data_point, "value") else data_point.root[1]
                if value is not None:
                    data.append({
                        "date": pd.to_datetime(timestamp).date(),
                        "region": region,
                        "metric": metric,
                        "value": value,
                        "unit": unit
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date and region
    if not df.empty:
        df = df.sort_values(["date", "region"])
        print(f"  Retrieved data for regions: {', '.join(df['region'].unique())}")
    
    return df


def create_stacked_bar_chart(df: pd.DataFrame):
    """
    Create a stacked bar chart showing curtailment energy by region over time.
    
    Args:
        df: DataFrame with curtailment energy data
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Use total curtailment energy metric
    total_df = df[df["metric"] == "curtailment_energy"].copy()
    
    if total_df.empty:
        # Fall back to calculating from solar + wind if total not available
        solar_df = df[df["metric"] == "curtailment_solar_utility_energy"]
        wind_df = df[df["metric"] == "curtailment_wind_energy"]
        
        # Combine solar and wind data
        combined = []
        for date in df["date"].unique():
            for region in df["region"].unique():
                solar_val = solar_df[(solar_df["date"] == date) & (solar_df["region"] == region)]["value"].sum()
                wind_val = wind_df[(wind_df["date"] == date) & (wind_df["region"] == region)]["value"].sum()
                combined.append({
                    "date": date,
                    "region": region,
                    "value": solar_val + wind_val
                })
        total_df = pd.DataFrame(combined)
    
    # Pivot data for stacked bar chart
    pivot_df = total_df.pivot(index="date", columns="region", values="value")
    pivot_df = pivot_df.fillna(0)
    
    # Convert MWh to GWh
    pivot_df = pivot_df / 1000
    
    # Define colors for each region
    region_colors = {
        "NSW1": "#FF6B00",  # Orange
        "QLD1": "#0066CC",  # Blue
        "VIC1": "#228B22",  # Green
        "SA1": "#DC143C",   # Red
        "TAS1": "#8B008B"   # Purple
    }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create stacked bar chart
    bottom = None
    for region in pivot_df.columns:
        color = region_colors.get(region, "#888888")
        ax.bar(
            pivot_df.index,
            pivot_df[region],
            bottom=bottom,
            label=region,
            color=color,
            alpha=0.8,
            width=0.8
        )
        if bottom is None:
            bottom = pivot_df[region]
        else:
            bottom = bottom + pivot_df[region]
    
    # Format the chart
    ax.set_title("NEM Curtailment Energy by Region (Last 30 Days)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Curtailment Energy (GWh)", fontsize=12)
    ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format x-axis labels
    ax.set_xticks(pivot_df.index[::2])  # Show every other date
    ax.set_xticklabels([d.strftime("%b %d") for d in pivot_df.index[::2]], rotation=45, ha="right")
    
    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = "curtailment_energy_30days.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {output_file}")
    
    # plt.show()  # Uncomment to display the chart interactively


def create_split_chart(df: pd.DataFrame):
    """
    Create separate stacked bar charts for solar and wind curtailment energy.
    
    Args:
        df: DataFrame with curtailment energy data
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Separate solar and wind data
    solar_df = df[df["metric"] == "curtailment_solar_utility_energy"]
    wind_df = df[df["metric"] == "curtailment_wind_energy"]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Define colors for each region
    region_colors = {
        "NSW1": "#FF6B00",  # Orange
        "QLD1": "#0066CC",  # Blue
        "VIC1": "#228B22",  # Green
        "SA1": "#DC143C",   # Red
        "TAS1": "#8B008B"   # Purple
    }
    
    # Solar curtailment chart
    if not solar_df.empty:
        solar_pivot = solar_df.pivot(index="date", columns="region", values="value")
        solar_pivot = solar_pivot.fillna(0)
        
        # Convert MWh to GWh
        solar_pivot = solar_pivot / 1000
        
        bottom = None
        for region in solar_pivot.columns:
            color = region_colors.get(region, "#888888")
            ax1.bar(
                solar_pivot.index,
                solar_pivot[region],
                bottom=bottom,
                label=region,
                color=color,
                alpha=0.8,
                width=0.8
            )
            if bottom is None:
                bottom = solar_pivot[region]
            else:
                bottom = bottom + solar_pivot[region]
        
        ax1.set_title("Solar Curtailment Energy by Region", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Energy (GWh)", fontsize=11)
        ax1.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticks(solar_pivot.index[::2])
        ax1.set_xticklabels([d.strftime("%b %d") for d in solar_pivot.index[::2]], rotation=45, ha="right")
    
    # Wind curtailment chart
    if not wind_df.empty:
        wind_pivot = wind_df.pivot(index="date", columns="region", values="value")
        wind_pivot = wind_pivot.fillna(0)
        
        # Convert MWh to GWh
        wind_pivot = wind_pivot / 1000
        
        bottom = None
        for region in wind_pivot.columns:
            color = region_colors.get(region, "#888888")
            ax2.bar(
                wind_pivot.index,
                wind_pivot[region],
                bottom=bottom,
                label=region,
                color=color,
                alpha=0.8,
                width=0.8
            )
            if bottom is None:
                bottom = wind_pivot[region]
            else:
                bottom = bottom + wind_pivot[region]
        
        ax2.set_title("Wind Curtailment Energy by Region", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Energy (GWh)", fontsize=11)
        ax2.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(wind_pivot.index[::2])
        ax2.set_xticklabels([d.strftime("%b %d") for d in wind_pivot.index[::2]], rotation=45, ha="right")
    
    plt.suptitle("NEM Renewable Curtailment Energy Analysis (Last 30 Days)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    # Save the figure
    output_file = "curtailment_energy_split_30days.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSplit chart saved to {output_file}")
    
    # plt.show()  # Uncomment to display the chart interactively


def print_summary_statistics(df: pd.DataFrame):
    """
    Print summary statistics for curtailment energy data.
    
    Args:
        df: DataFrame with curtailment energy data
    """
    if df.empty:
        print("No data to summarize")
        return
    
    print("\n" + "=" * 60)
    print("CURTAILMENT ENERGY SUMMARY (30 Days)")
    print("=" * 60)
    
    # Overall statistics by type
    solar_total = df[df["metric"] == "curtailment_solar_utility_energy"]["value"].sum()
    wind_total = df[df["metric"] == "curtailment_wind_energy"]["value"].sum()
    total = df[df["metric"] == "curtailment_energy"]["value"].sum()
    
    if total == 0:
        total = solar_total + wind_total
    
    print("\nTotal Curtailment Energy:")
    print(f"  Solar: {solar_total/1000:,.1f} GWh ({solar_total:,.0f} MWh)")
    print(f"  Wind:  {wind_total/1000:,.1f} GWh ({wind_total:,.0f} MWh)")
    print(f"  Total: {total/1000:,.1f} GWh ({total:,.0f} MWh)")
    
    # By region statistics
    print("\n" + "-" * 40)
    print("BY REGION (Total Energy):")
    print("-" * 40)
    
    regions = sorted(df["region"].unique())
    region_totals = {}
    
    for region in regions:
        region_df = df[df["region"] == region]
        solar = region_df[region_df["metric"] == "curtailment_solar_utility_energy"]["value"].sum()
        wind = region_df[region_df["metric"] == "curtailment_wind_energy"]["value"].sum()
        total_region = region_df[region_df["metric"] == "curtailment_energy"]["value"].sum()
        
        if total_region == 0:
            total_region = solar + wind
        
        region_totals[region] = total_region
        
        print(f"\n{region}:")
        print(f"  Solar: {solar/1000:,.1f} GWh")
        print(f"  Wind:  {wind/1000:,.1f} GWh")
        print(f"  Total: {total_region/1000:,.1f} GWh")
        
        if total_region > 0:
            solar_pct = (solar / total_region) * 100
            wind_pct = (wind / total_region) * 100
            print(f"  Breakdown: Solar {solar_pct:.1f}%, Wind {wind_pct:.1f}%")
    
    # Find peak day
    daily_totals = df[df["metric"] == "curtailment_energy"].groupby("date")["value"].sum()
    if daily_totals.empty:
        # Calculate from solar + wind
        solar_daily = df[df["metric"] == "curtailment_solar_utility_energy"].groupby("date")["value"].sum()
        wind_daily = df[df["metric"] == "curtailment_wind_energy"].groupby("date")["value"].sum()
        daily_totals = solar_daily.add(wind_daily, fill_value=0)
    
    if not daily_totals.empty:
        peak_date = daily_totals.idxmax()
        peak_value = daily_totals.max()
        avg_daily = daily_totals.mean()
        
        print("\n" + "-" * 40)
        print("DAILY STATISTICS:")
        print("-" * 40)
        print(f"  Average daily curtailment: {avg_daily/1000:,.1f} GWh")
        print(f"  Peak curtailment day: {peak_date} ({peak_value/1000:,.1f} GWh)")


def main():
    """Main function to run the curtailment energy chart analysis."""
    
    # Initialize the client
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    api_url = os.getenv("OPENELECTRICITY_API_URL", "https://api.openelectricity.org.au/v4")
    
    if not api_key:
        print("Error: OPENELECTRICITY_API_KEY environment variable not set")
        print("Please set your API key in the .env file or environment")
        return
    
    client = OEClient(api_key=api_key, base_url=api_url)
    
    print("OpenElectricity Curtailment Energy Chart Analysis")
    print("=" * 60)
    print()
    
    try:
        # Fetch curtailment energy data for the last 30 days
        df = fetch_curtailment_energy(client, days_back=30)
        
        if df.empty:
            print("No curtailment data retrieved")
            return
        
        print(f"\nRetrieved {len(df)} data points")
        
        # Print summary statistics
        print_summary_statistics(df)
        
        # Create visualizations
        print("\nGenerating charts...")
        create_stacked_bar_chart(df)
        create_split_chart(df)
        
    except Exception as e:
        print(f"Error fetching curtailment data: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()