#!/usr/bin/env python
"""
Example: Curtailment Analysis with Charts

This example demonstrates how to:
1. Fetch curtailment data for solar and wind
2. Process data for all NEM regions
3. Create line charts showing daily curtailment trends
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

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


def fetch_curtailment_data(client: OEClient, days_back: int = 7) -> pd.DataFrame:
    """
    Fetch curtailment data for all NEM regions.
    
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
    
    # NEM regions
    regions = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
    
    all_data = []
    
    for region in regions:
        print(f"  Fetching data for {region}...")
        
        try:
            # Fetch curtailment data
            response = client.get_market(
                network_code="NEM",
                metrics=[MarketMetric.CURTAILMENT_SOLAR, MarketMetric.CURTAILMENT_WIND],
                interval="1d",
                date_start=pd.to_datetime(date_start),
                date_end=pd.to_datetime(date_end),
                network_region=region,
                primary_grouping="network_region"
            )
            
            # Process each metric
            for timeseries in response.data:
                metric = timeseries.metric
                
                for result in timeseries.results:
                    # Extract data points
                    for data_point in result.data:
                        # data_point is a TimeSeriesDataPoint object
                        timestamp = data_point.timestamp if hasattr(data_point, 'timestamp') else data_point.root[0]
                        value = data_point.value if hasattr(data_point, 'value') else data_point.root[1]
                        if value is not None:
                            all_data.append({
                                "region": region,
                                "metric": metric,
                                "date": pd.to_datetime(timestamp),
                                "value": value
                            })
                            
        except Exception as e:
            print(f"    Error fetching data for {region}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by date
    if not df.empty:
        df = df.sort_values("date")
    
    return df


def create_curtailment_charts(df: pd.DataFrame):
    """
    Create stacked bar charts showing curtailment trends for each region.
    
    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Convert MW to MWh (divide by 12 as there are 12 5-minute intervals per hour)
    # Note: Since we're using daily data, MW values represent daily averages
    # To get MWh for the day: MW * 24 hours
    df["value_mwh"] = df["value"] * 24  # Convert daily average MW to MWh per day
    
    # Create pivot tables for solar and wind
    solar_df = df[df["metric"] == "curtailment_solar"].copy()
    wind_df = df[df["metric"] == "curtailment_wind"].copy()
    
    if not solar_df.empty:
        solar_pivot = solar_df.pivot(index="date", columns="region", values="value_mwh")
    else:
        solar_pivot = pd.DataFrame()
    
    if not wind_df.empty:
        wind_pivot = wind_df.pivot(index="date", columns="region", values="value_mwh")
    else:
        wind_pivot = pd.DataFrame()
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot solar curtailment as stacked bar chart
    ax1 = axes[0]
    if not solar_pivot.empty:
        solar_pivot.plot(kind="bar", stacked=True, ax=ax1, width=0.8)
        ax1.set_title("Solar Curtailment by Region (Daily Total)", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Curtailment (MWh)", fontsize=12)
        ax1.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis labels
        ax1.set_xticklabels([d.strftime('%b %d') for d in solar_pivot.index], rotation=45, ha='right')
        
        # Add horizontal line at zero
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Plot wind curtailment as stacked bar chart
    ax2 = axes[1]
    if not wind_pivot.empty:
        wind_pivot.plot(kind="bar", stacked=True, ax=ax2, width=0.8)
        ax2.set_title("Wind Curtailment by Region (Daily Total)", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Curtailment (MWh)", fontsize=12)
        ax2.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis labels
        ax2.set_xticklabels([d.strftime('%b %d') for d in wind_pivot.index], rotation=45, ha='right')
        
        # Add horizontal line at zero
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.suptitle("NEM Renewable Energy Curtailment Analysis (MWh)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    # Save the figure
    output_file = "curtailment_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {output_file}")
    
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
    combined_pivot = df.pivot_table(
        values="value_mwh",
        index="date",
        columns=["region", "metric"],
        aggfunc="sum",
        fill_value=0
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for stacked bar chart
    # Reorganize to have solar and wind as separate layers for each region
    dates = combined_pivot.index
    regions = df["region"].unique()
    
    # Define colors for each region (solar=lighter, wind=darker)
    region_colors = {
        'NSW1': ('#FFB366', '#FF6B00'),  # Orange
        'QLD1': ('#66B3FF', '#0066CC'),  # Blue
        'VIC1': ('#90EE90', '#228B22'),  # Green
        'SA1': ('#FFB6C1', '#DC143C'),  # Pink/Red
        'TAS1': ('#DDA0DD', '#8B008B')   # Purple
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
        if (region, 'curtailment_solar') in combined_pivot.columns:
            solar_values = combined_pivot[(region, 'curtailment_solar')].values
            ax.bar(x_pos, solar_values, bar_width, 
                   bottom=bottom_solar, 
                   label=f'{region} Solar',
                   color=region_colors.get(region, ('#888888', '#444444'))[0],
                   alpha=0.8)
            bottom_solar = [b + v for b, v in zip(bottom_solar, solar_values)]
        
        # Wind curtailment
        if (region, 'curtailment_wind') in combined_pivot.columns:
            wind_values = combined_pivot[(region, 'curtailment_wind')].values
            ax.bar(x_pos, wind_values, bar_width,
                   bottom=bottom_wind,
                   label=f'{region} Wind',
                   color=region_colors.get(region, ('#888888', '#444444'))[1],
                   alpha=0.8,
                   hatch='//')  # Add pattern to distinguish wind from solar
            bottom_wind = [b + v for b, v in zip(bottom_wind, wind_values)]
    
    # Format the chart
    ax.set_title("Total Renewable Curtailment by Region and Type (MWh)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total Curtailment (MWh)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.strftime('%b %d') for d in dates], rotation=45, ha='right')
    ax.legend(title="Region & Type", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
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
    
    # Convert MW to MWh for daily totals
    df["value_mwh"] = df["value"] * 24
    
    print("\n" + "=" * 60)
    print("CURTAILMENT SUMMARY STATISTICS (MWh)")
    print("=" * 60)
    
    # Overall statistics
    solar_data = df[df["metric"] == "curtailment_solar"]
    wind_data = df[df["metric"] == "curtailment_wind"]
    
    if not solar_data.empty:
        print("\nSolar Curtailment:")
        print(f"  Total: {solar_data['value_mwh'].sum():,.1f} MWh")
        print(f"  Average: {solar_data['value_mwh'].mean():,.1f} MWh/day")
        print(f"  Maximum: {solar_data['value_mwh'].max():,.1f} MWh/day")
    
    if not wind_data.empty:
        print("\nWind Curtailment:")
        print(f"  Total: {wind_data['value_mwh'].sum():,.1f} MWh")
        print(f"  Average: {wind_data['value_mwh'].mean():,.1f} MWh/day")
        print(f"  Maximum: {wind_data['value_mwh'].max():,.1f} MWh/day")
    
    # By region statistics
    print("\n" + "-" * 40)
    print("BY REGION (Total Curtailment MWh):")
    print("-" * 40)
    
    region_totals = df.groupby("region")["value_mwh"].sum().sort_values(ascending=False)
    for region, total in region_totals.items():
        print(f"  {region}: {total:,.1f} MWh")
    
    # By metric and region
    print("\n" + "-" * 40)
    print("BY REGION AND TYPE (MWh):")
    print("-" * 40)
    
    pivot_summary = df.pivot_table(
        values="value_mwh",
        index="region",
        columns="metric",
        aggfunc="sum",
        fill_value=0
    )
    
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
    
    # Fetch data for the last 14 days
    df = fetch_curtailment_data(client, days_back=14)
    
    if df.empty:
        print("No curtailment data retrieved")
        return
    
    print(f"\nRetrieved {len(df)} data points")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create charts
    print("\nGenerating charts...")
    create_curtailment_charts(df)
    create_combined_chart(df)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()