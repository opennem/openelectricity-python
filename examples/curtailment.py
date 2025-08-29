#!/usr/bin/env python
"""
Example: Curtailment Power Data Analysis (5-minute intervals)

This example demonstrates how to fetch and analyze curtailment POWER data (MW)
for renewable energy sources at 5-minute intervals across all NEM regions.

Use this for real-time monitoring of curtailment generation.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

from openelectricity import OEClient
from openelectricity.types import MarketMetric

# Load environment variables
load_dotenv()


def fetch_curtailment_power(client: OEClient, days_back: int = 3) -> pd.DataFrame:
    """
    Fetch 5-minute curtailment power data for all NEM regions.
    
    Args:
        client: OpenElectricity API client
        days_back: Number of days to fetch data for
        
    Returns:
        DataFrame with curtailment power data
    """
    # Calculate date range (omit end_date to get latest)
    start_date = datetime.now() - timedelta(days=days_back)
    
    print(f"Fetching curtailment power data for the last {days_back} days (latest available)")
    print("  Fetching data for all NEM regions...")
    
    # Fetch 5-minute curtailment power data for ALL regions
    response = client.get_market(
        network_code="NEM",
        metrics=[
            MarketMetric.CURTAILMENT_SOLAR_UTILITY,
            MarketMetric.CURTAILMENT_WIND,
            MarketMetric.CURTAILMENT
        ],
        interval="5m",
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
                        "datetime": pd.to_datetime(timestamp),
                        "region": region,
                        "metric": metric,
                        "value": value,
                        "unit": unit
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by datetime and region
    if not df.empty:
        df = df.sort_values(["datetime", "region"])
    
    return df


def display_latest_values(df: pd.DataFrame):
    """
    Display the latest curtailment values for each region.
    
    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        print("No data to display")
        return
    
    print("\n" + "=" * 60)
    print("LATEST CURTAILMENT VALUES (MW)")
    print("=" * 60)
    
    # Get latest timestamp
    latest_time = df["datetime"].max()
    latest_df = df[df["datetime"] == latest_time]
    
    regions = sorted(df["region"].unique())
    
    for region in regions:
        print(f"\n{region}:")
        print("-" * 40)
        
        region_data = latest_df[latest_df["region"] == region]
        
        solar = region_data[region_data["metric"] == "curtailment_solar_utility"]["value"].sum()
        wind = region_data[region_data["metric"] == "curtailment_wind"]["value"].sum()
        total = region_data[region_data["metric"] == "curtailment"]["value"].sum()
        
        # If total is not available, calculate it
        if total == 0:
            total = solar + wind
        
        print(f"  Solar:  {solar:8.2f} MW")
        print(f"  Wind:   {wind:8.2f} MW")
        print(f"  Total:  {total:8.2f} MW")
    
    print(f"\nLatest data timestamp: {latest_time}")


def display_period_summary(df: pd.DataFrame):
    """
    Display summary statistics for the entire period.
    
    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        return
    
    print("\n" + "=" * 60)
    print("PERIOD SUMMARY (Last 3 Days)")
    print("=" * 60)
    
    regions = sorted(df["region"].unique())
    
    for region in regions:
        print(f"\n{region}:")
        print("-" * 40)
        
        region_df = df[df["region"] == region]
        
        # Solar statistics
        solar_df = region_df[region_df["metric"] == "curtailment_solar_utility"]
        if not solar_df.empty:
            solar_values = solar_df["value"].values
            print("  Solar Curtailment:")
            print(f"    Average: {solar_values.mean():8.2f} MW")
            print(f"    Maximum: {solar_values.max():8.2f} MW")
            print(f"    Minimum: {solar_values.min():8.2f} MW")
        
        # Wind statistics
        wind_df = region_df[region_df["metric"] == "curtailment_wind"]
        if not wind_df.empty:
            wind_values = wind_df["value"].values
            print("  Wind Curtailment:")
            print(f"    Average: {wind_values.mean():8.2f} MW")
            print(f"    Maximum: {wind_values.max():8.2f} MW")
            print(f"    Minimum: {wind_values.min():8.2f} MW")


def display_nem_summary(df: pd.DataFrame):
    """
    Display NEM-wide curtailment summary.
    
    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        return
    
    print("\n" + "=" * 60)
    print("OVERALL NEM CURTAILMENT (Current)")
    print("=" * 60)
    
    # Get latest timestamp
    latest_time = df["datetime"].max()
    latest_df = df[df["datetime"] == latest_time]
    
    total_solar = latest_df[latest_df["metric"] == "curtailment_solar_utility"]["value"].sum()
    total_wind = latest_df[latest_df["metric"] == "curtailment_wind"]["value"].sum()
    total_curtailment = latest_df[latest_df["metric"] == "curtailment"]["value"].sum()
    
    # If total is not available, calculate it
    if total_curtailment == 0:
        total_curtailment = total_solar + total_wind
    
    print(f"\nCurrent Solar Curtailment: {total_solar:10.2f} MW")
    print(f"Current Wind Curtailment:  {total_wind:10.2f} MW")
    print(f"Current Total Curtailment: {total_curtailment:10.2f} MW")
    
    if total_curtailment > 0:
        solar_pct = (total_solar / total_curtailment) * 100
        wind_pct = (total_wind / total_curtailment) * 100
        
        print(f"\nCurtailment Breakdown:")
        print(f"  Solar: {solar_pct:5.1f}%")
        print(f"  Wind:  {wind_pct:5.1f}%")


def display_time_series_sample(df: pd.DataFrame):
    """
    Display a sample of recent time series data.
    
    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        return
    
    print("\n" + "=" * 60)
    print("TIME SERIES SAMPLE (Last 5 intervals)")
    print("=" * 60)
    
    # Show NSW1 solar as an example
    nsw1_solar = df[(df["region"] == "NSW1") & (df["metric"] == "curtailment_solar_utility")]
    
    if not nsw1_solar.empty:
        print("\nNSW1 Solar Curtailment (Last 5 intervals):")
        last_five = nsw1_solar.tail(5)
        
        for _, row in last_five.iterrows():
            timestamp = row["datetime"]
            value = row["value"]
            print(f"  {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {value:8.2f} MW")


def main():
    """Main function to run the curtailment power analysis."""
    
    # Initialize the client
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    api_url = os.getenv("OPENELECTRICITY_API_URL", "https://api.openelectricity.org.au/v4")
    
    if not api_key:
        print("Error: OPENELECTRICITY_API_KEY environment variable not set")
        print("Please set your API key in the .env file or environment")
        return
    
    client = OEClient(api_key=api_key, base_url=api_url)
    
    print("OpenElectricity Curtailment Power Analysis (5-minute intervals)")
    print("=" * 60)
    print()
    
    try:
        # Fetch curtailment power data for the last 3 days
        df = fetch_curtailment_power(client, days_back=3)
        
        if df.empty:
            print("No curtailment data retrieved")
            return
        
        print(f"\nRetrieved {len(df)} data points")
        
        # Display various analyses
        display_latest_values(df)
        display_period_summary(df)
        display_nem_summary(df)
        display_time_series_sample(df)
        
    except Exception as e:
        print(f"Error fetching curtailment data: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()