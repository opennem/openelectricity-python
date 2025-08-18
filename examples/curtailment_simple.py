#!/usr/bin/env python
"""
Example: Simple Curtailment Analysis

This example demonstrates how to fetch and analyze curtailment data
without requiring additional dependencies like matplotlib or seaborn.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

from openelectricity import OEClient
from openelectricity.types import MarketMetric

# Load environment variables
load_dotenv()


def fetch_curtailment_data(client: OEClient, region: str = "NSW1", days_back: int = 7) -> pd.DataFrame:
    """
    Fetch curtailment data for a specific region.
    
    Args:
        client: OpenElectricity API client
        region: NEM region code
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
    
    print(f"Fetching curtailment data for {region}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
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
    
    # Process the response
    data = []
    for timeseries in response.data:
        metric = timeseries.metric
        unit = timeseries.unit
        
        for result in timeseries.results:
            for data_point in result.data:
                # data_point is a TimeSeriesDataPoint object
                timestamp = data_point.timestamp if hasattr(data_point, 'timestamp') else data_point.root[0]
                value = data_point.value if hasattr(data_point, 'value') else data_point.root[1]
                if value is not None:
                    data.append({
                        "date": pd.to_datetime(timestamp).date(),
                        "metric": metric,
                        "value": value,
                        "unit": unit,
                        "region": region
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date
    if not df.empty:
        df = df.sort_values("date")
    
    return df


def analyze_curtailment(df: pd.DataFrame):
    """
    Analyze and display curtailment statistics.
    
    Args:
        df: DataFrame with curtailment data
    """
    if df.empty:
        print("No data to analyze")
        return
    
    print("\n" + "=" * 60)
    print("CURTAILMENT ANALYSIS")
    print("=" * 60)
    
    # Separate solar and wind data
    solar_df = df[df["metric"] == "curtailment_solar"]
    wind_df = df[df["metric"] == "curtailment_wind"]
    
    # Daily breakdown
    print("\nDaily Curtailment (MW):")
    print("-" * 40)
    
    # Create pivot table for better display
    pivot_df = df.pivot_table(
        values="value",
        index="date",
        columns="metric",
        aggfunc="first"
    )
    
    # Add total column
    pivot_df["total"] = pivot_df.sum(axis=1)
    
    # Format the output
    for date, row in pivot_df.iterrows():
        print(f"\n{date}:")
        if "curtailment_solar" in row and pd.notna(row["curtailment_solar"]):
            print(f"  Solar:  {row['curtailment_solar']:8.2f} MW")
        if "curtailment_wind" in row and pd.notna(row["curtailment_wind"]):
            print(f"  Wind:   {row['curtailment_wind']:8.2f} MW")
        print(f"  Total:  {row['total']:8.2f} MW")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    if not solar_df.empty:
        # Convert to MWh for daily totals (MW * 24 hours)
        solar_total_mwh = solar_df['value'].sum() * 24
        solar_avg_mwh = solar_df['value'].mean() * 24
        solar_max_mwh = solar_df['value'].max() * 24
        solar_min_mwh = solar_df['value'].min() * 24
        
        print("\nSolar Curtailment:")
        print(f"  Total:    {solar_total_mwh:10.2f} MWh")
        print(f"  Average:  {solar_avg_mwh:10.2f} MWh/day")
        print(f"  Maximum:  {solar_max_mwh:10.2f} MWh/day")
        print(f"  Minimum:  {solar_min_mwh:10.2f} MWh/day")
    
    if not wind_df.empty:
        # Convert to MWh for daily totals (MW * 24 hours)
        wind_total_mwh = wind_df['value'].sum() * 24
        wind_avg_mwh = wind_df['value'].mean() * 24
        wind_max_mwh = wind_df['value'].max() * 24
        wind_min_mwh = wind_df['value'].min() * 24
        
        print("\nWind Curtailment:")
        print(f"  Total:    {wind_total_mwh:10.2f} MWh")
        print(f"  Average:  {wind_avg_mwh:10.2f} MWh/day")
        print(f"  Maximum:  {wind_max_mwh:10.2f} MWh/day")
        print(f"  Minimum:  {wind_min_mwh:10.2f} MWh/day")
    
    # Overall statistics
    print("\nOverall:")
    print(f"  Total curtailment:     {df['value'].sum():10.2f} MW")
    print(f"  Average per day:       {df.groupby('date')['value'].sum().mean():10.2f} MW")
    
    # Proportion analysis
    if not solar_df.empty and not wind_df.empty:
        solar_total = solar_df['value'].sum()
        wind_total = wind_df['value'].sum()
        total = solar_total + wind_total
        
        print(f"\nCurtailment breakdown:")
        print(f"  Solar: {solar_total/total*100:5.1f}% ({solar_total:.1f} MW)")
        print(f"  Wind:  {wind_total/total*100:5.1f}% ({wind_total:.1f} MW)")


def main():
    """Main function to run the curtailment analysis."""
    
    # Initialize the client
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    api_url = os.getenv("OPENELECTRICITY_API_URL", "https://api.openelectricity.org.au/v4/")
    
    if not api_key:
        print("Error: OPENELECTRICITY_API_KEY environment variable not set")
        print("Please set your API key in the .env file or environment")
        return
    
    client = OEClient(api_key=api_key, base_url=api_url)
    
    print("OpenElectricity Curtailment Analysis Example")
    print("=" * 60)
    
    # Analyze curtailment for each NEM region
    regions = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
    
    all_data = []
    
    for region in regions:
        print(f"\n\nAnalyzing {region}...")
        print("-" * 40)
        
        try:
            # Fetch data for this region
            df = fetch_curtailment_data(client, region=region, days_back=7)
            
            if not df.empty:
                all_data.append(df)
                # Analyze the data
                analyze_curtailment(df)
            else:
                print(f"No curtailment data available for {region}")
                
        except Exception as e:
            print(f"Error fetching data for {region}: {e}")
    
    # Combined analysis
    if all_data:
        print("\n\n" + "=" * 60)
        print("COMBINED ANALYSIS (ALL REGIONS)")
        print("=" * 60)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Summary by region
        print("\nTotal Curtailment by Region (last 7 days):")
        print("-" * 40)
        
        region_totals = combined_df.groupby("region")["value"].sum().sort_values(ascending=False)
        for region, total in region_totals.items():
            print(f"  {region}: {total:10.2f} MW")
        
        # Summary by type
        print("\nTotal Curtailment by Type (all regions):")
        print("-" * 40)
        
        type_totals = combined_df.groupby("metric")["value"].sum()
        for metric, total in type_totals.items():
            print(f"  {metric.replace('curtailment_', '').title()}: {total:10.2f} MW")
        
        # Overall total
        print(f"\nTotal NEM Curtailment: {combined_df['value'].sum():10.2f} MW")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()