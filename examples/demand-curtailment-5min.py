#!/usr/bin/env python
"""
5-minute interval demand and curtailment charts for all NEM regions
"""

from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openelectricity import OEClient
from openelectricity.styles import set_openelectricity_style
from openelectricity.types import MarketMetric

# Initialize client
client = OEClient()

# Apply OpenElectricity style
set_openelectricity_style()

# Get data for last 7 days
start_time = datetime.now() - timedelta(days=7)

# Define NEM regions
nem_regions = [
    ("NSW1", "New South Wales"),
    ("QLD1", "Queensland"),
    ("VIC1", "Victoria"),
    ("SA1", "South Australia"),
    ("TAS1", "Tasmania")
]

# Create figure with 5 subplots (one for each region)
fig, axes = plt.subplots(5, 1, figsize=(14, 16))
fig.suptitle('NEM Regional Demand and Curtailment (5-minute intervals, last 7 days)', 
             fontsize=16, fontweight='bold', y=0.995)

# Process each region
for idx, (region_code, region_name) in enumerate(nem_regions):
    print(f"Fetching data for {region_code}...")
    
    # Fetch 5-minute interval data
    response = client.get_market(
        network_code="NEM",
        network_region=region_code,
        metrics=[
            MarketMetric.DEMAND,
            MarketMetric.CURTAILMENT,
        ],
        interval="5m",
        date_start=pd.to_datetime(start_time),
        # date_end omitted to get latest data
    )
    
    # Process the response
    data = []
    for timeseries in response.data:
        metric = timeseries.metric
        
        for result in timeseries.results:
            for data_point in result.data:
                timestamp = data_point.timestamp if hasattr(data_point, "timestamp") else data_point.root[0]
                value = data_point.value if hasattr(data_point, "value") else data_point.root[1]
                if value is not None:
                    data.append({
                        "time": pd.to_datetime(timestamp),
                        "metric": metric,
                        "value": value
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        print(f"No data available for {region_code}")
        axes[idx].text(0.5, 0.5, f'No data available for {region_name}',
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[idx].transAxes)
        axes[idx].set_title(f'{region_name}')
        continue
    
    # Pivot data by metric
    pivot_df = df.pivot(index='time', columns='metric', values='value')
    pivot_df = pivot_df.fillna(0)
    
    # Get demand and curtailment columns
    demand_value = pivot_df.get('demand', 0)
    curtailment_value = pivot_df.get('curtailment', 0)
    
    ax = axes[idx]
    
    # Plot demand
    ax.plot(pivot_df.index, demand_value, 
            label='Total Demand', color='#1f77b4', linewidth=1.5)
    
    # Plot curtailment
    ax.plot(pivot_df.index, curtailment_value, 
            label='Curtailment', color='#ff7f0e', linewidth=1.5, alpha=0.8)
    
    # Fill area under curtailment for better visibility
    ax.fill_between(pivot_df.index, 0, curtailment_value, 
                    color='#ff7f0e', alpha=0.3)
    
    # Formatting
    ax.set_title(f'{region_name}', fontweight='bold', fontsize=12)
    ax.set_ylabel('Power (MW)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add statistics
    if isinstance(demand_value, pd.Series):
        avg_demand = demand_value.mean()
        max_demand = demand_value.max()
    else:
        avg_demand = 0
        max_demand = 0
    
    if isinstance(curtailment_value, pd.Series):
        total_curtailment = curtailment_value.sum() / 12  # Convert to MWh (5min = 1/12 hour)
        max_curtailment = curtailment_value.max()
    else:
        total_curtailment = 0
        max_curtailment = 0
    
    stats_text = (f'Avg Demand: {avg_demand:,.0f} MW | '
                 f'Max Demand: {max_demand:,.0f} MW\n'
                 f'Total Curtailed: {total_curtailment:,.0f} MWh | '
                 f'Max Curtailment: {max_curtailment:,.0f} MW')
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Set common x-label only on bottom plot
axes[-1].set_xlabel('Time')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
output_file = 'demand-curtailment-5min.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nChart saved to {output_file}")

# Also show the plot
plt.show()

print("\nChart generation complete!")