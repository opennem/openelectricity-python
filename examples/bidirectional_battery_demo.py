#!/usr/bin/env python
"""
Bidirectional Battery Power Demo - COLLIE_ESR5 (WEM)

This script demonstrates how bidirectional battery data is stored and returned
by the API. Understanding this is critical for frontend chart rendering.

KEY INSIGHT FOR FRONTEND DEVELOPERS:
=====================================
Bidirectional batteries (fueltech='battery') store SIGNED values:
  - NEGATIVE values = charging (consuming power from grid)
  - POSITIVE values = discharging (supplying power to grid)

This is DIFFERENT from split battery units:
  - battery_charging: always positive, frontend inverts via loadFuelTechs
  - battery_discharging: always positive, displayed as-is

The chart Y-domain must include BOTH positive AND negative values for
bidirectional batteries, or the negative (charging) data will be clipped.

FIX: In FacilityPowerChart.svelte's capacitySums calculation, add:
  if (unit.fueltech_id === 'battery') {
      positive += capacity;  // discharge capacity
      negative += capacity;  // charge capacity
  }
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from openelectricity import OEClient
from openelectricity.types import DataMetric


def fetch_and_chart_bidirectional_battery():
    """Fetch COLLIE_ESR5 power data and create a chart showing signed values."""

    # Use production API
    client = OEClient(base_url="https://api.openelectricity.org.au/v4/")

    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"Fetching COLLIE_ESR5 power data from {start_date.date()} to {end_date.date()}")

    # Fetch facility power data for COLLIE_ESR5 on WEM
    # This is a bidirectional battery unit with fueltech='battery'
    response = client.get_facility_data(
        network_code="WEM",
        facility_code="COLLIE_ESR5",
        metrics=[DataMetric.POWER],
        interval="5m",
        date_start=start_date,
        date_end=end_date,
    )

    if not response.data:
        print("No data returned from API")
        return

    # Process the response - show all units for comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    # Find the bidirectional unit (COLLIE_ESR5, not the split L1/G1 units)
    bidirectional_data = None
    for result in response.data[0].results:
        unit_code = getattr(result.columns, "unit_code", result.name) if result.columns else result.name
        fueltech = getattr(result.columns, "fueltech_id", None) if result.columns else None

        # Extract timestamps and values
        timestamps = [pd.Timestamp(item.timestamp) for item in result.data]
        values = [item.value if item.value is not None else 0 for item in result.data]

        # Count positive and negative values
        positive_count = sum(1 for v in values if v > 0)
        negative_count = sum(1 for v in values if v < 0)
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0

        print(f"\nUnit: {unit_code}")
        print(f"  Fueltech: {fueltech}")
        print(f"  Data points: {len(values)}")
        print(f"  Positive (discharging): {positive_count} intervals")
        print(f"  Negative (charging): {negative_count} intervals")
        print(f"  Power range: {min_val:.1f} MW to {max_val:.1f} MW")

        # Store bidirectional unit data for charting
        if unit_code == "COLLIE_ESR5":
            bidirectional_data = (timestamps, values)

    if not bidirectional_data:
        print("ERROR: COLLIE_ESR5 bidirectional unit data not found")
        return

    # Chart only the bidirectional unit
    timestamps, values = bidirectional_data
    df = pd.DataFrame({"timestamp": timestamps, "power_mw": values})
    df.set_index("timestamp", inplace=True)

    # Plot line
    ax.plot(df.index, df['power_mw'], color='#3145CE', linewidth=0.8, alpha=0.8, label='COLLIE_ESR5')

    # Fill areas with distinct colors for charge/discharge
    ax.fill_between(df.index, 0, df['power_mw'],
                   where=(df['power_mw'] >= 0),
                   color='#52A972', alpha=0.4, label='Discharging (+)')
    ax.fill_between(df.index, 0, df['power_mw'],
                   where=(df['power_mw'] < 0),
                   color='#E15C34', alpha=0.4, label='Charging (-)')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Formatting
    ax.set_ylabel("Power (MW)")
    ax.set_xlabel("Time")
    ax.set_title(f"COLLIE_ESR5 - Bidirectional Battery Power (Last 7 Days)\n"
                 f"fueltech='battery' → signed values: negative=charging, positive=discharging",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)

    # IMPORTANT: Set y-limits to include BOTH positive AND negative range
    # This is where the frontend bug occurs - it sets yMin=0 for bidirectional batteries
    y_min = min(df['power_mw'].min() * 1.1, -50)  # At least show -50 MW
    y_max = max(df['power_mw'].max() * 1.1, 50)   # At least show +50 MW
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save the chart
    output_path = "bidirectional_battery_collie_esr5.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {output_path}")

    # Show chart
    plt.show()


if __name__ == "__main__":
    fetch_and_chart_bidirectional_battery()
