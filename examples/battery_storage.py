#!/usr/bin/env python
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from openelectricity import OEClient
from openelectricity.types import DataMetric

# Initialize client with local API
client = OEClient()


def fetch_battery_storage_august():
    # First, get facility information to get the maximum capacity for BALB1
    facilities_response = client.get_facilities(facility_code=["BALBESS"])

    # Find the BALB1 unit's maximum capacity (storage capacity in MWh)
    balb1_capacity_storage = None
    if facilities_response.data:
        for facility in facilities_response.data:
            for unit in facility.units:
                if unit.code == "BALB1":  # The bidirectional unit
                    balb1_capacity_storage = unit.capacity_storage
                    print(f"Found BALB1 unit with storage capacity: {balb1_capacity_storage} MWh")
                    break

    # Fetch hourly battery storage data for BALBESS for August 2025
    # Note: API returns data for all units, we'll filter for BALB1 in processing
    response = client.get_facility_data(
        network_code="NEM",
        facility_code="BALBESS",
        metrics=[DataMetric.STORAGE_BATTERY],
        interval="1h",
        date_start=datetime(2025, 8, 1, 0, 0, 0),
        date_end=datetime(2025, 8, 31, 23, 59, 59),
    )

    if not response.data:
        print("No data available")
        return

    # Process data for BALB1 unit only
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    color = "#1f77b4"

    for result in response.data[0].results:
        unit_code = getattr(result.columns, "unit_code", result.name)

        # Skip if not BALB1 (though we should only get BALB1 data)
        if unit_code != "BALB1":
            continue

        # Extract timestamps and values
        # Data is a list of TimeSeriesDataPoint objects with timestamp and value attributes
        timestamps = [pd.Timestamp(item.timestamp) for item in result.data]
        values = [item.value if item.value is not None else 0 for item in result.data]

        # Create DataFrame for easier manipulation
        df = pd.DataFrame({"timestamp": timestamps, "storage_mwh": values})
        df.set_index("timestamp", inplace=True)

        # Plot the data
        ax.plot(df.index, df["storage_mwh"], color=color, linewidth=1, alpha=0.8)
        ax.fill_between(df.index, 0, df["storage_mwh"], color=color, alpha=0.3)

        # Add maximum capacity line if available
        if balb1_capacity_storage:
            ax.axhline(
                y=balb1_capacity_storage,
                color="green",
                linestyle="-",
                alpha=0.5,
                label=f"Max Capacity: {balb1_capacity_storage:.1f} MWh",
            )

        # Calculate statistics
        valid_values = [v for v in values if v > 0]
        if valid_values:
            avg = sum(valid_values) / len(valid_values)
            ax.axhline(y=avg, color="red", linestyle="--", alpha=0.5, label=f"Avg: {avg:.1f} MWh")

        # Formatting
        ax.set_ylabel("Storage (MWh)")
        ax.set_title(f"{unit_code} - Battery Storage Level (Bidirectional Unit)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        # Set y-limit based on max capacity or max values
        y_max = max(balb1_capacity_storage * 1.1 if balb1_capacity_storage else 30, max(values) * 1.1 if values else 30)
        ax.set_ylim(0, y_max)

    ax.set_xlabel("Date")
    fig.suptitle("BALB1 Battery Storage Levels - August 2025", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save the chart
    output_path = "battery_storage_balb1_august_2025.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Chart saved to {output_path}")
    if response.data and response.data[0].results:
        print(f"Data points: {len(response.data[0].results[0].data)} hourly readings for BALB1")


if __name__ == "__main__":
    fetch_battery_storage_august()
