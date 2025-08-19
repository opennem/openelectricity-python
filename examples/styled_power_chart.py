#!/usr/bin/env python3
"""
Example of creating a styled power generation chart matching OpenElectricity branding.

This example fetches the last 3 days of NEM power generation data and creates
a stacked area chart with OpenElectricity styling and colors.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from openelectricity import OEClient
from openelectricity.styles import (
    BRAND_COLORS,
    create_styled_figure,
    format_chart,
    get_fueltech_color,
    set_openelectricity_style,
)
from openelectricity.types import DataMetric


def create_sample_data(days_back: int = 3) -> pd.DataFrame:
    """Create sample power generation data for demonstration."""

    # Generate time index
    periods = days_back * 48  # 30-minute intervals
    timestamps = pd.date_range(end=datetime.now(), periods=periods, freq="30min")

    # Generate realistic patterns for each fuel type
    hours = np.array([(t.hour + t.minute / 60) for t in timestamps])
    days = np.array([t.dayofyear for t in timestamps])

    data = pd.DataFrame(index=timestamps)

    # Solar - peaks at midday, zero at night
    solar_pattern = np.maximum(0, 4000 * np.sin(np.maximum(0, np.pi * (hours - 6) / 12)))
    data["solar_rooftop"] = solar_pattern * (0.4 + 0.1 * np.random.random(periods))
    data["solar_utility"] = solar_pattern * (0.6 + 0.1 * np.random.random(periods))

    # Wind - variable throughout day
    data["wind"] = 2500 + 1000 * np.sin(hours * np.pi / 12) + 500 * np.random.random(periods)

    # Coal - baseload with some variation
    data["coal_black"] = 3500 + 200 * np.random.random(periods)
    data["coal_brown"] = 2000 + 150 * np.random.random(periods)

    # Gas - follows demand curve
    demand_pattern = 1 + 0.3 * np.sin(np.pi * (hours - 14) / 12)
    data["gas_ccgt"] = 1500 * demand_pattern + 100 * np.random.random(periods)
    data["gas_ocgt"] = 800 * demand_pattern + 50 * np.random.random(periods)

    # Hydro - relatively stable
    data["hydro"] = 1800 + 200 * np.sin(hours * np.pi / 24) + 100 * np.random.random(periods)

    # Battery - charging during solar peak, discharging in evening
    data["battery_discharging"] = np.maximum(0, 500 * np.sin(np.pi * (hours - 18) / 6))
    data["battery_charging"] = np.maximum(0, -400 * np.sin(np.pi * (hours - 12) / 6))

    # Ensure all values are positive
    data = data.clip(lower=0)

    return data


def fetch_power_data(client: OEClient, days_back: int = 3) -> pd.DataFrame:
    """Fetch power generation data grouped by fuel technology."""

    print("Fetching power generation data...")

    try:
        # Get data for the last 3 days
        # Note: API uses different grouping than expected, using secondary_grouping instead
        response = client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.POWER],
            interval="5m",  # 5-minute intervals (30m not supported)
            date_start=datetime.now() - timedelta(days=days_back),
            primary_grouping="network",
            secondary_grouping="fueltech",
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using sample data instead...")
        return create_sample_data(days_back)

    # Convert to DataFrame
    data = []
    for timeseries in response.data:
        for result in timeseries.results:
            fueltech = result.name.replace("power.", "").replace("fuel_tech.", "")
            for data_point in result.data:
                if data_point.value is not None and data_point.value >= 0:
                    data.append({"timestamp": data_point.timestamp, "fueltech": fueltech, "power_mw": data_point.value})

    df = pd.DataFrame(data)

    # Pivot to get fueltechs as columns
    df_pivot = df.pivot_table(index="timestamp", columns="fueltech", values="power_mw", aggfunc="sum", fill_value=0)

    return df_pivot


def order_fueltechs(df: pd.DataFrame) -> list:
    """Order fuel technologies for stacking (renewables on top, fossil fuels at bottom)."""

    # Define display order (bottom to top in stack)
    order = [
        # Base load / fossil fuels (bottom)
        "coal_black",
        "coal_brown",
        "gas_steam",
        "gas_ccgt",
        "gas_ocgt",
        "gas_reciprocating",
        "gas_waste_coal_mine",
        "distillate",
        "bioenergy_biomass",
        # Storage and hydro (middle)
        "hydro",
        "pumps",
        "battery_charging",
        # Variable renewables (top)
        "battery_discharging",
        "wind",
        "solar_utility",
        "solar_rooftop",
    ]

    # Filter to only include columns that exist in the dataframe
    ordered_cols = [col for col in order if col in df.columns]

    # Add any remaining columns not in our order
    remaining = [col for col in df.columns if col not in ordered_cols]

    return ordered_cols + remaining


def create_power_chart(df: pd.DataFrame) -> None:
    """Create a stacked area chart of power generation by fuel technology."""

    # Apply OpenElectricity styling
    set_openelectricity_style()

    # Create figure
    fig, ax = create_styled_figure(figsize=(14, 8))

    # Order fuel technologies
    ordered_fueltechs = order_fueltechs(df)
    df_ordered = df[ordered_fueltechs]

    # Get colors for each fuel technology
    colors = [get_fueltech_color(ft) for ft in ordered_fueltechs]

    # Create stacked area chart
    ax.stackplot(
        df_ordered.index,
        *[df_ordered[col].values for col in ordered_fueltechs],
        labels=[ft.replace("_", " ").title() for ft in ordered_fueltechs],
        colors=colors,
        alpha=0.9,
    )

    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # Calculate statistics for title
    latest_time = df.index[-1]
    total_power = df_ordered.iloc[-1].sum()

    # Format the chart
    format_chart(
        ax,
        title=f"NEM Generation by Fuel Technology - {total_power:,.0f} MW at {latest_time.strftime('%H:%M %d %b %Y')}",
        xlabel=None,  # Date axis is self-explanatory
        ylabel="Generation (MW)",
        add_logo=True,
        logo_position=(0.98, 0.02),
        logo_size=0.18,  # Prominent watermark
        logo_alpha=0.15,  # Very subtle transparency
    )

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Add horizontal line at 0
    ax.axhline(y=0, color=BRAND_COLORS["text"], linewidth=0.5, alpha=0.5)

    # Create custom legend (similar to website)
    # Calculate totals for legend
    latest_values = df_ordered.iloc[-1]
    total = latest_values.sum()

    # Create legend entries with values and percentages
    legend_elements = []
    for ft in ordered_fueltechs:
        if latest_values[ft] > 0:  # Only show non-zero values
            value = latest_values[ft]
            percentage = (value / total) * 100 if total > 0 else 0
            label = f"{ft.replace('_', ' ').title()}: {value:,.0f} MW ({percentage:.1f}%)"
            color = get_fueltech_color(ft)
            legend_elements.append(Patch(facecolor=color, label=label, alpha=0.9))

    # Add legend to the right of the plot
    ax.legend(
        handles=legend_elements[::-1],  # Reverse to match stack order
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=9,
        title="Current Generation",
        title_fontsize=10,
    )

    # Add grid styling
    ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits
    ax.set_ylim(bottom=0)
    max_val = df_ordered.sum(axis=1).max()
    ax.set_ylim(top=max_val * 1.05)  # Add 5% padding at top

    # Save the figure first
    output_path = Path(__file__).parent / "power_generation_styled.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BRAND_COLORS["background"])
    print(f"\nChart saved to: {output_path}")

    # Show the plot only if running interactively
    import sys

    if sys.flags.interactive or (hasattr(sys, "ps1")):
        plt.show()
    else:
        print("Run in interactive mode or view saved file to see the chart")
        plt.close(fig)


def main():
    """Main function to create styled power generation chart."""

    print("OpenElectricity Styled Power Generation Chart")
    print("=" * 50)

    # Try to fetch real data, fall back to sample if needed
    try:
        # Initialize client
        with OEClient() as client:
            # Fetch data
            df = fetch_power_data(client, days_back=3)
    except Exception as e:
        print(f"Could not initialize client: {e}")
        print("Using sample data for demonstration...")
        df = create_sample_data(days_back=3)

    if df.empty:
        print("No data available")
        return

    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Fuel technologies: {', '.join(df.columns)}")

    # Create chart
    create_power_chart(df)


if __name__ == "__main__":
    main()
