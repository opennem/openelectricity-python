#!/usr/bin/env python3
"""
Demonstration of OpenElectricity styling features for charts.

This example shows how to use the styles module to create charts
that match the OpenElectricity brand guidelines.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openelectricity.styles import (
    BRAND_COLORS,
    FUELTECH_GROUPS,
    create_styled_figure,
    format_chart,
    get_fueltech_palette,
    set_openelectricity_style,
)


def demo_bar_chart():
    """Create a bar chart with fuel technology colors."""

    # Sample data
    fueltechs = ["Solar", "Wind", "Hydro", "Gas", "Coal", "Battery"]
    values = [4500, 3200, 2100, 5600, 8900, 450]

    # Create styled figure
    fig, ax = create_styled_figure(figsize=(10, 6))

    # Get colors
    colors = get_fueltech_palette(fueltechs)

    # Create bar chart
    bars = ax.bar(fueltechs, values, color=colors, alpha=0.9, edgecolor="white", linewidth=1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:,.0f} MW", ha="center", va="bottom", fontsize=10)

    # Format chart
    format_chart(ax, title="Generation Capacity by Technology", xlabel="Technology", ylabel="Capacity (MW)", add_logo=True)

    plt.tight_layout()
    return fig


def demo_line_chart():
    """Create a multi-line chart showing trends."""

    # Generate sample time series data
    dates = pd.date_range("2024-01-01", periods=96, freq="30min")

    # Simulate generation patterns
    solar = np.maximum(0, 3000 * np.sin(np.pi * np.arange(96) / 48) + np.random.normal(0, 200, 96))
    wind = 2000 + 1000 * np.sin(np.pi * np.arange(96) / 24) + np.random.normal(0, 300, 96)
    coal = 5000 + np.random.normal(0, 100, 96)

    # Create styled figure
    fig, ax = create_styled_figure(figsize=(12, 6))

    # Plot lines
    ax.plot(dates, solar, label="Solar", color=FUELTECH_GROUPS["Solar"], linewidth=2)
    ax.plot(dates, wind, label="Wind", color=FUELTECH_GROUPS["Wind"], linewidth=2)
    ax.plot(dates, coal, label="Coal", color=FUELTECH_GROUPS["Coal"], linewidth=2)

    # Format chart
    format_chart(ax, title="24-Hour Generation Profile", xlabel="Time", ylabel="Generation (MW)", add_logo=True, logo_alpha=0.6)

    # Add legend
    ax.legend(loc="upper left", frameon=True, fancybox=False, shadow=False)

    # Format x-axis
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

    plt.tight_layout()
    return fig


def demo_pie_chart():
    """Create a pie chart showing generation mix."""

    # Sample data - ensure it matches realistic generation mix
    data = {
        "Coal": 30,  # Coal typically dominant in Australia
        "Gas": 20,  # Gas is significant
        "Wind": 15,  # Growing renewable
        "Solar": 12,  # Growing renewable
        "Hydro": 15,  # Stable renewable
        "Battery": 3,  # Small but growing
        "Other": 5,  # Biomass, distillate, etc.
    }

    fueltechs = list(data.keys())
    values = list(data.values())
    colors = [FUELTECH_GROUPS[ft] for ft in fueltechs]

    # Create styled figure
    fig, ax = create_styled_figure(figsize=(10, 8))

    # Create pie chart with better formatting
    wedges, texts, autotexts = ax.pie(
        values,
        labels=[f"{ft}\n({v}%)" for ft, v in zip(fueltechs, values, strict=False)],
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 10},
        pctdistance=0.85,
        labeldistance=1.1,
    )

    # Enhance text - make percentages readable
    for i, autotext in enumerate(autotexts):
        # Use white text on dark colors, black on light colors
        if fueltechs[i] in ["Coal", "Gas", "Wind", "Battery"]:
            autotext.set_color("white")
        else:
            autotext.set_color("black")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(12)

    # Format chart
    format_chart(
        ax,
        title="Energy Generation Mix - Last 24 Hours",
        add_logo=True,
        logo_position=(0.95, 0.05),
        logo_size=0.18,  # Larger watermark
        logo_alpha=0.3,
    )

    plt.tight_layout()
    return fig


def demo_stacked_area():
    """Create a stacked area chart like the website."""

    # Generate sample data
    hours = pd.date_range("2024-01-01", periods=72, freq="1h")

    # Simulate generation patterns
    data = pd.DataFrame(
        {
            "timestamp": hours,
            "coal": 5000 + np.random.normal(0, 200, 72),
            "gas": 3000 + np.random.normal(0, 150, 72),
            "hydro": 2000 + np.random.normal(0, 100, 72),
            "wind": np.maximum(0, 2500 + 1000 * np.sin(np.pi * np.arange(72) / 36) + np.random.normal(0, 200, 72)),
            "solar": np.maximum(0, 4000 * np.maximum(0, np.sin(np.pi * np.arange(72) / 24)) + np.random.normal(0, 100, 72)),
        }
    )

    # Create styled figure
    fig, ax = create_styled_figure(figsize=(14, 7))

    # Prepare data for stacking
    x = data["timestamp"]
    y_data = [data[col].values for col in ["coal", "gas", "hydro", "wind", "solar"]]
    labels = ["Coal", "Gas", "Hydro", "Wind", "Solar"]
    colors = [FUELTECH_GROUPS[label] for label in labels]

    # Create stacked area chart
    ax.stackplot(x, *y_data, labels=labels, colors=colors, alpha=0.85)

    # Format chart
    format_chart(
        ax,
        title="NEM Generation Stack - 3 Day Profile",
        xlabel=None,
        ylabel="Generation (MW)",
        add_logo=True,
        logo_size=0.16,  # Larger watermark
        logo_alpha=0.3,
    )

    # Add legend
    ax.legend(loc="upper left", frameon=True, fancybox=False, shadow=False, ncol=5)

    # Format axes
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Format x-axis
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

    plt.tight_layout()
    return fig


def main():
    """Run all demo charts."""

    print("OpenElectricity Style Demonstrations")
    print("=" * 40)

    # Set the style globally
    set_openelectricity_style()

    # Create demo charts
    demos = [
        ("Bar Chart", demo_bar_chart),
        ("Line Chart", demo_line_chart),
        ("Pie Chart", demo_pie_chart),
        ("Stacked Area Chart", demo_stacked_area),
    ]

    for name, demo_func in demos:
        print(f"\nCreating {name}...")
        fig = demo_func()

        # Save figure
        output_path = Path(__file__).parent / f"style_demo_{name.lower().replace(' ', '_')}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BRAND_COLORS["background"])
        print(f"Saved to: {output_path}")

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
