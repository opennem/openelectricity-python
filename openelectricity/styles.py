"""OpenElectricity chart styling and branding utilities."""

import io
import urllib.request

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

# OpenElectricity brand colors
BRAND_COLORS = {
    "primary": "#8b3119",  # Rust/terracotta from logo
    "primary_light": "#9b3f23",
    "primary_dark": "#7b2315",
    "background": "#ffffff",
    "text": "#1a1a1a",
    "text_light": "#666666",
    "grid": "#e0e0e0",
}

# Fuel technology colors extracted from OpenElectricity tracker
FUELTECH_COLORS = {
    # Renewables - Greens and Blues
    "solar_rooftop": "#FFD700",  # Gold/Yellow
    "solar_utility": "#FFA500",  # Orange
    "solar": "#FFA500",  # Orange (generic solar)
    "wind": "#4A8E3C",  # Forest Green
    "hydro": "#4A90E2",  # Light Blue
    # Storage
    "battery_discharging": "#6366F1",  # Indigo
    "battery_charging": "#8B5CF6",  # Purple
    "pumps": "#06B6D4",  # Cyan
    # Fossil Fuels - Browns and Grays
    "gas_waste_coal_mine": "#8B4513",  # Saddle Brown
    "gas_reciprocating": "#A0522D",  # Sienna
    "gas_ccgt": "#D2691E",  # Chocolate
    "gas_ocgt": "#DEB887",  # Burlesque
    "gas_steam": "#F4A460",  # Sandy Brown
    "gas": "#D2691E",  # Generic gas - Chocolate
    # Coal
    "coal_black": "#2C2C2C",  # Very Dark Gray
    "coal_brown": "#654321",  # Dark Brown
    "coal": "#2C2C2C",  # Generic coal - Very Dark Gray
    # Other
    "distillate": "#DC143C",  # Crimson
    "bioenergy_biomass": "#8B7355",  # Tan
    "bioenergy": "#8B7355",  # Generic bioenergy
    # Imports/Exports
    "imports": "#9CA3AF",  # Gray
    "exports": "#6B7280",  # Darker Gray
    "interconnector": "#9CA3AF",  # Gray
}

# Simplified color mapping for common aggregations
FUELTECH_GROUPS = {
    "Solar": "#FFB300",  # Amber
    "Wind": "#4A8E3C",  # Forest Green
    "Hydro": "#4A90E2",  # Light Blue
    "Battery": "#6366F1",  # Indigo
    "Gas": "#D2691E",  # Chocolate
    "Coal": "#2C2C2C",  # Very Dark Gray
    "Other": "#9CA3AF",  # Gray
}

# Chart style configuration
CHART_STYLE = {
    "figure.facecolor": BRAND_COLORS["background"],
    "axes.facecolor": BRAND_COLORS["background"],
    "axes.edgecolor": BRAND_COLORS["grid"],
    "axes.labelcolor": BRAND_COLORS["text"],
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": BRAND_COLORS["grid"],
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "xtick.color": BRAND_COLORS["text"],
    "ytick.color": BRAND_COLORS["text"],
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["DM Sans", "Segoe UI", "Arial", "sans-serif"],
}

# Logo URL
LOGO_URL = "https://platform.openelectricity.org.au/oe_logo_full.png"
_logo_cache = None


def set_openelectricity_style():
    """Apply OpenElectricity styling to matplotlib/seaborn charts."""
    # Set seaborn style first
    sns.set_style("whitegrid", CHART_STYLE)

    # Apply matplotlib rcParams
    for key, value in CHART_STYLE.items():
        plt.rcParams[key] = value


def get_fueltech_color(fueltech: str) -> str:
    """
    Get the color for a specific fuel technology.

    Args:
        fueltech: The fuel technology identifier

    Returns:
        Hex color code for the fuel technology
    """
    # Convert to lowercase for matching
    fueltech_lower = fueltech.lower().replace(" ", "_")

    # Try exact match first
    if fueltech_lower in FUELTECH_COLORS:
        return FUELTECH_COLORS[fueltech_lower]

    # Try group match
    for group_name, color in FUELTECH_GROUPS.items():
        if group_name.lower() in fueltech_lower:
            return color

    # Default to gray if not found
    return BRAND_COLORS["text_light"]


def get_fueltech_palette(fueltechs: list) -> list:
    """
    Get a color palette for a list of fuel technologies.

    Args:
        fueltechs: List of fuel technology identifiers

    Returns:
        List of hex color codes
    """
    return [get_fueltech_color(ft) for ft in fueltechs]


def download_logo() -> Image.Image | None:
    """Download and cache the OpenElectricity logo."""
    global _logo_cache

    if _logo_cache is not None:
        return _logo_cache

    try:
        with urllib.request.urlopen(LOGO_URL) as response:
            logo_data = response.read()
            _logo_cache = Image.open(io.BytesIO(logo_data))
            return _logo_cache
    except Exception as e:
        print(f"Warning: Could not download logo: {e}")
        return None


def add_watermark(ax: Axes, position: tuple[float, float] = (0.98, 0.02), size: float = 0.15, alpha: float = 0.2) -> None:
    """
    Add OpenElectricity logo watermark to a matplotlib axes.

    Args:
        ax: Matplotlib axes to add watermark to
        position: (x, y) position in axes coordinates (0-1)
        size: Size of logo as fraction of figure width (default 0.15)
        alpha: Transparency of logo (0-1, default 0.2 for subtle appearance)
    """
    logo = download_logo()
    if logo is None:
        return

    # Get figure dimensions
    fig = ax.get_figure()
    fig_width_inch = fig.get_figwidth()
    fig_height_inch = fig.get_figheight()

    # Calculate logo size maintaining aspect ratio
    logo_width, logo_height = logo.size
    aspect_ratio = logo_height / logo_width

    # Logo width as fraction of figure width
    logo_width_fig = size
    logo_height_fig = logo_width_fig * aspect_ratio * (fig_width_inch / fig_height_inch)

    # Position calculation - ensure logo fits within figure
    x_pos = position[0] - logo_width_fig
    y_pos = position[1]

    # Make sure logo doesn't go outside figure bounds
    x_pos = max(0, min(x_pos, 1 - logo_width_fig))
    y_pos = max(0, min(y_pos, 1 - logo_height_fig))

    # Create an axes for the logo
    logo_ax = fig.add_axes(
        [x_pos, y_pos, logo_width_fig, logo_height_fig],
        frameon=False,
        zorder=1,  # Put behind main content
    )

    # Display logo with transparency
    logo_ax.imshow(logo, alpha=alpha, interpolation="lanczos")
    logo_ax.axis("off")
    logo_ax.set_xticks([])
    logo_ax.set_yticks([])


def format_chart(
    ax: Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    add_logo: bool = True,
    logo_position: tuple[float, float] = (0.98, 0.02),
    logo_size: float = 0.15,
    logo_alpha: float = 0.2,
) -> None:
    """
    Apply OpenElectricity formatting to a chart.

    Args:
        ax: Matplotlib axes to format
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        add_logo: Whether to add the OpenElectricity logo watermark
        logo_position: Position of logo in axes coordinates
        logo_size: Size of logo as fraction of figure width (default 0.15)
        logo_alpha: Transparency of logo (default 0.3)
    """
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color=BRAND_COLORS["text"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=BRAND_COLORS["text"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=BRAND_COLORS["text"])

    # Format grid
    ax.grid(True, color=BRAND_COLORS["grid"], linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BRAND_COLORS["grid"])
    ax.spines["bottom"].set_color(BRAND_COLORS["grid"])

    # Add watermark if requested
    if add_logo:
        add_watermark(ax, position=logo_position, size=logo_size, alpha=logo_alpha)


def create_styled_figure(figsize: tuple[float, float] = (12, 6), dpi: int = 100) -> tuple[Figure, Axes]:
    """
    Create a figure with OpenElectricity styling.

    Args:
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for the figure

    Returns:
        Tuple of (figure, axes)
    """
    set_openelectricity_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BRAND_COLORS["background"])
    ax.set_facecolor(BRAND_COLORS["background"])
    return fig, ax


# Export convenience mappings
def get_color_map() -> dict[str, str]:
    """Get the complete fuel technology color mapping."""
    return FUELTECH_COLORS.copy()


def get_brand_colors() -> dict[str, str]:
    """Get the OpenElectricity brand colors."""
    return BRAND_COLORS.copy()
