"""
Type definitions for the OpenElectricity API.

This module contains type definitions, enums, and type aliases used across the API.
Matches the TypeScript definitions from the official client.
"""

from enum import Enum
from typing import Any, Literal, TypeAlias

# Network and Data Types
NetworkCode = Literal["NEM", "WEM", "AU"]
DataInterval = Literal["5m", "1h", "1d", "7d", "1M", "3M", "season", "1y", "fy"]
DataPrimaryGrouping = Literal["network", "network_region"]
DataSecondaryGrouping = Literal["fueltech", "fueltech_group", "renewable"]


class Network(str, Enum):
    """Supported networks"""

    NEM = "NEM"
    WEM = "WEM"
    AU = "AU"


class DataMetric(str, Enum):
    """Data metrics available for network and facility data."""

    POWER = "power"
    ENERGY = "energy"
    EMISSIONS = "emissions"
    MARKET_VALUE = "market_value"


class MarketMetric(str, Enum):
    """Market metrics available for market data."""

    PRICE = "price"
    DEMAND = "demand"
    DEMAND_ENERGY = "demand_energy"


class MilestoneType(str, Enum):
    """Types of milestones."""

    POWER = "power"
    ENERGY = "energy"
    DEMAND = "demand"
    PRICE = "price"
    MARKET_VALUE = "market_value"
    EMISSIONS = "emissions"
    PROPORTION = "proportion"


class MilestonePeriod(str, Enum):
    """Time periods for milestone data."""

    INTERVAL = "interval"
    DAY = "day"
    WEEK = "7d"
    MONTH = "month"
    QUARTER = "quarter"
    SEASON = "season"
    YEAR = "year"
    FINANCIAL_YEAR = "financial_year"


class MilestoneAggregate(str, Enum):
    """Aggregation types for milestone data."""

    LOW = "low"
    HIGH = "high"


# Constants for validation
VALID_NETWORKS = ["NEM", "WEM", "AU"]
VALID_INTERVALS = ["5m", "1h", "1d", "7d", "1M", "3M", "season", "1y", "fy"]
VALID_PRIMARY_GROUPINGS = ["network", "network_region"]
VALID_SECONDARY_GROUPINGS = ["fueltech", "fueltech_group", "renewable"]

# Type aliases for documentation
Metric: TypeAlias = str  # Union of DataMetric and MarketMetric values
TimeSeriesResult: TypeAlias = dict[str, Any]  # Matches ITimeSeriesResult
NetworkTimeSeries: TypeAlias = dict[str, Any]  # Matches INetworkTimeSeries
