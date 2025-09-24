"""
Type definitions for the OpenElectricity API.

This module contains type definitions, enums, and type aliases used across the API.
Matches the TypeScript definitions from the official client.
"""

from enum import Enum
from typing import Any, Literal

# Network and Data Types
NetworkCode = Literal["NEM", "WEM", "AU"]
DataInterval = Literal["5m", "1h", "1d", "7d", "1M", "3M", "season", "1y", "fy"]
DataPrimaryGrouping = Literal["network", "network_region"]
DataSecondaryGrouping = Literal["fueltech", "fueltech_group", "status", "renewable"]


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
    RENEWABLE_PROPORTION = "renewable_proportion"
    STORAGE_BATTERY = "storage_battery"


class MarketMetric(str, Enum):
    """Market metrics available for market data."""

    PRICE = "price"
    DEMAND = "demand"
    DEMAND_ENERGY = "demand_energy"
    CURTAILMENT = "curtailment"
    CURTAILMENT_ENERGY = "curtailment_energy"
    CURTAILMENT_SOLAR_UTILITY = "curtailment_solar_utility"
    CURTAILMENT_SOLAR_UTILITY_ENERGY = "curtailment_solar_utility_energy"
    CURTAILMENT_WIND = "curtailment_wind"
    CURTAILMENT_WIND_ENERGY = "curtailment_wind_energy"


class UnitFueltechType(str, Enum):
    """Types of fuel technologies for units."""

    BATTERY = "battery"
    BATTERY_CHARGING = "battery_charging"
    BATTERY_DISCHARGING = "battery_discharging"
    BIOENERGY_BIOGAS = "bioenergy_biogas"
    BIOENERGY_BIOMASS = "bioenergy_biomass"
    COAL_BLACK = "coal_black"
    COAL_BROWN = "coal_brown"
    DISTILLATE = "distillate"
    GAS_CCGT = "gas_ccgt"
    GAS_OCGT = "gas_ocgt"
    GAS_RECIP = "gas_recip"
    GAS_STEAM = "gas_steam"
    GAS_WCMG = "gas_wcmg"
    HYDRO = "hydro"
    PUMPS = "pumps"
    SOLAR_ROOFTOP = "solar_rooftop"
    SOLAR_THERMAL = "solar_thermal"
    SOLAR_UTILITY = "solar_utility"
    NUCLEAR = "nuclear"
    OTHER = "other"
    SOLAR = "solar"
    WIND = "wind"
    WIND_OFFSHORE = "wind_offshore"
    IMPORTS = "imports"
    EXPORTS = "exports"
    INTERCONNECTOR = "interconnector"
    AGGREGATOR_VPP = "aggregator_vpp"
    AGGREGATOR_DR = "aggregator_dr"


class UnitStatusType(str, Enum):
    """Types of unit statuses."""

    COMMITTED = "committed"
    OPERATING = "operating"
    RETIRED = "retired"


class UnitDateSpecificity(str, Enum):
    """Date specificity for unit dates."""

    YEAR = "year"
    MONTH = "month"
    QUARTER = "quarter"
    DAY = "day"


class FueltechGroupType(str, Enum):
    """Types of fuel technology groups."""

    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    BATTERY = "battery"
    GAS = "gas"
    COAL = "coal"
    RENEWABLE = "renewable"
    FOSSIL = "fossil"
    OTHER = "other"


class OpenNEMRoles(str, Enum):
    """User roles in the OpenNEM system."""

    ADMIN = "admin"
    PRO = "pro"
    ACADEMIC = "academic"
    USER = "user"
    ANONYMOUS = "anonymous"


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
VALID_SECONDARY_GROUPINGS = ["fueltech", "fueltech_group", "status", "renewable"]

# Type aliases for documentation
Metric = str  # Union of DataMetric and MarketMetric values
TimeSeriesResult = dict[str, Any]  # Matches ITimeSeriesResult
NetworkTimeSeries = dict[str, Any]  # Matches INetworkTimeSeries
