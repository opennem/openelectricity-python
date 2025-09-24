"""
OpenElectricity Python SDK

This package provides a Python client for interacting with the OpenElectricity API.
"""

from openelectricity.client import AsyncOEClient, OEClient
from openelectricity.types import (
    DataInterval,
    DataMetric,
    DataPrimaryGrouping,
    DataSecondaryGrouping,
    FueltechGroupType,
    MarketMetric,
    NetworkCode,
    UnitFueltechType,
    UnitStatusType,
)

__name__ = "openelectricity"

__version__ = "0.9.2"

__all__ = [
    "OEClient",
    "AsyncOEClient",
    "DataMetric",
    "UnitFueltechType",
    "UnitStatusType",
    "MarketMetric",
    "NetworkCode",
    "DataInterval",
    "DataPrimaryGrouping",
    "DataSecondaryGrouping",
    "FueltechGroupType",
    "UnitFueltechType",
]

# Optional imports for styling (won't fail if dependencies are missing)
# We don't actually import the module here, just expose it conditionally
try:
    import openelectricity.styles  # noqa: F401

    __all__.append("styles")
except ImportError:
    pass  # Styling module requires matplotlib/seaborn which are optional
