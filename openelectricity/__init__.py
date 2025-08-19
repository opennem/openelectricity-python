"""
OpenElectricity Python SDK

This package provides a Python client for interacting with the OpenElectricity API.
"""

from openelectricity.client import AsyncOEClient, OEClient

__name__ = "openelectricity"

__version__ = "0.5.1"

__all__ = ["OEClient", "AsyncOEClient"]

# Optional imports for styling (won't fail if dependencies are missing)
try:
    from openelectricity import styles
    __all__.append("styles")
except ImportError:
    pass  # Styling module requires matplotlib/seaborn which are optional
