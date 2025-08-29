"""
OpenElectricity Python SDK

This package provides a Python client for interacting with the OpenElectricity API.
"""

from openelectricity.client import AsyncOEClient, OEClient

__name__ = "openelectricity"

__version__ = "0.7.0"

__all__ = ["OEClient", "AsyncOEClient"]

# Optional imports for styling (won't fail if dependencies are missing)
# We don't actually import the module here, just expose it conditionally
try:
    import openelectricity.styles  # noqa: F401

    __all__.append("styles")
except ImportError:
    pass  # Styling module requires matplotlib/seaborn which are optional
