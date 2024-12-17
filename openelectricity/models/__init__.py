"""
OpenElectricity API Models

This package contains Pydantic models for the OpenElectricity API responses.
"""

from openelectricity.models.networks import Network, NetworkList

__all__ = ["Network", "NetworkList"]
