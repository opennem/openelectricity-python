"""
Network Models

This module contains Pydantic models for network-related API responses.
"""

from pydantic import BaseModel, Field


class NetworkRegion(BaseModel):
    """A region within a network."""

    code: str = Field(..., description="Region code")
    timezone: str = Field(..., description="Region timezone")


class Network(BaseModel):
    """A network in the OpenElectricity system."""

    code: str = Field(..., description="Network code")
    country: str = Field(..., description="Country code")
    label: str = Field(..., description="Network label/name")
    regions: list[NetworkRegion] | None = Field(None, description="List of regions in the network")
    timezone: str = Field(..., description="Network timezone")
    interval_size: int = Field(..., description="Size of data intervals in minutes")


class NetworkList(BaseModel):
    """A list of networks."""

    networks: list[Network] = Field(..., description="List of networks")
    count: int = Field(..., description="Total number of networks")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
