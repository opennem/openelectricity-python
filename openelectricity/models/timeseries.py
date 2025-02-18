"""
Time series models for the OpenElectricity API.

This module contains models for time series data responses.
"""

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, Field, RootModel

from openelectricity.models.base import APIResponse
from openelectricity.types import DataInterval, NetworkCode


class TimeSeriesDataPoint(RootModel):
    """Individual data point in a time series."""

    root: tuple[datetime, float]

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp from the data point."""
        return self.root[0]

    @property
    def value(self) -> float:
        """Get the value from the data point."""
        return self.root[1]


class TimeSeriesColumns(BaseModel):
    """Column metadata for time series results."""

    unit_code: str


class TimeSeriesResult(BaseModel):
    """Individual time series result set."""

    name: str
    date_start: datetime
    date_end: datetime
    columns: TimeSeriesColumns
    data: list[TimeSeriesDataPoint]


class NetworkTimeSeries(BaseModel):
    """Network time series data point."""

    network_code: NetworkCode
    metric: str
    unit: str
    interval: DataInterval
    start: datetime
    end: datetime
    groupings: list[str] = Field(default_factory=list)
    results: list[TimeSeriesResult]
    network_timezone_offset: str


class TimeSeriesResponse(APIResponse[NetworkTimeSeries]):
    """Response model for time series data."""

    data: Sequence[NetworkTimeSeries]
