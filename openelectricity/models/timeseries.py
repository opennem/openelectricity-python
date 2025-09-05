"""
Time series models for the OpenElectricity API.

This module contains models for time series data responses.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, RootModel

from openelectricity.models.base import APIResponse
from openelectricity.types import DataInterval, NetworkCode


class TimeSeriesDataPoint(RootModel):
    """Individual data point in a time series."""

    root: tuple[datetime, float | None]

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp from the data point."""
        return self.root[0]

    @property
    def value(self) -> float | None:
        """Get the value from the data point."""
        return self.root[1]


class TimeSeriesColumns(BaseModel):
    """Column metadata for time series results."""

    unit_code: str | None = None
    fueltech_group: str | None = None
    network_region: str | None = None


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
    start: datetime | None = None
    end: datetime | None = None
    groupings: list[str] = Field(default_factory=list)
    results: list[TimeSeriesResult]
    network_timezone_offset: str

    @property
    def date_range(self) -> tuple[datetime | None, datetime | None]:
        """Get the date range from the results if not explicitly set."""
        if self.start is not None and self.end is not None:
            return self.start, self.end

        # Try to get dates from results
        if not self.results:
            return None, None

        start_dates = [r.date_start for r in self.results if r.date_start is not None]
        end_dates = [r.date_end for r in self.results if r.date_end is not None]

        if not start_dates or not end_dates:
            return None, None

        return min(start_dates), max(end_dates)


class TimeSeriesResponse(APIResponse[NetworkTimeSeries]):
    """Response model for time series data."""

    data: Sequence[NetworkTimeSeries]

    def _create_network_date(self, timestamp: datetime, timezone_offset: str) -> datetime:
        """
        Create a datetime with the correct network timezone.

        Args:
            timestamp: The UTC timestamp
            timezone_offset: The timezone offset string (e.g., "+10:00")

        Returns:
            A datetime adjusted to the network timezone
        """
        if not timezone_offset:
            return timestamp

        # Parse the timezone offset
        sign = 1 if timezone_offset.startswith("+") else -1
        hours, minutes = map(int, timezone_offset[1:].split(":"))
        offset_minutes = (hours * 60 + minutes) * sign

        # Adjust the timestamp
        return timestamp.replace(tzinfo=None) + timedelta(minutes=offset_minutes)

    def to_records(self) -> list[dict[str, Any]]:
        """
        Convert time series data into a list of records suitable for data analysis.

        Returns:
            List of dictionaries, each representing a row in the resulting table
        """
        if not self.data:
            return []

        records: list[dict[str, Any]] = []

        for series in self.data:
            # Process each result group
            for result in series.results:
                # Get grouping information
                groupings = {k: v for k, v in result.columns.__dict__.items() if v is not None and k != "unit_code"}

                # Process each data point
                for point in result.data:
                    # Create or update record
                    record_key = (point.timestamp.isoformat(), *sorted(groupings.items()))
                    existing_record = next(
                        (r for r in records if (r["interval"].isoformat(), *sorted((k, r[k]) for k in groupings)) == record_key),
                        None,
                    )

                    if existing_record:
                        # Update existing record with this metric
                        existing_record[series.metric] = point.value
                    else:
                        # Create new record
                        record = {
                            "interval": self._create_network_date(point.timestamp, series.network_timezone_offset),
                            **groupings,
                            series.metric: point.value,
                        }
                        records.append(record)

        return records

    def get_metric_units(self) -> dict[str, str]:
        """
        Get a mapping of metrics to their units.

        Returns:
            Dictionary mapping metric names to their units
        """
        return {series.metric: series.unit for series in self.data}

    def to_polars(self) -> "pl.DataFrame":  # noqa: F821
        """
        Convert time series data into a Polars DataFrame.

        Returns:
            A Polars DataFrame containing the time series data
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "Polars is required for DataFrame conversion. Install it with: uv add 'openelectricity[analysis]'"
            ) from None

        return pl.DataFrame(self.to_records())

    def to_pandas(self) -> "pd.DataFrame":  # noqa: F821
        """
        Convert time series data into a Pandas DataFrame.

        Returns:
            A Pandas DataFrame containing the time series data
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required for DataFrame conversion. Install it with: uv add 'openelectricity[analysis]'"
            ) from None

        return pd.DataFrame(self.to_records())
