"""
Time series models for the OpenElectricity API.

This module contains models for time series data responses.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field, RootModel, model_validator

from openelectricity.models.base import APIResponse
from openelectricity.types import DataInterval, NetworkCode


def _mirror_aliases(model: BaseModel, pairs: tuple[tuple[str, str], ...]) -> None:
    """Fill whichever of each (current, deprecated) field pair is empty from the other.

    Writes to ``__dict__`` so the copy isn't marked as set (``to_records`` only emits keys the
    API sent) and reading the deprecated field here doesn't warn.
    """
    values = model.__dict__
    for current, deprecated in pairs:
        if values[current] is None:
            values[current] = values[deprecated]
        elif values[deprecated] is None:
            values[deprecated] = values[current]


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
    """Column metadata for time series results.

    Populated according to the groupings used in the request: ``region``
    (``primary_grouping="network_region"``), ``fueltech`` / ``fueltech_group`` /
    ``renewable`` / ``status`` (``secondary_grouping``) and ``unit_code``
    (facility data). ``network_region`` is a deprecated alias of ``region``.
    """

    unit_code: str | None = None
    region: str | None = None
    fueltech: str | None = None
    fueltech_group: str | None = None
    renewable: bool | None = None
    status: str | None = None
    network_region: str | None = Field(default=None, deprecated="use `region`, the key the API returns")

    @model_validator(mode="after")
    def _sync_region(self) -> "TimeSeriesColumns":
        _mirror_aliases(self, (("region", "network_region"),))
        return self


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
    date_start: datetime | None = None
    date_end: datetime | None = None
    start: datetime | None = Field(default=None, deprecated="use `date_start`, the key the API returns")
    end: datetime | None = Field(default=None, deprecated="use `date_end`, the key the API returns")
    groupings: list[str] = Field(default_factory=list)
    results: list[TimeSeriesResult]
    network_timezone_offset: str
    forecast_run_time: datetime | None = None  # issue time of the newest forecast run, forecast metrics only

    @model_validator(mode="after")
    def _sync_dates(self) -> "NetworkTimeSeries":
        _mirror_aliases(self, (("date_start", "start"), ("date_end", "end")))
        return self

    @property
    def date_range(self) -> tuple[datetime | None, datetime | None]:
        """Get the date range from the results if not explicitly set."""
        if self.date_start is not None and self.date_end is not None:
            return self.date_start, self.date_end

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
        Convert a timestamp to naive network-local time.

        Args:
            timestamp: The timestamp. The API sends network-local times with an offset
                (e.g. ``+10:00``); naive timestamps are treated as UTC.
            timezone_offset: The network timezone offset string (e.g., "+10:00")

        Returns:
            A naive datetime holding the network-local wall clock time
        """
        if not timezone_offset:
            return timestamp

        # Parse the timezone offset
        sign = 1 if timezone_offset.startswith("+") else -1
        hours, minutes = map(int, timezone_offset[1:].split(":"))
        network_tz = timezone(timedelta(minutes=(hours * 60 + minutes) * sign))

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        return timestamp.astimezone(network_tz).replace(tzinfo=None)

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
                # Get grouping information, only the column keys the API sent
                groupings = {
                    k: v for k, v in result.columns.model_dump(exclude_unset=True).items() if v is not None and k != "unit_code"
                }

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
