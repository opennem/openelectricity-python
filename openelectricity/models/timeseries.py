"""
Time series models for the OpenElectricity API.

This module contains models for time series data responses.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

from pydantic import BaseModel, Field, RootModel, model_validator
from typing_extensions import Self

from openelectricity.models.base import APIResponse
from openelectricity.types import DataInterval, NetworkCode


class _AliasedModel(BaseModel):
    """Keeps each (current, deprecated) field pair in ``_alias_pairs`` holding the same value.

    Copies go straight into ``__dict__`` so they aren't marked as set and reading the
    deprecated field here doesn't warn.
    """

    _alias_pairs: ClassVar[tuple[tuple[str, str], ...]] = ()

    def _partners(self, name: str) -> list[str]:
        return [b if name == a else a for a, b in self._alias_pairs if name in (a, b)]

    def _fill_aliases(self) -> None:
        values = vars(self)
        for current, deprecated in self._alias_pairs:
            if values.get(current) is None:
                values[current] = values.get(deprecated)
            elif values.get(deprecated) is None:
                values[deprecated] = values[current]

    @model_validator(mode="after")
    def _fill_aliases_after_validation(self) -> Self:
        self._fill_aliases()
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        for partner in self._partners(name):
            vars(self)[partner] = value

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        model = super().model_construct(_fields_set, **values)
        model._fill_aliases()
        return model

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        copy = super().model_copy(update=update, deep=deep)
        for name, value in (update or {}).items():
            for partner in self._partners(name):
                vars(copy)[partner] = value
        return copy

    def __setstate__(self, state: dict[Any, Any]) -> None:
        # pickles from older versions lack the newer fields
        super().__setstate__(state)
        values = vars(self)
        for name, field in type(self).model_fields.items():
            if name not in values and not field.is_required():
                values[name] = field.get_default(call_default_factory=True)
        self._fill_aliases()


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


class TimeSeriesColumns(_AliasedModel):
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

    _alias_pairs: ClassVar[tuple[tuple[str, str], ...]] = (("region", "network_region"),)

    def _record_columns(self) -> dict[str, Any]:
        """Non-empty columns, minus alias copies of a key the API sent."""
        values = {k: v for k, v in vars(self).items() if v is not None}
        sent = self.model_fields_set
        for current, deprecated in self._alias_pairs:
            for name, partner in ((current, deprecated), (deprecated, current)):
                if partner in sent and name not in sent:
                    values.pop(name, None)
        return values


class TimeSeriesResult(BaseModel):
    """Individual time series result set."""

    name: str
    date_start: datetime
    date_end: datetime
    columns: TimeSeriesColumns
    data: list[TimeSeriesDataPoint]


class NetworkTimeSeries(_AliasedModel):
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

    _alias_pairs: ClassVar[tuple[tuple[str, str], ...]] = (("date_start", "start"), ("date_end", "end"))

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
        records_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}

        for series in self.data:
            # Process each result group
            for result in series.results:
                # Get grouping information
                groupings = result.columns._record_columns()

                # Process each data point
                for point in result.data:
                    interval = self._create_network_date(point.timestamp, series.network_timezone_offset)

                    # One record per interval and grouping, with a column per metric
                    record_key = (interval, *sorted(groupings.items()))
                    record = records_by_key.get(record_key)
                    if record is None:
                        record = {"interval": interval, **groupings}
                        records_by_key[record_key] = record
                        records.append(record)
                    record[series.metric] = point.value

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
