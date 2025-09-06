"""
Time series models for the OpenElectricity API.

This module contains models for time series data responses.
"""

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
import datetime as dt
from typing import Any, Optional, TYPE_CHECKING

import warnings
from pydantic import BaseModel, Field, RootModel, ValidationError
from pydantic_core import ErrorDetails

from openelectricity.models.base import APIResponse
from openelectricity.types import DataInterval, NetworkCode

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pyspark.sql import DataFrame


def handle_validation_errors(e: ValidationError) -> None:
    """
    Convert validation errors to warnings instead of failing.
    
    Based on Pydantic's error handling documentation:
    https://docs.pydantic.dev/latest/errors/errors/
    """
    for error in e.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        warnings.warn(
            f"Validation warning for {field_path}: {error['msg']} "
            f"(value: {error.get('input', 'N/A')})",
            UserWarning,
            stacklevel=3
        )


def filter_problematic_fields(obj, errors):
    """
    Filter out fields that are causing validation errors to allow partial data parsing.
    """
    if not isinstance(obj, dict):
        return obj
    
    # Get all problematic field paths
    problematic_paths = set()
    for error in errors:
        path = error["loc"]
        if path:
            problematic_paths.add(tuple(path))
    
    # Create a filtered copy
    filtered_obj = obj.copy()
    
    # Remove problematic fields
    for path in problematic_paths:
        current = filtered_obj
        for i, key in enumerate(path[:-1]):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                break
        else:
            # Remove the problematic field
            if isinstance(current, dict) and path[-1] in current:
                del current[path[-1]]
    
    return filtered_obj


def fix_none_values_in_data(obj):
    """
    Recursively preserve None values in data arrays.
    Let users decide how to handle missing data rather than guessing.
    """
    if isinstance(obj, dict):
        return {k: fix_none_values_in_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_none_values_in_data(item) for item in obj]
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        # This might be a time series data point tuple - keep None as-is
        return obj
    else:
        return obj


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

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Override model_validate to handle validation errors gracefully."""
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as e:
            # Convert validation errors to warnings
            handle_validation_errors(e)
            # Try to fix None values in data arrays first
            try:
                fixed_obj = fix_none_values_in_data(obj)
                return super().model_validate(fixed_obj, *args, **kwargs)
            except Exception:
                # If fixing None values doesn't work, try filtering problematic fields
                try:
                    filtered_obj = filter_problematic_fields(obj, e.errors())
                    return super().model_validate(filtered_obj, *args, **kwargs)
                except Exception:
                    # If even the filtered validation fails, re-raise the original error
                    raise e


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

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Override model_validate to handle validation errors gracefully."""
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as e:
            # Convert validation errors to warnings
            handle_validation_errors(e)
            # Try to fix None values in data arrays first
            try:
                fixed_obj = fix_none_values_in_data(obj)
                return super().model_validate(fixed_obj, *args, **kwargs)
            except Exception:
                # If fixing None values doesn't work, try filtering problematic fields
                try:
                    filtered_obj = filter_problematic_fields(obj, e.errors())
                    return super().model_validate(filtered_obj, *args, **kwargs)
                except Exception:
                    # If even the filtered validation fails, re-raise the original error
                    raise e

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

    data: Sequence[NetworkTimeSeries] = Field(default_factory=list)

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Override model_validate to handle validation errors gracefully."""
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as e:
            # Convert validation errors to warnings
            handle_validation_errors(e)
            # Try to fix None values in data arrays first
            try:
                fixed_obj = fix_none_values_in_data(obj)
                return super().model_validate(fixed_obj, *args, **kwargs)
            except Exception:
                # If fixing None values doesn't work, try filtering problematic fields
                try:
                    filtered_obj = filter_problematic_fields(obj, e.errors())
                    return super().model_validate(filtered_obj, *args, **kwargs)
                except Exception:
                    # If even the filtered validation fails, re-raise the original error
                    raise e



    def _create_network_date(self, timestamp: datetime, timezone_offset: str) -> datetime:
        """
        Create a datetime with the correct network timezone.

        Args:
            timestamp: The timestamp (may already have timezone info)
            timezone_offset: The timezone offset string (e.g., "+10:00")

        Returns:
            A datetime in the network timezone
        """
        if not timezone_offset:
            return timestamp

        # Parse the timezone offset
        sign = 1 if timezone_offset.startswith("+") else -1
        hours, minutes = map(int, timezone_offset[1:].split(":"))
        offset_minutes = (hours * 60 + minutes) * sign

        # Create target timezone
        from datetime import timezone
        target_tz = timezone(timedelta(minutes=offset_minutes))

        # If timestamp already has timezone info, check if it matches target
        if timestamp.tzinfo is not None:
            # Convert to target timezone if different
            if timestamp.tzinfo != target_tz:
                return timestamp.astimezone(target_tz)
            # Already in correct timezone, return as-is
            return timestamp
        else:
            # No timezone info, assume UTC and convert to target
            utc_timestamp = timestamp.replace(tzinfo=timezone.utc)
            return utc_timestamp.astimezone(target_tz)

    def to_records(self) -> list[dict[str, Any]]:
        """
        Convert time series data into a list of records suitable for data analysis.

        Returns:
            List of dictionaries, each representing a row in the resulting table
        """
        if not self.data:
            return []

        # Use a dictionary for O(1) lookups instead of O(n) list searches
        records_dict: dict[tuple, dict[str, Any]] = {}
        
        # Pre-compile regex for better performance
        # Updated pattern to match unit codes like BW01, BW02, etc. without requiring a pipe
        region_regex = re.compile(r'_([A-Z]+\d+)$')

        for series in self.data:
            # Process each result group
            for result in series.results:
                # Get grouping information - cache the dict comprehension
                groupings = {k: v for k, v in result.columns.__dict__.items() if v is not None and k != "unit_code"}
                
                # Extract network_region from result.name if not available in columns
                if "network_region" not in groupings or groupings.get("network_region") is None:
                    # Use pre-compiled regex for better performance
                    region_match = region_regex.search(result.name)
                    if region_match:
                        groupings["network_region"] = region_match.group(1)
                    else:
                        # Fallback: try to extract the last part after underscore
                        name_parts = result.name.split('_')
                        if len(name_parts) > 1:
                            # Take the last part as the region
                            # This handles cases like "market_value_BW01" -> "BW01"
                            region_part = name_parts[-1]
                            if region_part and region_part not in ['value', 'power', 'energy', 'emissions']:
                                groupings["network_region"] = region_part

                # Create a frozen set of groupings for faster comparison
                groupings_frozen = frozenset(groupings.items())
                
                # Process each data point
                for point in result.data:
                    # Create a simple tuple key for O(1) dictionary lookup
                    # Use timestamp directly instead of isoformat() for better performance
                    record_key = (point.timestamp, groupings_frozen)
                    
                    if record_key in records_dict:
                        # Update existing record with this metric
                        records_dict[record_key][series.metric] = point.value
                    else:
                        # Create new record
                        record = {
                            "interval": self._create_network_date(point.timestamp, series.network_timezone_offset),
                            **groupings,
                            series.metric: point.value,
                        }
                        records_dict[record_key] = record

        # Convert back to list
        return list(records_dict.values())

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

    def to_pyspark(self, spark_session=None, use_batching: bool = False, batch_size: int = 10000) -> "Optional['DataFrame']":  # noqa: F821
        """
        Convert time series data into a PySpark DataFrame with optimized performance.

        Args:
            spark_session: Optional PySpark session. If not provided, will try to create one.
            use_batching: Whether to use batched processing for large datasets (default: False)
            batch_size: Number of records per batch when batching is enabled (default: 10000)

        Returns:
            A PySpark DataFrame containing the time series data, or None if PySpark is not available
        """
        try:
            from openelectricity.spark_utils import get_spark_session, detect_timeseries_schema, clean_timeseries_records_fast, create_timeseries_dataframe_batched
            import logging
            
            logger = logging.getLogger(__name__)
            
            # Get records (this is already optimized)
            records = self.to_records()
            if not records:
                return None
            
            # Get or create Spark session
            if spark_session is None:
                spark_session = get_spark_session()
            
            # Choose processing method based on dataset size and user preference
            if use_batching and len(records) > batch_size:
                logger.debug(f"Using batched processing for {len(records)} records (batch size: {batch_size})")
                return create_timeseries_dataframe_batched(records, spark_session, batch_size)
            else:
                # Use automatically detected schema for maximum performance (eliminates expensive schema inference)
                timeseries_schema = detect_timeseries_schema(records)
                
                # Fast data cleaning with optimized type conversion
                cleaned_records = clean_timeseries_records_fast(records)
                
                logger.debug(f"Using auto-detected schema with {len(timeseries_schema.fields)} fields")
                logger.debug(f"Processed {len(cleaned_records)} records with fast cleaning")
                
                # Create DataFrame with pre-defined schema for maximum performance
                return spark_session.createDataFrame(cleaned_records, schema=timeseries_schema)
                
        except ImportError:
            # Log warning but don't raise error to maintain compatibility
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("PySpark not available. Install with: uv add 'openelectricity[analysis]'")
            return None
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error converting to PySpark DataFrame: {e}")
            return None
