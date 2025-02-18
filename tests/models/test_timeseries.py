"""
Tests for time series models.

This module contains tests for the time series response models
using real API response examples.
"""

from datetime import UTC, datetime

import pytest

from openelectricity.models.timeseries import (
    NetworkTimeSeries,
    TimeSeriesDataPoint,
    TimeSeriesResponse,
    TimeSeriesResult,
)


@pytest.fixture
def facility_response() -> dict:
    """Example facility response fixture."""
    return {
        "version": "4.0.3.dev0",
        "created_at": "2025-02-18T07:27:28+11:00",
        "success": True,
        "data": [
            {
                "network_code": "NEM",
                "metric": "energy",
                "unit": "MWh",
                "interval": "1d",
                "start": "2025-02-13T00:00:00",
                "end": "2025-02-15T00:00:00",
                "groupings": [],
                "results": [
                    {
                        "name": "energy_BANGOWF1",
                        "date_start": "2025-02-13T00:00:00",
                        "date_end": "2025-02-15T00:00:00",
                        "columns": {"unit_code": "BANGOWF1"},
                        "data": [
                            ["2025-02-12T13:00:00Z", 931.4554],
                            ["2025-02-13T13:00:00Z", 1198.4969],
                            ["2025-02-14T13:00:00Z", 4610.3691],
                        ],
                    },
                    {
                        "name": "energy_BANGOWF2",
                        "date_start": "2025-02-13T00:00:00",
                        "date_end": "2025-02-15T00:00:00",
                        "columns": {"unit_code": "BANGOWF2"},
                        "data": [
                            ["2025-02-12T13:00:00Z", 555.5546],
                            ["2025-02-13T13:00:00Z", 790.281],
                            ["2025-02-14T13:00:00Z", 1745.9334],
                        ],
                    },
                ],
                "network_timezone_offset": "+10:00",
            },
            {
                "network_code": "NEM",
                "metric": "market_value",
                "unit": "$",
                "interval": "1d",
                "start": "2025-02-13T00:00:00",
                "end": "2025-02-15T00:00:00",
                "groupings": [],
                "results": [
                    {
                        "name": "market_value_BANGOWF1",
                        "date_start": "2025-02-13T00:00:00",
                        "date_end": "2025-02-15T00:00:00",
                        "columns": {"unit_code": "BANGOWF1"},
                        "data": [
                            ["2025-02-12T13:00:00Z", 80408.191],
                            ["2025-02-13T13:00:00Z", 127704.6],
                            ["2025-02-14T13:00:00Z", 168568.15],
                        ],
                    },
                    {
                        "name": "market_value_BANGOWF2",
                        "date_start": "2025-02-13T00:00:00",
                        "date_end": "2025-02-15T00:00:00",
                        "columns": {"unit_code": "BANGOWF2"},
                        "data": [
                            ["2025-02-12T13:00:00Z", 46632.327],
                            ["2025-02-13T13:00:00Z", 89052.421],
                            ["2025-02-14T13:00:00Z", 136116.71],
                        ],
                    },
                ],
                "network_timezone_offset": "+10:00",
            },
        ],
    }


def test_timeseries_response_parsing(facility_response):
    """Test parsing a complete time series response."""
    response = TimeSeriesResponse.model_validate(facility_response)

    assert response.version == "4.0.3.dev0"
    assert response.success is True
    assert len(response.data) == 2

    # Check first time series (energy)
    energy_series = response.data[0]
    assert energy_series.network_code == "NEM"
    assert energy_series.metric == "energy"
    assert energy_series.unit == "MWh"
    assert energy_series.interval == "1d"
    assert len(energy_series.results) == 2

    # Check second time series (market value)
    market_series = response.data[1]
    assert market_series.network_code == "NEM"
    assert market_series.metric == "market_value"
    assert market_series.unit == "$"
    assert market_series.interval == "1d"
    assert len(market_series.results) == 2


def test_network_timeseries_parsing(facility_response):
    """Test parsing individual network time series."""
    energy_series = NetworkTimeSeries.model_validate(facility_response["data"][0])

    assert energy_series.network_code == "NEM"
    assert energy_series.metric == "energy"
    assert energy_series.unit == "MWh"
    assert energy_series.interval == "1d"
    assert energy_series.network_timezone_offset == "+10:00"
    assert len(energy_series.results) == 2

    # Check start and end dates
    assert energy_series.start == datetime(2025, 2, 13, tzinfo=None)
    assert energy_series.end == datetime(2025, 2, 15, tzinfo=None)


def test_timeseries_result_parsing(facility_response):
    """Test parsing individual time series results."""
    result = TimeSeriesResult.model_validate(facility_response["data"][0]["results"][0])

    assert result.name == "energy_BANGOWF1"
    assert result.columns.unit_code == "BANGOWF1"
    assert len(result.data) == 3

    # Check first data point
    first_point = result.data[0]
    assert first_point.timestamp == datetime(2025, 2, 12, 13, tzinfo=UTC)
    assert first_point.value == 931.4554


def test_timeseries_datapoint_parsing():
    """Test parsing individual time series data points."""
    data = ["2025-02-12T13:00:00Z", 931.4554]
    point = TimeSeriesDataPoint.model_validate(data)

    assert point.timestamp == datetime(2025, 2, 12, 13, tzinfo=UTC)
    assert point.value == 931.4554


def test_invalid_network_code():
    """Test validation error for invalid network code."""
    with pytest.raises(ValueError):
        NetworkTimeSeries.model_validate(
            {
                "network_code": "INVALID",
                "metric": "energy",
                "unit": "MWh",
                "interval": "1d",
                "start": "2025-02-13T00:00:00",
                "end": "2025-02-15T00:00:00",
                "groupings": [],
                "results": [],
                "network_timezone_offset": "+10:00",
            }
        )


def test_invalid_interval():
    """Test validation error for invalid interval."""
    with pytest.raises(ValueError):
        NetworkTimeSeries.model_validate(
            {
                "network_code": "NEM",
                "metric": "energy",
                "unit": "MWh",
                "interval": "invalid",
                "start": "2025-02-13T00:00:00",
                "end": "2025-02-15T00:00:00",
                "groupings": [],
                "results": [],
                "network_timezone_offset": "+10:00",
            }
        )
