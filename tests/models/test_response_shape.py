"""
Tests against the response shape the live API returns (prod 4.5.16, dev 4.5.17):
``date_start`` / ``date_end`` on each metric block, ``columns.region`` and
network-local timestamps with an offset. Also covers the deprecated names
(``start`` / ``end`` / ``columns.network_region``) and the older UTC response shape.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openelectricity.models.timeseries import NetworkTimeSeries, TimeSeriesResponse

AEST = timezone(timedelta(hours=10))


def _block(metric: str, unit: str, values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "network_code": "NEM",
        "metric": metric,
        "unit": unit,
        "interval": "1h",
        "date_start": "2026-10-05T00:00:00+10:00",
        "date_end": "2026-10-05T01:00:00+10:00",
        "groupings": [],
        "network_timezone_offset": "+10:00",
        "results": [
            {
                "name": f"{metric}_{region}",
                "date_start": "2026-10-05T00:00:00+10:00",
                "date_end": "2026-10-05T01:00:00+10:00",
                "columns": {"region": region},
                "data": [
                    ["2026-10-05T00:00:00+10:00", points[0]],
                    ["2026-10-05T01:00:00+10:00", points[1]],
                ],
            }
            for region, points in values.items()
        ],
    }


@pytest.fixture
def market_response() -> dict[str, Any]:
    """Trimmed from a live prod 4.5.16 /v4/market/network/NEM response (price + demand by region)."""
    return {
        "version": "4.5.16",
        "created_at": "2026-10-06T13:07:20+11:00",
        "success": True,
        "data": [
            _block("price", "$/MWh", {"NSW1": [144.510833, 150.6325], "VIC1": [98.2, 101.4]}),
            _block("demand", "MW", {"NSW1": [6685.229167, 6420.1], "VIC1": [4511.0, 4380.7]}),
        ],
    }


@pytest.fixture
def legacy_response() -> dict[str, Any]:
    """Older shape: ``start`` / ``end``, ``columns.network_region`` and UTC timestamps."""
    return {
        "version": "4.0.3.dev0",
        "created_at": "2025-02-18T07:27:28+11:00",
        "success": True,
        "data": [
            {
                "network_code": "NEM",
                "metric": "price",
                "unit": "$/MWh",
                "interval": "1h",
                "start": "2025-02-13T00:00:00",
                "end": "2025-02-14T00:00:00",
                "groupings": ["network_region"],
                "results": [
                    {
                        "name": "price_NSW1",
                        "date_start": "2025-02-13T00:00:00",
                        "date_end": "2025-02-14T00:00:00",
                        "columns": {"network_region": "NSW1"},
                        "data": [["2025-02-12T13:00:00Z", 90.5]],
                    }
                ],
                "network_timezone_offset": "+10:00",
            }
        ],
    }


def test_live_shape_parses_date_start_and_region(market_response: dict[str, Any]) -> None:
    response = TimeSeriesResponse.model_validate(market_response)
    series = response.data[0]

    assert series.date_start == datetime(2026, 10, 5, tzinfo=AEST)
    assert series.date_end == datetime(2026, 10, 5, 1, tzinfo=AEST)
    assert series.date_range == (series.date_start, series.date_end)
    assert series.results[0].columns.region == "NSW1"


def test_to_records_keeps_network_wall_clock_and_instant(market_response: dict[str, Any]) -> None:
    """00:00+10:00 must come out as 00:00 network time, not 10:00."""
    response = TimeSeriesResponse.model_validate(market_response)
    point = response.data[0].results[0].data[0]
    record = response.to_records()[0]

    assert record["interval"] == datetime(2026, 10, 5, 0, 0)
    assert record["interval"].replace(tzinfo=AEST) == point.timestamp
    assert record["region"] == "NSW1"
    assert "network_region" not in record


def test_to_pandas_has_region_and_local_intervals(market_response: dict[str, Any]) -> None:
    pd = pytest.importorskip("pandas")
    df = TimeSeriesResponse.model_validate(market_response).to_pandas()

    assert set(df["region"]) == {"NSW1", "VIC1"}
    assert df["interval"].min() == pd.Timestamp("2026-10-05 00:00:00")
    assert df["interval"].max() == pd.Timestamp("2026-10-05 01:00:00")


def test_deprecated_names_mirror_real_fields(market_response: dict[str, Any]) -> None:
    series = TimeSeriesResponse.model_validate(market_response).data[0]

    with pytest.warns(DeprecationWarning, match="date_start"):
        assert series.start == series.date_start
    with pytest.warns(DeprecationWarning, match="date_end"):
        assert series.end == series.date_end
    with pytest.warns(DeprecationWarning, match="region"):
        assert series.results[0].columns.network_region == "NSW1"


def test_legacy_shape_still_works(legacy_response: dict[str, Any]) -> None:
    """Old keys keep parsing, fill the new fields, and records keep their old columns and times."""
    response = TimeSeriesResponse.model_validate(legacy_response)
    series = response.data[0]
    columns = series.results[0].columns

    assert series.date_start == datetime(2025, 2, 13)
    assert series.date_end == datetime(2025, 2, 14)
    assert columns.region == "NSW1"
    with pytest.warns(DeprecationWarning):
        assert series.start == datetime(2025, 2, 13)
    with pytest.warns(DeprecationWarning):
        assert columns.network_region == "NSW1"

    record = response.to_records()[0]
    assert record["interval"] == datetime(2025, 2, 12, 23, 0)
    assert record["network_region"] == "NSW1"
    assert "region" not in record


def test_constructing_with_deprecated_names_still_works() -> None:
    series = NetworkTimeSeries(
        network_code="NEM",
        metric="price",
        unit="$/MWh",
        interval="1h",
        start=datetime(2026, 10, 5),
        end=datetime(2026, 10, 6),
        results=[],
        network_timezone_offset="+10:00",
    )

    assert series.date_start == datetime(2026, 10, 5)
    assert series.date_end == datetime(2026, 10, 6)
    assert series.model_dump()["start"] == datetime(2026, 10, 5)


@pytest.fixture
def network_data_response() -> dict[str, Any]:
    """Live /v4/data/network shape for the README example (power + energy by fueltech_group)."""

    def block(metric: str, unit: str, values: dict[str, float]) -> dict[str, Any]:
        return {
            "network_code": "NEM",
            "metric": metric,
            "unit": unit,
            "interval": "1d",
            "date_start": "2026-10-04T00:00:00+10:00",
            "date_end": "2026-10-04T00:00:00+10:00",
            "groupings": ["fueltech_group"],
            "network_timezone_offset": "+10:00",
            "results": [
                {
                    "name": f"{metric}_{group}",
                    "date_start": "2026-10-04T00:00:00+10:00",
                    "date_end": "2026-10-04T00:00:00+10:00",
                    "columns": {"fueltech_group": group},
                    "data": [["2026-10-04T00:00:00+10:00", value]],
                }
                for group, value in values.items()
            ],
        }

    return {
        "version": "4.5.16",
        "created_at": "2026-10-06T13:07:24+11:00",
        "success": True,
        "data": [
            block("power", "MW", {"coal": 11000.0, "solar": 6000.0}),
            block("energy", "MWh", {"coal": 264000.0, "solar": 72000.0}),
        ],
    }


def test_readme_access_patterns(network_data_response: dict[str, Any]) -> None:
    """The README / examples/basic.py access patterns keep working on the live shape."""
    response = TimeSeriesResponse.model_validate(network_data_response)

    for series in response.data:
        start, end = series.date_range
        assert start is not None and end is not None
        assert series.metric in {"power", "energy"}
        for result in series.results:
            assert result.columns.fueltech_group in {"coal", "solar"}
            for point in result.data:
                assert point.timestamp == datetime(2026, 10, 4, tzinfo=AEST)
                assert point.value is not None

    assert response.get_metric_units() == {"power": "MW", "energy": "MWh"}

    pd = pytest.importorskip("pandas")
    df = response.to_pandas()
    by_group = df.groupby("fueltech_group").agg({"energy": "sum", "power": "mean"})
    assert by_group.loc["coal", "energy"] == 264000.0
    assert by_group.loc["solar", "power"] == 6000.0
    assert (df["interval"] == pd.Timestamp("2026-10-04 00:00:00")).all()


def test_to_records_merges_metrics_per_interval_and_region(market_response: dict[str, Any]) -> None:
    records = TimeSeriesResponse.model_validate(market_response).to_records()

    assert len(records) == 4  # 2 regions x 2 intervals, price and demand on the same row
    nsw = next(r for r in records if r["region"] == "NSW1" and r["interval"] == datetime(2026, 10, 5))
    assert nsw["price"] == 144.510833
    assert nsw["demand"] == 6685.229167
