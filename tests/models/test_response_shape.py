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


def _pre_change_to_records(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """to_records as shipped in 0.11.3, run on the raw response dict.

    Kept verbatim in behaviour: columns limited to the old model's fields, the network offset
    added to the timestamp, and a merge lookup that compares the raw timestamp with the shifted
    interval.
    """
    old_columns = ("fueltech", "fueltech_group", "renewable", "network_region")
    records: list[dict[str, Any]] = []
    for series in raw["data"]:
        sign = 1 if series["network_timezone_offset"].startswith("+") else -1
        hours, minutes = map(int, series["network_timezone_offset"][1:].split(":"))
        offset = timedelta(minutes=(hours * 60 + minutes) * sign)
        for result in series["results"]:
            groupings = {k: result["columns"][k] for k in old_columns if result["columns"].get(k) is not None}
            for ts, value in result["data"]:
                timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                record_key = (timestamp.isoformat(), *sorted(groupings.items()))
                existing = next(
                    (r for r in records if (r["interval"].isoformat(), *sorted((k, r[k]) for k in groupings)) == record_key),
                    None,
                )
                if existing:
                    existing[series["metric"]] = value
                else:
                    records.append({"interval": timestamp.replace(tzinfo=None) + offset, **groupings, series["metric"]: value})
    return records


def test_default_records_match_pre_change_shape(market_response: dict[str, Any]) -> None:
    """Default output keeps the 0.11.3 row shape: one row per value, in the same order.

    The only differences are the +10h fix and the additive region column.
    """
    records = TimeSeriesResponse.model_validate(market_response).to_records()
    before = _pre_change_to_records(market_response)

    assert len(records) == len(before) == 8  # 2 metrics x 2 regions x 2 intervals
    for new, old in zip(records, before, strict=True):
        assert new["interval"] == old["interval"] - timedelta(hours=10)
        assert {k: v for k, v in new.items() if k not in ("interval", "region")} == {
            k: v for k, v in old.items() if k != "interval"
        }
    assert [r["region"] for r in records[:2]] == ["NSW1", "NSW1"]


def test_default_records_equal_pre_change_on_legacy_shape(legacy_response: dict[str, Any]) -> None:
    """On the older UTC response shape the default output is exactly what 0.11.3 returned."""
    assert TimeSeriesResponse.model_validate(legacy_response).to_records() == _pre_change_to_records(legacy_response)


def test_default_dataframes_keep_one_row_per_value(market_response: dict[str, Any]) -> None:
    response = TimeSeriesResponse.model_validate(market_response)
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")

    df = response.to_pandas()
    assert len(df) == 8
    assert df["price"].notna().sum() == 4 and df["demand"].notna().sum() == 4
    assert isinstance(df, pd.DataFrame)
    assert response.to_polars().height == 8
    assert isinstance(response.to_polars(), pl.DataFrame)


def test_merge_metrics_opt_in(market_response: dict[str, Any]) -> None:
    response = TimeSeriesResponse.model_validate(market_response)
    records = response.to_records(merge_metrics=True)

    assert len(records) == 4  # 2 regions x 2 intervals, price and demand on the same row
    nsw = next(r for r in records if r["region"] == "NSW1" and r["interval"] == datetime(2026, 10, 5))
    assert nsw == {"interval": datetime(2026, 10, 5), "region": "NSW1", "price": 144.510833, "demand": 6685.229167}

    pytest.importorskip("pandas")
    assert len(response.to_pandas(merge_metrics=True)) == 4
    pytest.importorskip("polars")
    assert response.to_polars(merge_metrics=True).columns == ["interval", "region", "price", "demand"]


def test_aliases_stay_in_step_after_assignment_and_copy(market_response: dict[str, Any]) -> None:
    series = TimeSeriesResponse.model_validate(market_response).data[0]
    new_start = datetime(2026, 10, 4, tzinfo=AEST)

    series.start = new_start
    assert series.date_start == new_start
    assert series.date_range[0] == new_start

    copy = series.model_copy(update={"date_end": new_start})
    with pytest.warns(DeprecationWarning):
        assert copy.end == new_start

    constructed = NetworkTimeSeries.model_construct(start=new_start)
    assert constructed.date_start == new_start

    columns = series.results[0].columns
    columns.region = "QLD1"
    with pytest.warns(DeprecationWarning):
        assert columns.network_region == "QLD1"


def test_pickle_round_trip_and_older_pickles(market_response: dict[str, Any]) -> None:
    import pickle

    series = TimeSeriesResponse.model_validate(market_response).data[0]
    assert pickle.loads(pickle.dumps(series)) == series

    # a pickle from before date_start/date_end existed restores with them filled from start/end
    old_state = series.__getstate__()
    old_state["__dict__"] = {k: v for k, v in old_state["__dict__"].items() if k not in ("date_start", "date_end")}
    restored = NetworkTimeSeries.__new__(NetworkTimeSeries)
    restored.__setstate__(old_state)
    assert restored.date_start == series.date_start
    assert restored.date_range == (series.date_start, series.date_end)


def test_serialisation_keeps_old_keys(market_response: dict[str, Any]) -> None:
    """model_dump / JSON still carry start / end / network_region, now alongside the real keys."""
    series = TimeSeriesResponse.model_validate(market_response).data[0]
    dumped = series.model_dump()

    assert dumped["start"] == dumped["date_start"]
    assert dumped["end"] == dumped["date_end"]
    assert dumped["results"][0]["columns"]["network_region"] == "NSW1"
    assert NetworkTimeSeries.model_validate_json(series.model_dump_json()) == series


def test_to_polars_keeps_metrics_that_start_after_100_rows() -> None:
    """One row per value puts each metric's rows after the previous metric's; polars must see them all."""
    pytest.importorskip("polars")
    start = datetime(2026, 10, 5, tzinfo=AEST)
    points = [[(start + timedelta(minutes=5 * i)).isoformat(), float(i)] for i in range(150)]

    def block(metric: str) -> dict[str, Any]:
        return {
            "network_code": "NEM",
            "metric": metric,
            "unit": "MW",
            "interval": "5m",
            "network_timezone_offset": "+10:00",
            "results": [{"name": metric, "date_start": points[0][0], "date_end": points[-1][0], "columns": {}, "data": points}],
        }

    response = TimeSeriesResponse.model_validate(
        {"version": "4.5.17", "created_at": "2026-10-06T13:00:00+11:00", "data": [block("price"), block("demand")]}
    )
    df = response.to_polars()

    assert df.height == 300
    assert df["demand"].drop_nulls().len() == 150
