#!/usr/bin/env python
"""
Example: splice AEMO's rooftop solar forecast onto rooftop solar actuals

Rooftop actuals land 30 to 60 minutes behind the 5 minute feed. The forecast
fills the gap from the last actual interval forward.

Point at another API with OPENELECTRICITY_API_URL, e.g.
OPENELECTRICITY_API_URL=https://api.oedev.org/v4 uv run examples/rooftop_forecast.py
"""

from datetime import datetime

from dotenv import load_dotenv

from openelectricity import OEClient
from openelectricity.types import DataMetric, MarketMetric, UnitFueltechType

load_dotenv()


def main() -> None:
    with OEClient() as client:
        actual = client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.POWER],
            interval="5m",
            fueltech=[UnitFueltechType.SOLAR_ROOFTOP],
        )
        actual_points = [p for p in actual.data[0].results[0].data if p.value is not None]
        if not actual_points:
            raise SystemExit("no rooftop actuals returned")

        # date_start is network-local naive; date_end defaults to the end of the latest forecast run
        last_actual = actual_points[-1].timestamp
        forecast = client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.SOLAR_ROOFTOP_FORECAST],
            interval="5m",
            date_start=last_actual.replace(tzinfo=None),
        )
        forecast_series = forecast.data[0]
        print(f"forecast run: {forecast_series.forecast_run_time}")

        # actual wins where present, forecast fills the rest
        spliced: dict[datetime, tuple[float | None, str]] = {
            p.timestamp: (p.value, "forecast") for p in forecast_series.results[0].data
        }
        spliced.update({p.timestamp: (p.value, "actual") for p in actual_points})

        for ts in sorted(spliced)[-48:]:
            value, source = spliced[ts]
            print(f"{ts.isoformat()}  {source:8}  {value}")


if __name__ == "__main__":
    main()
