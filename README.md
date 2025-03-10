# OpenElectricity Python Client

![logo](https://platform.openelectricity.org.au/oe_logo_full.png)

> [!WARNING]
> This project and the v4 API are currently under active development.

A Python client for the [OpenElectricity](https://openelectricity.org.au) API, providing access to electricity and energy network data and metrics for Australia.

> [!NOTE]
> API key signups are currently waitlisted and will be released gradually.

To obtain an API key visit [platform.openelectricity.org.au](https://platform.openelectricity.org.au)

For documentation visit [docs.openelectricity.org.au](https://docs.openelectricity.org.au/introduction)

## Features

-   Synchronous and asynchronous API clients
-   Fully typed with comprehensive type annotations
-   Automatic request retries and error handling
-   Context manager support
-   Modern Python (3.10+) with full type annotations
-   Direct conversion to Pandas and Polars DataFrames

## Installation

```bash
# Install base package
pip install openelectricity

# or with uv (recommended)
uv add openelectricity

# Install with data analysis support (Polars/Pandas)
uv add "openelectricity[analysis]"
```

## Quick Start

First, set your API key in the environment:

```bash
# Set your API key
export OPENELECTRICITY_API_KEY=your-api-key

# Optional: Override API server (defaults to production)
export OPENELECTRICITY_API_URL=http://localhost:8000/v4
```

Then in your code:

```python
from datetime import datetime, timedelta
from openelectricity import OEClient
from openelectricity.types import DataMetric, UnitFueltechType, UnitStatusType

# Calculate date range
end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
start_date = end_date - timedelta(days=7)

# Using context manager (recommended)
with OEClient() as client:
    # Get operating solar and wind facilities
    facilities = client.get_facilities(
        network_id=["NEM"],
        status_id=[UnitStatusType.OPERATING],
        fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
    )

    # Get network data for NEM
    response = client.get_network_data(
        network_code="NEM",
        metrics=[DataMetric.POWER, DataMetric.ENERGY],
        interval="1d",
        date_start=start_date,
        date_end=end_date,
        secondary_grouping="fueltech_group",
    )

    # Print results
    for series in response.data:
        print(f"\nMetric: {series.metric}")
        print(f"Unit: {series.unit}")

        for result in series.results:
            print(f"\n  {result.name}:")
            print(f"  Fuel Tech Group: {result.columns.fueltech_group}")
            for point in result.data:
                print(f"    {point.timestamp}: {point.value:.2f} {series.unit}")
```

For async usage:

```python
from openelectricity import AsyncOEClient
import asyncio

async def main():
    async with AsyncOEClient() as client:
        # Get operating solar and wind facilities
        facilities = await client.get_facilities(
            network_id=["NEM"],
            status_id=[UnitStatusType.OPERATING],
            fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
        )

        # Get network data
        response = await client.get_network_data(
            network_code="NEM",
            metrics=[DataMetric.POWER],
            interval="1d",
            secondary_grouping="fueltech_group",
        )
        # Process response...

asyncio.run(main())
```

## Data Analysis

The client provides built-in support for converting API responses to popular data analysis formats.

### Using with Polars

```python
# Make sure you've installed with analysis extras
# uv add "openelectricity[analysis]"

from openelectricity import OEClient
from openelectricity.types import DataMetric

with OEClient() as client:
    response = client.get_network_data(
        network_code="NEM",
        metrics=[DataMetric.POWER, DataMetric.ENERGY],
        interval="1d",
        secondary_grouping="fueltech_group",
    )

    # Convert to Polars DataFrame
    df = response.to_polars()

    # Get metric units
    units = response.get_metric_units()

    # Analyze data
    energy_by_fueltech = (
        df.group_by("fueltech_group")
        .agg(
            pl.col("energy").sum().alias("total_energy_mwh"),
            pl.col("power").mean().alias("avg_power_mw"),
        )
        .sort("total_energy_mwh", descending=True)
    )
```

### Using with Pandas

```python
# Make sure you've installed with analysis extras
# uv add "openelectricity[analysis]"

from openelectricity import OEClient
from openelectricity.types import DataMetric

with OEClient() as client:
    response = client.get_network_data(
        network_code="NEM",
        metrics=[DataMetric.POWER, DataMetric.ENERGY],
        interval="1d",
        secondary_grouping="fueltech_group",
    )

    # Convert to Pandas DataFrame
    df = response.to_pandas()

    # Get metric units
    units = response.get_metric_units()

    # Analyze data
    energy_by_fueltech = (
        df.groupby("fueltech_group")
        .agg({
            "energy": "sum",
            "power": "mean",
        })
        .sort_values("energy", ascending=False)
    )
```

## Development

1. Clone the repository
2. Install development dependencies:

    ```bash
    make install
    ```

3. Run tests:

    ```bash
    make test
    ```

4. Format code:

    ```bash
    make format
    ```

5. Run linters:
    ```bash
    make lint
    ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
