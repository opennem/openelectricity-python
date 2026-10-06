# Changelog

## 0.12.0

Rooftop solar forecast support ([opennem#675](https://github.com/opennem/opennem/issues/675)).

### Added

- `MarketMetric.SOLAR_ROOFTOP_FORECAST` (`solar_rooftop_forecast`, MW, NEM only).
- `30m` added to `DataInterval` and `VALID_INTERVALS`, accepted on market and
  data endpoints. Responses at `30m` previously failed model validation.
- `NetworkTimeSeries.forecast_run_time` (optional `datetime`), the issue time
  of the newest forecast run used. Set only on forecast metric series, and
  `None` when the values come from history loaded before run times were
  recorded (before October 2026).
- `examples/rooftop_forecast.py` splices the forecast onto rooftop actuals.

Forecast metrics accept a `date_end` in the future. The client does no date
validation, so no client change was needed for that.

### Fixed

- `to_records()` / `to_pandas()` / `to_polars()` shifted every interval forward
  by the network offset (+10h for NEM). API timestamps already carry the
  offset (`2026-10-05T00:00:00+10:00`). `interval` is now the network-local
  wall clock time, still a naive `datetime`.
- `region` grouping values were dropped on parse because the API sends
  `columns.region`. `TimeSeriesColumns` gains `region` and `status`, and
  records include them as columns.
- `to_records()` now puts every metric for an interval and grouping on one row,
  as documented. It previously emitted one row per value. Facility records gain
  a `unit_code` column so units stay apart.

### Changed

- `NetworkTimeSeries` reads `date_start` / `date_end`, the keys the API
  returns. `start` / `end` are deprecated aliases holding the same values, and
  older responses that send `start` / `end` fill `date_start` / `date_end`.
- `TimeSeriesColumns.network_region` is a deprecated alias of `region`.
- Reading a deprecated field emits a `DeprecationWarning`. Records only include
  the column keys the API sent, so existing record columns are unchanged.

## 0.11.3

### Added

- `get_facility_data` now raises `OpenElectricityError` client-side when
  `facility_code` or `unit_code` lists exceed the API's per-request cap
  (30), with a message naming the param and the limit. Avoids a
  round-trip + the server-side 422 payload. Docstrings updated to
  document the cap. (#10, #36)

### Fixed

- DNS resolution now uses the OS resolver (`ThreadedResolver`/`getaddrinfo`)
  instead of aiodns. The `aiohttp[speedups]` extra installs aiodns, which made
  aiohttp default to the c-ares `AsyncResolver`. c-ares resolves DNS
  independently of the OS stub resolver and failed with
  `aiodns.error.DNSError (11, 'Could not contact DNS servers')` in
  environments where the OS resolver (and `nslookup`) work fine — Windows
  `ProactorEventLoop`, WSL/containers pointing at `127.0.0.53`, split-DNS
  VPNs. The connector is now pinned to `ThreadedResolver`.

## 0.11.2

Bug fixes surfaced by the v0.11.1 end-to-end review.

### Fixed

- `AsyncOEClient.get_market` now accepts `network_region` to match the
  sync `OEClient.get_market`. Async users can apply the same market
  region filter as sync users. (#35)
- `TimeSeriesColumns` now exposes `fueltech` and `renewable`, so
  `secondary_grouping='fueltech'` and `secondary_grouping='renewable'`
  no longer silently drop the grouping value on parse. (#34)

### CI

- New parametrised signature-parity test across the shared `OEClient` /
  `AsyncOEClient` public surface (`__init__`, `get_facilities`,
  `get_network_data`, `get_facility_data`, `get_market`,
  `get_current_user`). Would have caught both this release's
  `network_region` drift and the original v0.11.0 `unit_code` drift.

## 0.11.1

### Fixed

- Removed `DataMetric.RENEWABLE_PROPORTION`. `renewable_proportion` is a
  market-level metric and only works via `get_market` with
  `MarketMetric.RENEWABLE_PROPORTION` (shipped in 0.11.0). The
  `DataMetric` value returned 400s on prod and confused users. (#18, #33)

## 0.11.0

Backwards-compatible fixes, proxy/TLS support, new market metrics, a
notebook-safe sync client, and the first CI matrix.

### Added

- Proxy and TLS/cert configuration on `OEClient` / `AsyncOEClient` —
  keyword-only options `proxy`, `proxy_auth`, `ssl_context`, `ca_cert`,
  `verify_ssl`, `trust_env`. Session construction centralised in
  `BaseOEClient._build_session()`. `ca_cert` adds an extra CA to the
  default trust store rather than replacing it. (#22, #29)
- 9 new `MarketMetric` values: `DEMAND_GROSS`, `DEMAND_GROSS_ENERGY`,
  `GENERATION_RENEWABLE`, `GENERATION_RENEWABLE_ENERGY`,
  `GENERATION_RENEWABLE_WITH_STORAGE`,
  `GENERATION_RENEWABLE_WITH_STORAGE_ENERGY`, `RENEWABLE_PROPORTION`,
  `RENEWABLE_WITH_STORAGE_PROPORTION`, `HYDRO_AND_STORAGE`. (#31)
- `unit_code` argument on `AsyncOEClient.get_facility_data` (already on
  the sync client). (#27)

### Fixed

- Sync `OEClient` is now safe to call from inside an existing event
  loop, including Jupyter / IPython notebooks. Sync methods route
  through `_run_sync()`, which falls back to a worker thread when a
  loop is already running. (#16, #32)
- Python 3.10 support restored — `enum.StrEnum` and `datetime.UTC`
  backports for 3.10, and the PEP 695 generic syntax that broke
  3.10/3.11 reverted to `Generic[T]`. (#28)
- `OpennemUserResponse` no longer fails validation against the `/v4/me`
  response shape; `version` and `created_at` are optional on the user
  response only. (#20)

### CI

- New `.github/workflows/ci.yml` with a test matrix across Python 3.10,
  3.11, 3.12, 3.13 plus a ruff lint/format job. (#30)
- ruff `target-version` aligned to py310 so lint stops suggesting
  py3.12-only generics.

## 0.10.1

- Internal type fixes.
