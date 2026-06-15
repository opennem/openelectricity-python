# Changelog

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
