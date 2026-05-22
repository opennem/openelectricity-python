"""
OpenElectricity API Client

This module provides both synchronous and asynchronous clients for the OpenElectricity API.
"""

import asyncio
import concurrent.futures
import os
import ssl
from collections.abc import Coroutine
from datetime import datetime
from typing import Any, TypeVar, cast

from aiohttp import BasicAuth, ClientResponse, ClientSession, TCPConnector

from openelectricity.logging import get_logger
from openelectricity.models.facilities import FacilityResponse
from openelectricity.models.timeseries import TimeSeriesResponse
from openelectricity.models.user import OpennemUserResponse
from openelectricity.types import (
    DataInterval,
    DataMetric,
    DataPrimaryGrouping,
    DataSecondaryGrouping,
    FueltechGroupType,
    MarketMetric,
    NetworkCode,
    UnitFueltechType,
    UnitStatusType,
)

T = TypeVar("T")
logger = get_logger("client")


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code.

    Uses ``asyncio.run`` when no event loop is running. When a loop is already
    running — as in a Jupyter/IPython notebook — the coroutine is run on its
    own loop in a worker thread, so the synchronous client works there too.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class OpenElectricityError(Exception):
    """Base exception for OpenElectricity API errors."""

    pass


class APIError(OpenElectricityError):
    """Exception raised for API errors."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


class BaseOEClient:
    """
    Base client for the OpenElectricity API.

    Args:
        api_key: Optional API key for authentication. If not provided, will look for
                OPENELECTRICITY_API_KEY environment variable.
        base_url: Optional base URL for the API. Defaults to production API.
        proxy: Optional proxy URL (e.g. "http://proxy.corp:8080") for all requests.
        proxy_auth: Optional aiohttp.BasicAuth for proxy authentication.
        verify_ssl: Whether to verify TLS certificates. Defaults to True. Set to
                False to disable verification (not recommended).
        ssl_context: Optional pre-built ssl.SSLContext, e.g. for a corporate CA.
        ca_cert: Optional path to a CA certificate bundle. A convenience over
                ssl_context for the common "trust this extra CA" case.
        trust_env: Whether aiohttp should read proxy settings, .netrc and TLS
                config from the environment (HTTP_PROXY/HTTPS_PROXY etc).
                Defaults to False.

    Note:
        ssl_context and ca_cert are mutually exclusive, and neither can be
        combined with verify_ssl=False.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        proxy: str | None = None,
        proxy_auth: BasicAuth | None = None,
        verify_ssl: bool = True,
        ssl_context: ssl.SSLContext | None = None,
        ca_cert: str | os.PathLike[str] | None = None,
        trust_env: bool = False,
    ) -> None:
        # Ensure base_url has a trailing slash for aiohttp ClientSession
        if base_url:
            self.base_url = base_url.rstrip("/") + "/"
        else:
            self.base_url = os.getenv("OPENELECTRICITY_API_URL", "https://api.openelectricity.org.au/v4/")
            if not self.base_url.endswith("/"):
                self.base_url += "/"

        self.api_key = api_key or os.getenv("OPENELECTRICITY_API_KEY")

        if not self.api_key:
            raise OpenElectricityError(
                "API key must be provided either as argument or via OPENELECTRICITY_API_KEY environment variable"
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Proxy and TLS configuration, applied in _build_session().
        self.proxy = proxy
        self.proxy_auth = proxy_auth
        self.trust_env = trust_env
        self._ssl = self._resolve_ssl(verify_ssl, ssl_context, ca_cert)

        logger.debug("Initialized client with base URL: %s", self.base_url)

    @staticmethod
    def _resolve_ssl(
        verify_ssl: bool,
        ssl_context: ssl.SSLContext | None,
        ca_cert: str | os.PathLike[str] | None,
    ) -> ssl.SSLContext | bool | None:
        """Resolve TLS settings into a value for aiohttp's TCPConnector.

        Returns None when defaults apply (no custom connector needed), False to
        disable verification, or an SSLContext for a custom CA.
        """
        if ssl_context is not None and ca_cert is not None:
            raise OpenElectricityError("Provide either ssl_context or ca_cert, not both")
        if (ssl_context is not None or ca_cert is not None) and not verify_ssl:
            raise OpenElectricityError("verify_ssl=False conflicts with a custom ssl_context/ca_cert")

        if ssl_context is not None:
            return ssl_context
        if ca_cert is not None:
            # Add the extra CA to the default trust store rather than
            # replacing it (cafile= on create_default_context would drop the
            # system CAs and break normal public TLS).
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=os.fspath(ca_cert))
            return context
        if not verify_ssl:
            return False
        return None

    def _build_session(self) -> ClientSession:
        """Create a ClientSession with the configured proxy and TLS options.

        Single point of session construction so proxy, custom certificates and
        trust_env behaviour stay consistent across the sync and async clients.
        """
        connector = TCPConnector(ssl=self._ssl) if self._ssl is not None else None
        return ClientSession(
            base_url=self.base_url,
            headers=self.headers,
            trust_env=self.trust_env,
            proxy=self.proxy,
            proxy_auth=self.proxy_auth,
            connector=connector,
        )


class OEClient(BaseOEClient):
    """
    Synchronous client for the OpenElectricity API.

    It runs aiohttp under the hood and is safe to call from inside an existing
    event loop (e.g. a Jupyter/IPython notebook) — when a loop is already
    running, requests are dispatched to a worker thread.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        proxy: str | None = None,
        proxy_auth: BasicAuth | None = None,
        verify_ssl: bool = True,
        ssl_context: ssl.SSLContext | None = None,
        ca_cert: str | os.PathLike[str] | None = None,
        trust_env: bool = False,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            proxy=proxy,
            proxy_auth=proxy_auth,
            verify_ssl=verify_ssl,
            ssl_context=ssl_context,
            ca_cert=ca_cert,
            trust_env=trust_env,
        )
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        logger.debug("Initialized synchronous client")

    def _ensure_session(self) -> None:
        """Ensure session and event loop are initialized."""
        if self._session is None or self._session.closed:
            logger.debug("Creating new client session")
            self._session = self._build_session()

    async def _handle_response(self, response: ClientResponse) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle API response and raise appropriate errors."""
        if not response.ok:
            try:
                detail = (await response.json()).get("detail", response.reason or "Unknown error")
            except Exception:
                detail = response.reason or "Unknown error"
            logger.error("API error: %s - %s", response.status, detail)
            raise APIError(response.status, detail)

        logger.debug("Received successful response: %s", response.status)
        return await response.json()

    async def _async_get_facilities(
        self,
        facility_code: list[str] | None = None,
        status_id: list[UnitStatusType | str] | None = None,
        fueltech_id: list[UnitFueltechType | str] | None = None,
        network_id: list[str] | None = None,
        network_region: str | None = None,
    ) -> FacilityResponse:
        """Async implementation of get_facilities."""
        logger.debug("Getting facilities")
        self._ensure_session()
        params = {
            "facility_code": facility_code,
            "status_id": [s.value if hasattr(s, "value") else s for s in status_id] if status_id else None,
            "fueltech_id": [f.value if hasattr(f, "value") else f for f in fueltech_id] if fueltech_id else None,
            "network_id": network_id,
            "network_region": network_region,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self._session).get("/facilities/", params=params) as response:
            data = await self._handle_response(response)
            return FacilityResponse.model_validate(data)

    async def _async_get_network_data(
        self,
        network_code: NetworkCode,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        network_region: str | None = None,
        fueltech: list[UnitFueltechType] | None = None,
        fueltech_group: list[FueltechGroupType] | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
        secondary_grouping: DataSecondaryGrouping | None = None,
    ) -> TimeSeriesResponse:
        """
        Async implementation of get_network_data.

        Args:
            network_code: The network to get data for
            metrics: List of metrics to query (e.g. energy, power, price)
            interval: The time interval to aggregate by
            date_start: Start time for the query
            date_end: End time for the query
            network_region: Network region to get data for
            fueltech: List of individual fuel technologies to filter by (UnitFueltechType enum values)
            fueltech_group: List of fuel technology groups to filter by (FueltechGroupType enum values)
            primary_grouping: Primary grouping to apply
            secondary_grouping: Optional secondary grouping to apply

        Returns:
            TimeSeriesResponse: Time series data response containing a list of TimeSeries objects
        """
        logger.debug(
            "Getting network data for %s (metrics: %s, interval: %s)",
            network_code,
            metrics,
            interval,
        )
        self._ensure_session()
        params = {
            "metrics": [m.value for m in metrics],
            "interval": interval,
            "date_start": date_start.isoformat() if date_start else None,
            "date_end": date_end.isoformat() if date_end else None,
            "network_region": network_region,
            "fueltech": [f.value for f in fueltech] if fueltech else None,
            "fueltech_group": [fg.value for fg in fueltech_group] if fueltech_group else None,
            "primary_grouping": primary_grouping,
            "secondary_grouping": secondary_grouping,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self._session).get(f"/data/network/{network_code}", params=params) as response:
            data = await self._handle_response(response)
            return TimeSeriesResponse.model_validate(data)

    async def _async_get_facility_data(
        self,
        network_code: NetworkCode,
        facility_code: str | list[str] | None = None,
        metrics: list[DataMetric] | None = None,
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        unit_code: str | list[str] | None = None,
    ) -> TimeSeriesResponse:
        """Async implementation of get_facility_data."""
        logger.debug(
            "Getting facility data for %s/%s (metrics: %s, interval: %s)",
            network_code,
            facility_code or unit_code,
            metrics,
            interval,
        )
        self._ensure_session()
        params = {
            "facility_code": facility_code,
            "unit_code": unit_code,
            "metrics": [m.value for m in metrics] if metrics else None,
            "interval": interval,
            "date_start": date_start.isoformat() if date_start else None,
            "date_end": date_end.isoformat() if date_end else None,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self._session).get(f"/data/facilities/{network_code}", params=params) as response:
            data = await self._handle_response(response)
            return TimeSeriesResponse.model_validate(data)

    async def _async_get_market(
        self,
        network_code: NetworkCode,
        metrics: list[MarketMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
        network_region: str | None = None,
    ) -> TimeSeriesResponse:
        """Async implementation of get_market."""
        logger.debug(
            "Getting market data for %s (metrics: %s, interval: %s, region: %s)",
            network_code,
            metrics,
            interval,
            network_region,
        )
        self._ensure_session()
        params = {
            "metrics": [m.value for m in metrics],
            "interval": interval,
            "date_start": date_start.isoformat() if date_start else None,
            "date_end": date_end.isoformat() if date_end else None,
            "primary_grouping": primary_grouping,
            "network_region": network_region,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self._session).get(f"/market/network/{network_code}", params=params) as response:
            data = await self._handle_response(response)
            return TimeSeriesResponse.model_validate(data)

    async def _async_get_current_user(self) -> OpennemUserResponse:
        """Async implementation of get_current_user."""
        logger.debug("Getting current user information")
        self._ensure_session()
        async with cast(ClientSession, self._session).get("/me") as response:
            data = await self._handle_response(response)
            return OpennemUserResponse.model_validate(data)

    def get_facilities(
        self,
        facility_code: list[str] | None = None,
        status_id: list[UnitStatusType | str] | None = None,
        fueltech_id: list[UnitFueltechType | str] | None = None,
        network_id: list[str] | None = None,
        network_region: str | None = None,
    ) -> FacilityResponse:
        """Get a list of facilities."""

        async def _run():
            async with self._build_session() as session:
                self._session = session
                return await self._async_get_facilities(facility_code, status_id, fueltech_id, network_id, network_region)

        return _run_sync(_run())

    def get_network_data(
        self,
        network_code: NetworkCode,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        network_region: str | None = None,
        fueltech: list[UnitFueltechType] | None = None,
        fueltech_group: list[FueltechGroupType] | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
        secondary_grouping: DataSecondaryGrouping | None = None,
    ) -> TimeSeriesResponse:
        """
        Get network data for specified metrics.

        Args:
            network_code: The network to get data for
            metrics: List of metrics to query (e.g. energy, power, price)
            interval: The time interval to aggregate by
            date_start: Start time for the query
            date_end: End time for the query
            network_region: Network region to get data for
            fueltech: List of individual fuel technologies to filter by (UnitFueltechType enum values)
            fueltech_group: List of fuel technology groups to filter by (FueltechGroupType enum values)
            primary_grouping: Primary grouping to apply
            secondary_grouping: Optional secondary grouping to apply

        Returns:
            TimeSeriesResponse: Time series data response containing a list of TimeSeries objects
        """

        async def _run():
            async with self._build_session() as session:
                self._session = session
                return await self._async_get_network_data(
                    network_code,
                    metrics,
                    interval,
                    date_start,
                    date_end,
                    network_region,
                    fueltech,
                    fueltech_group,
                    primary_grouping,
                    secondary_grouping,
                )

        return _run_sync(_run())

    def get_facility_data(
        self,
        network_code: NetworkCode,
        facility_code: str | list[str] | None = None,
        metrics: list[DataMetric] | None = None,
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        unit_code: str | list[str] | None = None,
    ) -> TimeSeriesResponse:
        """Get facility data for specified metrics."""

        async def _run():
            async with self._build_session() as session:
                self._session = session
                return await self._async_get_facility_data(
                    network_code, facility_code, metrics, interval, date_start, date_end, unit_code
                )

        return _run_sync(_run())

    def get_market(
        self,
        network_code: NetworkCode,
        metrics: list[MarketMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
        network_region: str | None = None,
    ) -> TimeSeriesResponse:
        """Get market data for specified metrics."""

        async def _run():
            async with self._build_session() as session:
                self._session = session
                return await self._async_get_market(
                    network_code, metrics, interval, date_start, date_end, primary_grouping, network_region
                )

        return _run_sync(_run())

    def get_current_user(self) -> OpennemUserResponse:
        """Get current user information."""

        async def _run():
            async with self._build_session() as session:
                self._session = session
                return await self._async_get_current_user()

        return _run_sync(_run())

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._session and not self._session.closed:

            async def _close():
                await cast(ClientSession, self._session).close()

            _run_sync(_close())

    def __enter__(self) -> "OEClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AsyncOEClient(BaseOEClient):
    """
    Asynchronous client for the OpenElectricity API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        proxy: str | None = None,
        proxy_auth: BasicAuth | None = None,
        verify_ssl: bool = True,
        ssl_context: ssl.SSLContext | None = None,
        ca_cert: str | os.PathLike[str] | None = None,
        trust_env: bool = False,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            proxy=proxy,
            proxy_auth=proxy_auth,
            verify_ssl=verify_ssl,
            ssl_context=ssl_context,
            ca_cert=ca_cert,
            trust_env=trust_env,
        )
        self.client: ClientSession | None = None
        logger.debug("Initialized asynchronous client")

    async def _ensure_client(self) -> None:
        """Ensure client session is initialized."""
        if self.client is None or self.client.closed:
            logger.debug("Creating new async client session")
            self.client = self._build_session()

    async def _handle_response(self, response: ClientResponse) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle API response and raise appropriate errors."""
        if not response.ok:
            try:
                detail = (await response.json()).get("detail", response.reason or "Unknown error")
            except Exception:
                detail = response.reason or "Unknown error"
            logger.error("API error: %s - %s", response.status, detail)
            raise APIError(response.status, detail)

        logger.debug("Received successful response: %s", response.status)
        return await response.json()

    async def get_facilities(
        self,
        facility_code: list[str] | None = None,
        status_id: list[UnitStatusType | str] | None = None,
        fueltech_id: list[UnitFueltechType | str] | None = None,
        network_id: list[str] | None = None,
        network_region: str | None = None,
    ) -> FacilityResponse:
        """Get a list of facilities."""
        logger.debug("Getting facilities")
        await self._ensure_client()
        params = {
            "facility_code": facility_code,
            "status_id": [s.value if hasattr(s, "value") else s for s in status_id] if status_id else None,
            "fueltech_id": [f.value if hasattr(f, "value") else f for f in fueltech_id] if fueltech_id else None,
            "network_id": network_id,
            "network_region": network_region,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self.client).get("/facilities/", params=params) as response:
            data = await self._handle_response(response)
            return FacilityResponse.model_validate(data)

    async def get_network_data(
        self,
        network_code: NetworkCode,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        network_region: str | None = None,
        fueltech: list[UnitFueltechType] | None = None,
        fueltech_group: list[FueltechGroupType] | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
        secondary_grouping: DataSecondaryGrouping | None = None,
    ) -> TimeSeriesResponse:
        """
        Get network data for specified metrics.

        Args:
            network_code: The network to get data for
            metrics: List of metrics to query (e.g. energy, power, price)
            interval: The time interval to aggregate by
            date_start: Start time for the query
            date_end: End time for the query
            network_region: Network region to get data for
            fueltech: List of individual fuel technologies to filter by (UnitFueltechType enum values)
            fueltech_group: List of fuel technology groups to filter by (FueltechGroupType enum values)
            primary_grouping: Primary grouping to apply
            secondary_grouping: Optional secondary grouping to apply

        Returns:
            TimeSeriesResponse: Time series data response containing a list of TimeSeries objects
        """
        logger.debug(
            "Getting network data for %s (metrics: %s, interval: %s)",
            network_code,
            metrics,
            interval,
        )
        await self._ensure_client()
        params = {
            "metrics": [m.value for m in metrics],
            "interval": interval,
            "date_start": date_start.isoformat() if date_start else None,
            "date_end": date_end.isoformat() if date_end else None,
            "network_region": network_region,
            "fueltech": [f.value for f in fueltech] if fueltech else None,
            "fueltech_group": [fg.value for fg in fueltech_group] if fueltech_group else None,
            "primary_grouping": primary_grouping,
            "secondary_grouping": secondary_grouping,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self.client).get(f"/data/network/{network_code}", params=params) as response:
            data = await self._handle_response(response)
            return TimeSeriesResponse.model_validate(data)

    async def get_facility_data(
        self,
        network_code: NetworkCode,
        facility_code: str | list[str] | None = None,
        metrics: list[DataMetric] | None = None,
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        unit_code: str | list[str] | None = None,
    ) -> TimeSeriesResponse:
        """Get facility data for specified metrics."""
        logger.debug(
            "Getting facility data for %s/%s (metrics: %s, interval: %s)",
            network_code,
            facility_code or unit_code,
            metrics,
            interval,
        )
        await self._ensure_client()
        params = {
            "facility_code": facility_code,
            "unit_code": unit_code,
            "metrics": [m.value for m in metrics] if metrics else None,
            "interval": interval,
            "date_start": date_start.isoformat() if date_start else None,
            "date_end": date_end.isoformat() if date_end else None,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self.client).get(f"/data/facilities/{network_code}", params=params) as response:
            data = await self._handle_response(response)
            return TimeSeriesResponse.model_validate(data)

    async def get_market(
        self,
        network_code: NetworkCode,
        metrics: list[MarketMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
    ) -> TimeSeriesResponse:
        """Get market data for specified metrics."""
        logger.debug(
            "Getting market data for %s (metrics: %s, interval: %s)",
            network_code,
            metrics,
            interval,
        )
        await self._ensure_client()
        params = {
            "metrics": [m.value for m in metrics],
            "interval": interval,
            "date_start": date_start.isoformat() if date_start else None,
            "date_end": date_end.isoformat() if date_end else None,
            "primary_grouping": primary_grouping,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self.client).get(f"/market/network/{network_code}", params=params) as response:
            data = await self._handle_response(response)
            return TimeSeriesResponse.model_validate(data)

    async def get_current_user(self) -> OpennemUserResponse:
        """Get current user information."""
        logger.debug("Getting current user information")
        await self._ensure_client()
        async with cast(ClientSession, self.client).get("/me") as response:
            data = await self._handle_response(response)
            return OpennemUserResponse.model_validate(data)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self.client and not self.client.closed:
            logger.debug("Closing async client session")
            await self.client.close()

    async def __aenter__(self) -> "AsyncOEClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
