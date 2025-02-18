"""
OpenElectricity API Client

This module provides both synchronous and asynchronous clients for the OpenElectricity API.
"""

import asyncio
from datetime import datetime
from typing import Any, TypeVar, cast

from aiohttp import ClientResponse, ClientSession

from openelectricity.logging import get_logger
from openelectricity.models.networks import Network
from openelectricity.models.timeseries import TimeSeriesResponse
from openelectricity.models.user import UserResponse
from openelectricity.settings_schema import settings
from openelectricity.types import (
    DataInterval,
    DataMetric,
    DataPrimaryGrouping,
    DataSecondaryGrouping,
    MarketMetric,
    NetworkCode,
)

T = TypeVar("T")
logger = get_logger("client")


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
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.base_url = base_url.rstrip("/") if base_url else settings.base_url
        self.api_key = api_key or settings.api_key

        if not self.api_key:
            raise OpenElectricityError(
                "API key must be provided either as argument or via OPENELECTRICITY_API_KEY environment variable"
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        logger.debug("Initialized client with base URL: %s", self.base_url)


class OEClient(BaseOEClient):
    """
    Synchronous client for the OpenElectricity API.

    Note: This client uses aiohttp with asyncio.run() internally to maintain
    API consistency while using the same underlying HTTP client as the async version.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key, base_url)
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        logger.debug("Initialized synchronous client")

    def _ensure_session(self) -> None:
        """Ensure session and event loop are initialized."""
        if self._session is None or self._session.closed:
            logger.debug("Creating new client session")
            self._session = ClientSession(
                base_url=self.base_url,
                headers=self.headers,
            )

    async def _handle_response(self, response: ClientResponse) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle API response and raise appropriate errors."""
        if not response.ok:
            try:
                detail = (await response.json()).get("detail", response.reason)
            except Exception:
                detail = response.reason
            logger.error("API error: %s - %s", response.status, detail)
            raise APIError(response.status, detail)

        logger.debug("Received successful response: %s", response.status)
        return await response.json()

    async def _async_get_networks(self) -> list[Network]:
        """Async implementation of get_networks."""
        logger.debug("Getting networks")
        self._ensure_session()
        async with cast(ClientSession, self._session).get("/networks") as response:
            data = await self._handle_response(response)
            return [Network.model_validate(item) for item in data]

    async def _async_get_network_data(
        self,
        network_code: NetworkCode,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
        secondary_grouping: DataSecondaryGrouping | None = None,
    ) -> TimeSeriesResponse:
        """Async implementation of get_network_data."""
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
        facility_code: str,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
    ) -> TimeSeriesResponse:
        """Async implementation of get_facility_data."""
        logger.debug(
            "Getting facility data for %s/%s (metrics: %s, interval: %s)",
            network_code,
            facility_code,
            metrics,
            interval,
        )
        self._ensure_session()
        params = {
            "metrics": [m.value for m in metrics],
            "interval": interval,
            "date_start": date_start.isoformat() if date_start else None,
            "date_end": date_end.isoformat() if date_end else None,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self._session).get(
            f"/data/facility/{network_code}/{facility_code}", params=params
        ) as response:
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
    ) -> TimeSeriesResponse:
        """Async implementation of get_market."""
        logger.debug(
            "Getting market data for %s (metrics: %s, interval: %s)",
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
            "primary_grouping": primary_grouping,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self._session).get(f"/market/network/{network_code}", params=params) as response:
            data = await self._handle_response(response)
            return TimeSeriesResponse.model_validate(data)

    async def _async_get_current_user(self) -> UserResponse:
        """Async implementation of get_current_user."""
        logger.debug("Getting current user information")
        self._ensure_session()
        async with cast(ClientSession, self._session).get("/me") as response:
            data = await self._handle_response(response)
            return UserResponse.model_validate(data)

    def get_networks(self) -> list[Network]:
        """Get a list of networks."""

        async def _run():
            async with ClientSession(base_url=self.base_url, headers=self.headers) as session:
                self._session = session
                return await self._async_get_networks()

        return asyncio.run(_run())

    def get_network_data(
        self,
        network_code: NetworkCode,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
        secondary_grouping: DataSecondaryGrouping | None = None,
    ) -> TimeSeriesResponse:
        """Get network data for specified metrics."""

        async def _run():
            async with ClientSession(base_url=self.base_url, headers=self.headers) as session:
                self._session = session
                return await self._async_get_network_data(
                    network_code, metrics, interval, date_start, date_end, primary_grouping, secondary_grouping
                )

        return asyncio.run(_run())

    def get_facility_data(
        self,
        network_code: NetworkCode,
        facility_code: str,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
    ) -> TimeSeriesResponse:
        """Get facility data for specified metrics."""

        async def _run():
            async with ClientSession(base_url=self.base_url, headers=self.headers) as session:
                self._session = session
                return await self._async_get_facility_data(network_code, facility_code, metrics, interval, date_start, date_end)

        return asyncio.run(_run())

    def get_market(
        self,
        network_code: NetworkCode,
        metrics: list[MarketMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
    ) -> TimeSeriesResponse:
        """Get market data for specified metrics."""

        async def _run():
            async with ClientSession(base_url=self.base_url, headers=self.headers) as session:
                self._session = session
                return await self._async_get_market(network_code, metrics, interval, date_start, date_end, primary_grouping)

        return asyncio.run(_run())

    def get_current_user(self) -> UserResponse:
        """Get current user information."""

        async def _run():
            async with ClientSession(base_url=self.base_url, headers=self.headers) as session:
                self._session = session
                return await self._async_get_current_user()

        return asyncio.run(_run())

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._session and not self._session.closed:

            async def _close():
                await cast(ClientSession, self._session).close()

            asyncio.run(_close())

    def __enter__(self) -> "OEClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AsyncOEClient(BaseOEClient):
    """
    Asynchronous client for the OpenElectricity API.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key, base_url)
        self.client: ClientSession | None = None
        logger.debug("Initialized asynchronous client")

    async def _ensure_client(self) -> None:
        """Ensure client session is initialized."""
        if self.client is None or self.client.closed:
            logger.debug("Creating new async client session")
            self.client = ClientSession(
                base_url=self.base_url,
                headers=self.headers,
            )

    async def _handle_response(self, response: ClientResponse) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle API response and raise appropriate errors."""
        if not response.ok:
            try:
                detail = (await response.json()).get("detail", response.reason)
            except Exception:
                detail = response.reason
            logger.error("API error: %s - %s", response.status, detail)
            raise APIError(response.status, detail)

        logger.debug("Received successful response: %s", response.status)
        return await response.json()

    async def get_networks(self) -> list[Network]:
        """Get a list of networks."""
        logger.debug("Getting networks")
        await self._ensure_client()
        async with cast(ClientSession, self.client).get("/networks") as response:
            data = await self._handle_response(response)
            return [Network.model_validate(item) for item in data]

    async def get_network_data(
        self,
        network_code: NetworkCode,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
        primary_grouping: DataPrimaryGrouping | None = None,
        secondary_grouping: DataSecondaryGrouping | None = None,
    ) -> TimeSeriesResponse:
        """Get network data for specified metrics."""
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
        facility_code: str,
        metrics: list[DataMetric],
        interval: DataInterval | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
    ) -> TimeSeriesResponse:
        """Get facility data for specified metrics."""
        logger.debug(
            "Getting facility data for %s/%s (metrics: %s, interval: %s)",
            network_code,
            facility_code,
            metrics,
            interval,
        )
        await self._ensure_client()
        params = {
            "metrics": [m.value for m in metrics],
            "interval": interval,
            "date_start": date_start.isoformat() if date_start else None,
            "date_end": date_end.isoformat() if date_end else None,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        logger.debug("Request parameters: %s", params)

        async with cast(ClientSession, self.client).get(
            f"/data/facility/{network_code}/{facility_code}", params=params
        ) as response:
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

    async def get_current_user(self) -> UserResponse:
        """Get current user information."""
        logger.debug("Getting current user information")
        await self._ensure_client()
        async with cast(ClientSession, self.client).get("/me") as response:
            data = await self._handle_response(response)
            return UserResponse.model_validate(data)

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
