"""
OpenElectricity API Client

This module provides both synchronous and asynchronous clients for the OpenElectricity API.
"""

import os
from typing import Any

import httpx
from pydantic import BaseModel

from openelectricity.models.networks import Network


class OpenElectricityError(Exception):
    """Base exception for OpenElectricity API errors."""

    pass


class APIError(OpenElectricityError):
    """Exception raised for API errors."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


class OEClient:
    """
    Synchronous client for the OpenElectricity API.

    Args:
        api_key: Optional API key for authentication. If not provided, will look for
                OPENELECTRICITY_API_KEY environment variable.
        base_url: Optional base URL for the API. Defaults to production API.
    """

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.openelectricity.org.au/v4") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENELECTRICITY_API_KEY")

        if not self.api_key:
            raise OpenElectricityError(
                "API key must be provided either as argument or via OPENELECTRICITY_API_KEY environment variable"
            )

        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )

    def _handle_response(self, response: httpx.Response) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle API response and raise appropriate errors."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                detail = response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            raise APIError(response.status_code, detail)

        return response.json()

    def get_networks(self) -> list[Network]:
        """
        Get a list of networks.

        Returns:
            List of Network objects
        """
        response = self.client.get("/networks")
        data = self._handle_response(response)
        return [Network.model_validate(item) for item in data]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> "OEClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AsyncOEClient:
    """
    Asynchronous client for the OpenElectricity API.

    Args:
        api_key: Optional API key for authentication. If not provided, will look for
                OPENELECTRICITY_API_KEY environment variable.
        base_url: Optional base URL for the API. Defaults to production API.
    """

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.openelectricity.org.au/v4") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENELECTRICITY_API_KEY")

        if not self.api_key:
            raise OpenElectricityError(
                "API key must be provided either as argument or via OPENELECTRICITY_API_KEY environment variable"
            )

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )

    async def _handle_response(self, response: httpx.Response) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle API response and raise appropriate errors."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                detail = response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            raise APIError(response.status_code, detail)

        return response.json()

    async def get_networks(self) -> list[Network]:
        """
        Get a list of networks.

        Returns:
            List of Network objects
        """
        response = await self.client.get("/networks")
        data = await self._handle_response(response)
        return [Network.model_validate(item) for item in data]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncOEClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
