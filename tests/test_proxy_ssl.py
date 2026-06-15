"""
Tests for proxy and TLS/certificate configuration (issue #22).

These cover how OEClient/AsyncOEClient resolve proxy and SSL options and how
they are applied to the underlying aiohttp ClientSession.
"""

import asyncio
import ssl

import pytest
from aiohttp import BasicAuth

from openelectricity import AsyncOEClient, OEClient
from openelectricity.client import OpenElectricityError

API_KEY = "test-key"


def test_defaults_apply_no_custom_ssl_or_proxy() -> None:
    """With no options, no custom connector/proxy is configured."""
    client = OEClient(api_key=API_KEY)
    assert client._ssl is None
    assert client.proxy is None
    assert client.proxy_auth is None
    assert client.trust_env is False


def test_verify_ssl_false_disables_verification() -> None:
    client = OEClient(api_key=API_KEY, verify_ssl=False)
    assert client._ssl is False


def test_ssl_context_is_passed_through() -> None:
    ctx = ssl.create_default_context()
    client = OEClient(api_key=API_KEY, ssl_context=ctx)
    assert client._ssl is ctx


def test_ca_cert_builds_ssl_context() -> None:
    cafile = ssl.get_default_verify_paths().cafile
    if not cafile:
        pytest.skip("no system CA bundle available to test against")
    client = OEClient(api_key=API_KEY, ca_cert=cafile)
    assert isinstance(client._ssl, ssl.SSLContext)


def test_ssl_context_and_ca_cert_are_mutually_exclusive() -> None:
    with pytest.raises(OpenElectricityError, match="not both"):
        OEClient(api_key=API_KEY, ssl_context=ssl.create_default_context(), ca_cert="/tmp/ca.pem")


def test_custom_ca_with_verify_disabled_is_rejected() -> None:
    with pytest.raises(OpenElectricityError, match="conflicts"):
        OEClient(api_key=API_KEY, verify_ssl=False, ca_cert="/tmp/ca.pem")


def test_proxy_settings_are_stored() -> None:
    auth = BasicAuth("user", "pass")
    client = OEClient(api_key=API_KEY, proxy="http://proxy.corp:8080", proxy_auth=auth, trust_env=True)
    assert client.proxy == "http://proxy.corp:8080"
    assert client.proxy_auth is auth
    assert client.trust_env is True


def test_build_session_applies_proxy_and_trust_env() -> None:
    """_build_session() wires proxy/trust_env onto the ClientSession."""
    client = OEClient(api_key=API_KEY, proxy="http://proxy.corp:8080", trust_env=True)

    async def _check() -> None:
        session = client._build_session()
        try:
            assert "proxy.corp:8080" in str(session._default_proxy)
            assert session._trust_env is True
        finally:
            await session.close()

    asyncio.run(_check())


def test_build_session_uses_threaded_resolver() -> None:
    """DNS goes through the OS resolver, not aiodns/c-ares.

    aiohttp[speedups] installs aiodns and aiohttp then defaults to the c-ares
    AsyncResolver, which fails (DNSError 11, 'Could not contact DNS servers')
    in environments where getaddrinfo works fine. Pin ThreadedResolver so the
    SDK resolves DNS the same way the OS does.
    """
    from aiohttp.resolver import ThreadedResolver

    client = OEClient(api_key=API_KEY)

    async def _check() -> None:
        session = client._build_session()
        try:
            assert isinstance(session.connector._resolver, ThreadedResolver)
        finally:
            await session.close()

    asyncio.run(_check())


def test_async_client_accepts_proxy_and_ssl_kwargs() -> None:
    client = AsyncOEClient(api_key=API_KEY, proxy="http://proxy.corp:8080", verify_ssl=False)
    assert client.proxy == "http://proxy.corp:8080"
    assert client._ssl is False
