"""
Signature-parity tests for the sync and async clients.

Public methods shared between :class:`OEClient` and :class:`AsyncOEClient`
should accept the same parameters. Drift like the missing ``unit_code`` on
``get_facility_data`` (issue #27) or the missing ``network_region`` on
``get_market`` should fail here next time it happens.
"""

import inspect

import pytest

from openelectricity import AsyncOEClient, OEClient

_SHARED_METHODS = [
    "__init__",
    "get_facilities",
    "get_network_data",
    "get_facility_data",
    "get_market",
    "get_current_user",
]


@pytest.mark.parametrize("method_name", _SHARED_METHODS)
def test_sync_async_signature_parity(method_name: str) -> None:
    sync_sig = inspect.signature(getattr(OEClient, method_name))
    async_sig = inspect.signature(getattr(AsyncOEClient, method_name))

    def normalise(sig: inspect.Signature) -> dict[str, tuple[object, inspect._ParameterKind]]:
        return {name: (p.default, p.kind) for name, p in sig.parameters.items() if name != "self"}

    sync_params = normalise(sync_sig)
    async_params = normalise(async_sig)

    if sync_params != async_params:
        sync_only = set(sync_params) - set(async_params)
        async_only = set(async_params) - set(sync_params)
        kind_or_default = {n for n in sync_params.keys() & async_params.keys() if sync_params[n] != async_params[n]}
        pytest.fail(
            f"signature drift on {method_name}: sync-only={sync_only or '{}'}, "
            f"async-only={async_only or '{}'}, differs={kind_or_default or '{}'}"
        )
