"""
Tests for the pre-flight facility_code / unit_code length check on
get_facility_data (issue #10).
"""

import pytest

from openelectricity import AsyncOEClient, OEClient
from openelectricity.client import OpenElectricityError, _check_facility_list_limit


def _codes(n: int) -> list[str]:
    return [f"F{i:03d}" for i in range(n)]


def test_helper_passes_under_the_cap() -> None:
    _check_facility_list_limit(_codes(30), None)
    _check_facility_list_limit(None, _codes(30))
    _check_facility_list_limit(_codes(15), _codes(15))
    _check_facility_list_limit("ONLY_ONE", None)


def test_helper_raises_on_31_facility_codes() -> None:
    with pytest.raises(OpenElectricityError, match="facility_code accepts at most 30"):
        _check_facility_list_limit(_codes(31), None)


def test_helper_raises_on_31_unit_codes() -> None:
    with pytest.raises(OpenElectricityError, match="unit_code accepts at most 30"):
        _check_facility_list_limit(None, _codes(31))


def test_sync_get_facility_data_rejects_too_many_codes_before_sending() -> None:
    """The check fires before any network call — a dummy api_key is enough."""
    with OEClient(api_key="test") as c:
        with pytest.raises(OpenElectricityError, match="facility_code accepts at most 30"):
            c.get_facility_data(network_code="NEM", facility_code=_codes(36))


@pytest.mark.asyncio
async def test_async_get_facility_data_rejects_too_many_codes_before_sending() -> None:
    c = AsyncOEClient(api_key="test")
    try:
        with pytest.raises(OpenElectricityError, match="unit_code accepts at most 30"):
            await c.get_facility_data(network_code="NEM", unit_code=_codes(31))
    finally:
        await c.close()
