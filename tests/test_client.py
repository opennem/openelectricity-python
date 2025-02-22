"""
Tests for the OpenElectricity API client.

This module contains tests for both synchronous and asynchronous clients.
"""

import pytest

from openelectricity import AsyncOEClient, OEClient
from openelectricity.models.facilities import Facility, FacilityResponse
from openelectricity.types import UnitFueltechType, UnitStatusType


@pytest.fixture
def facility_response() -> dict:
    """Example facility response fixture."""
    return {
        "version": "4.0.3.dev0",
        "created_at": "2025-02-18T07:27:28+11:00",
        "success": True,
        "data": [
            {
                "code": "BAYSW",
                "name": "Bayswater",
                "network_id": "NEM",
                "network_region": "NSW1",
                "description": "Bayswater Power Station",
                "units": [
                    {
                        "code": "BW01",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 660.0,
                        "emissions_factor_co2": 0.88,
                        "data_first_seen": "2010-07-01T00:05:00+10:00",
                        "data_last_seen": "2025-02-22T14:30:00+10:00",
                        "dispatch_type": "GENERATOR",
                    },
                    {
                        "code": "BW02",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 660.0,
                        "emissions_factor_co2": 0.88,
                        "data_first_seen": "2010-07-01T00:05:00+10:00",
                        "data_last_seen": "2025-02-22T14:30:00+10:00",
                        "dispatch_type": "GENERATOR",
                    },
                ],
            }
        ],
        "total_records": 1,
    }


def test_facility_response_parsing(facility_response):
    """Test parsing a complete facility response."""
    response = FacilityResponse.model_validate(facility_response)

    assert response.version == "4.0.3.dev0"
    assert response.success is True
    assert len(response.data) == 1

    # Check facility
    facility = response.data[0]
    assert facility.code == "BAYSW"
    assert facility.name == "Bayswater"
    assert facility.network_id == "NEM"
    assert facility.network_region == "NSW1"
    assert len(facility.units) == 2

    # Check first unit
    unit = facility.units[0]
    assert unit.code == "BW01"
    assert unit.fueltech_id == UnitFueltechType.COAL_BLACK
    assert unit.status_id == UnitStatusType.OPERATING
    assert unit.capacity_registered == 660.0
    assert unit.emissions_factor_co2 == 0.88
    assert unit.dispatch_type == "GENERATOR"


@pytest.mark.asyncio
async def test_async_get_facilities():
    """Test getting facilities with async client."""
    client = AsyncOEClient()
    try:
        facilities = await client.get_facilities(
            network_id=["NEM"],
            status_id=[UnitStatusType.OPERATING],
            fueltech_id=[UnitFueltechType.COAL_BLACK],
        )
        assert isinstance(facilities, FacilityResponse)
        assert facilities.success is True
        assert len(facilities.data) > 0

        # Check first facility
        facility = facilities.data[0]
        assert isinstance(facility, Facility)
        assert facility.network_id == "NEM"
        assert len(facility.units) > 0

    finally:
        await client.close()


def test_sync_get_facilities():
    """Test getting facilities with sync client."""
    with OEClient() as client:
        facilities = client.get_facilities(
            network_id=["NEM"],
            status_id=[UnitStatusType.OPERATING],
            fueltech_id=[UnitFueltechType.COAL_BLACK],
        )
        assert isinstance(facilities, FacilityResponse)
        assert facilities.success is True
        assert len(facilities.data) > 0

        # Check first facility
        facility = facilities.data[0]
        assert isinstance(facility, Facility)
        assert facility.network_id == "NEM"
        assert len(facility.units) > 0
