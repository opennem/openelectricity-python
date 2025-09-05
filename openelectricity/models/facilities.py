"""
Facility models for the OpenElectricity API.

This module contains models related to facility data and responses.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from openelectricity.models.base import APIResponse
from openelectricity.types import NetworkCode, UnitFueltechType, UnitStatusType


class FacilityUnit(BaseModel):
    """A unit within a facility."""

    code: str = Field(..., description="Unit code")
    fueltech_id: UnitFueltechType = Field(..., description="Fuel technology type")
    status_id: UnitStatusType = Field(..., description="Unit status")
    capacity_registered: float = Field(..., description="Registered capacity in MW")
    capacity_maximum: float | None = Field(None, description="Maximum capacity in MW")
    capacity_storage: float | None = Field(None, description="Storage capacity in MWh")
    emissions_factor_co2: float | None = Field(None, description="CO2 emissions factor")
    data_first_seen: datetime | None = Field(None, description="When data was first seen for this unit")
    data_last_seen: datetime | None = Field(None, description="When data was last seen for this unit")
    dispatch_type: str = Field(..., description="Dispatch type")


class FacilityLocation(BaseModel):
    """Location coordinates for a facility."""

    lat: float = Field(..., description="Latitude")
    lng: float = Field(..., description="Longitude")


class Facility(BaseModel):
    """A facility in the OpenElectricity system."""

    code: str = Field(..., description="Facility code")
    name: str = Field(..., description="Facility name")
    network_id: NetworkCode = Field(..., description="Network code")
    network_region: str = Field(..., description="Network region")
    description: str | None = Field(None, description="Facility description")
    npi_id: str | None = Field(None, description="NPI facility ID")
    location: FacilityLocation | None = Field(None, description="Facility location coordinates")
    units: list[FacilityUnit] = Field(..., description="Units within the facility")


class FacilityResponse(APIResponse[Facility]):
    """Response model for facility endpoints."""

    data: list[Facility]
