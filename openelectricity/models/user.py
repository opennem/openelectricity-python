"""
User models for the OpenElectricity API.

This module contains models related to user data and authentication.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from openelectricity.models.base import APIResponse
from openelectricity.types import OpenNEMRoles


class OpenNEMUserRateLimit(BaseModel):
    """Rate limit information for a user."""

    limit: int = Field(description="Maximum number of API calls allowed in the period")
    remaining: int = Field(description="Remaining API calls in the current period")
    reset: datetime | int = Field(description="When the rate limit resets")


class OpennemAPIRequestMeta(BaseModel):
    """API request metadata."""

    remaining: int | None = Field(None, description="Remaining API calls in the current period")
    reset: datetime | None = Field(None, description="When the rate limit resets")


class OpenNEMUser(BaseModel):
    """User data model representing an OpenElectricity API user."""

    id: str
    full_name: str | None = None
    email: str | None = None
    owner_id: str | None = None
    plan: str | None = None
    rate_limit: OpenNEMUserRateLimit | None = None
    unkey_meta: dict[str, Any] | None = None
    roles: list[OpenNEMRoles] = Field(default_factory=lambda: [OpenNEMRoles.ANONYMOUS])
    meta: OpennemAPIRequestMeta | None = None


class OpennemUserResponse(APIResponse[OpenNEMUser]):
    """Response model for user endpoints."""

    data: OpenNEMUser
