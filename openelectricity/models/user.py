"""
User models for the OpenElectricity API.

This module contains models related to user data and authentication.
"""

from typing import Literal

from pydantic import BaseModel, Field


class UserMeta(BaseModel):
    """User metadata including rate limit information."""

    remaining: int = Field(description="Remaining API calls in the current period")


class UserData(BaseModel):
    """User data model representing an OpenElectricity API user."""

    id: str
    full_name: str
    email: str
    owner_id: str
    plan: Literal["BASIC", "PRO", "ENTERPRISE"] = "BASIC"
    meta: UserMeta


class UserResponse(BaseModel):
    """Response model for user endpoints."""

    data: UserData
