"""
Base models for the OpenElectricity API.

This module contains the base models used across the API.
"""

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, Field


class APIResponse[T](BaseModel):
    """Base API response model."""

    version: str
    created_at: datetime
    success: bool = True
    error: str | None = None
    data: Sequence[T] = Field(default_factory=list)
    total_records: int | None = None
