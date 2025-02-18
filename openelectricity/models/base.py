"""
Base models for the OpenElectricity API.

This module contains the base models used across the API.
"""

from datetime import datetime
from typing import TypeVar, Generic, Any, Sequence
from pydantic import BaseModel, Field


T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Base API response model."""

    version: str
    created_at: datetime
    success: bool = True
    error: str | None = None
    data: Sequence[T] = Field(default_factory=list)
    total_records: int | None = None
