"""
Tests for user models.

Covers the ``/v4/me`` response shape, which omits the ``version`` and
``created_at`` fields present on other API responses (see issue #20).
"""

from openelectricity.models.user import OpennemUserResponse


def test_me_response_without_version_and_created_at() -> None:
    """The /v4/me payload omits version/created_at — these must be optional."""
    me_payload = {
        "data": {
            "id": "key_53tFoo",
            "full_name": "Test User",
            "email": "test@example.com",
            "owner_id": "owner_123",
            "plan": "BASIC",
            "meta": {"remaining": 497},
        }
    }

    response = OpennemUserResponse.model_validate(me_payload)

    assert response.version is None
    assert response.created_at is None
    assert response.data.id == "key_53tFoo"
    assert response.data.plan == "BASIC"
    assert response.data.meta is not None
    assert response.data.meta.remaining == 497


def test_me_response_with_version_and_created_at() -> None:
    """A payload that does include version/created_at still validates."""
    me_payload = {
        "version": "4.0.4",
        "created_at": "2026-01-01T00:00:00+10:00",
        "data": {"id": "key_abc"},
    }

    response = OpennemUserResponse.model_validate(me_payload)

    assert response.version == "4.0.4"
    assert response.created_at is not None
    assert response.data.id == "key_abc"
