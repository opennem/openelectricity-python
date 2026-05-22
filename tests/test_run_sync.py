"""
Tests for _run_sync, the helper that lets the synchronous client work both
standalone and inside an already-running event loop (e.g. a notebook).
"""

import asyncio

from openelectricity.client import _run_sync


async def _echo(value: str) -> str:
    await asyncio.sleep(0)
    return value


def test_run_sync_without_running_loop() -> None:
    """With no event loop running, the coroutine runs via asyncio.run."""
    assert _run_sync(_echo("standalone")) == "standalone"


def test_run_sync_inside_running_loop() -> None:
    """With a loop already running (the notebook case), it still completes."""

    async def _notebook_like() -> str:
        # A loop is running here; a naive asyncio.run() would raise RuntimeError.
        return _run_sync(_echo("notebook"))

    assert asyncio.run(_notebook_like()) == "notebook"
