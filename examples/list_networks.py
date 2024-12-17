#!/usr/bin/env python
"""
Example script demonstrating how to list networks using the OpenElectricity SDK.

This script shows both synchronous and asynchronous usage patterns.
"""

import asyncio
import os
from typing import NoReturn

from openelectricity import OEClient, AsyncOEClient
from openelectricity.models.networks import Network


def print_network(network: Network) -> None:
    """Print network information in a formatted way."""
    print(f"\nNetwork: {network.label} ({network.code})")
    print(f"  Country: {network.country}")
    print(f"  Network Timezone: {network.timezone}")
    print(f"  Interval Size: {network.interval_size} minutes")
    print("  Regions:")

    if network.regions:
        for region in network.regions:
            print(f"    - {region.code} (Timezone: {region.timezone})")
    else:
        print("    - None")


def sync_example() -> None:
    """Demonstrate synchronous network listing."""
    print("\nSynchronous Example:")
    print("-------------------")

    with OEClient() as client:
        # Get networks
        networks = client.get_networks()
        print(f"\nFound {len(networks)} networks")

        for network in networks:
            print_network(network)


async def async_example() -> None:
    """Demonstrate asynchronous network listing."""
    print("\nAsynchronous Example:")
    print("--------------------")

    async with AsyncOEClient() as client:
        # Get networks
        networks = await client.get_networks()
        print(f"\nFound {len(networks)} networks")

        for network in networks:
            print_network(network)


def main() -> NoReturn:
    """Run both sync and async examples."""
    if not os.getenv("OPENELECTRICITY_API_KEY"):
        raise ValueError("Please set the OPENELECTRICITY_API_KEY environment variable")

    # Run sync example
    sync_example()

    # Run async example
    asyncio.run(async_example())


if __name__ == "__main__":
    main()
