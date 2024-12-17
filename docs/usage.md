# OpenElectricity Python SDK Usage

The OpenElectricity Python SDK provides a simple interface to interact with the OpenElectricity API. It supports both synchronous and asynchronous operations.

## Installation

```bash
pip install openelectricity
```

## Authentication

The SDK requires an API key for authentication. You can provide it in two ways:

1. Environment variable:

```bash
export OPENELECTRICITY_API_KEY="your-api-key"
```

2. Directly in code:

```python
from openelectricity import OEClient

client = OEClient(api_key="your-api-key")
```

## Synchronous Usage

```python
from openelectricity import OEClient

# Create a client
client = OEClient()

# Use context manager for automatic cleanup
with OEClient() as client:
    # Get list of networks
    networks = client.get_networks()
    print(f"Found {len(networks)} networks")

    # Access network information
    for network in networks:
        print(f"Network: {network.label} ({network.code})")
        print(f"Regions: {[r.code for r in network.regions]}")
```

## Asynchronous Usage

```python
from openelectricity import AsyncOEClient
import asyncio

async def main():
    async with AsyncOEClient() as client:
        # Get list of networks
        networks = await client.get_networks()
        print(f"Found {len(networks)} networks")

        # Access network information
        for network in networks:
            print(f"Network: {network.label} ({network.code})")
            print(f"Regions: {[r.code for r in network.regions]}")

# Run the async code
asyncio.run(main())
```

## Network Operations

### List Networks

```python
# Synchronous
networks = client.get_networks()
for network in networks:
    print(f"Network: {network.label} ({network.code})")
    for region in network.regions:
        print(f"  Region: {region.code} (Timezone: {region.timezone})")

# Asynchronous
networks = await client.get_networks()
for network in networks:
    print(f"Network: {network.label} ({network.code})")
    for region in network.regions:
        print(f"  Region: {region.code} (Timezone: {region.timezone})")
```

The `Network` object includes:

-   `code`: Network code
-   `country`: Country code
-   `label`: Network label/name
-   `regions`: List of regions in the network
-   `timezone`: Network timezone
-   `interval_size`: Size of data intervals in minutes

The `NetworkRegion` object includes:

-   `code`: Region code
-   `timezone`: Region timezone

## Error Handling

The SDK provides custom exceptions for error handling:

-   `OpenElectricityError`: Base exception class
-   `APIError`: Raised when the API returns an error response

```python
from openelectricity import OEClient, OpenElectricityError, APIError

try:
    with OEClient() as client:
        networks = client.get_networks()
except APIError as e:
    print(f"API Error {e.status_code}: {e.detail}")
except OpenElectricityError as e:
    print(f"SDK Error: {e}")
```

## Configuration

You can customize the client by providing a different base URL:

```python
client = OEClient(base_url="https://api.staging.openelectricity.org.au/v4")
```
