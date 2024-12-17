# OpenElectricity Python SDK

A Python SDK for interacting with the OpenElectricity API. This SDK provides both synchronous and asynchronous clients for accessing OpenElectricity data.

## Features

-   Synchronous and asynchronous API clients
-   Type hints for better IDE support
-   Automatic request retries and error handling
-   Context manager support for proper resource cleanup
-   Modern Python (3.12+) with full type annotations

## Installation

```bash
pip install openelectricity
```

## Quick Start

```python
from openelectricity import Client

# Using environment variable OPENELECTRICITY_API_KEY
with Client() as client:
    # API calls will be implemented here
    pass

# Or provide API key directly
client = Client(api_key="your-api-key")
```

For async usage:

```python
from openelectricity import AsyncClient
import asyncio

async def main():
    async with AsyncClient() as client:
        # API calls will be implemented here
        pass

asyncio.run(main())
```

## Documentation

For detailed usage instructions and API reference, see the [documentation](docs/usage.md).

## Development

1. Clone the repository
2. Install development dependencies:

    ```bash
    make install
    ```

3. Run tests:

    ```bash
    make test
    ```

4. Format code:

    ```bash
    make format
    ```

5. Run linters:
    ```bash
    make lint
    ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
