# OpenElectricity Python Client

![logo](https://platform.openelectricity.org.au/oe_logo_full.png)

> [!WARNING]
> This project and the v4 API are currently under active development.

A Python client for the [OpenElectricity](https://openelectricity.org.au) API, providing access to electricity and energy network data and metrics for Australia.

> [!NOTE]
> API key signups are currently waitlisted and will be released gradually.

To obtain an API key visit [platform.openelectricity.org.au](https://platfrom.openelectricity.org.au)

For documentation visit [docs.openelectricity.org.au](https://docs.openelectricity.org.au/introduction)

## Features

-   Synchronous and asynchronous API clients
-   Fully typed with comprehensive type annotations
-   Automatic request retries and error handling
-   Context manager support
-   Modern Python (3.10+) with full type annotations

## Installation

```bash
pip install openelectricity

# or
uv add openelectricity
```

## Quick Start

Setup your API key in the environment variable `OPENELECTRICITY_API_KEY`.

```bash
export OPENELECTRICITY_API_KEY=<your-api-key>
```

Then in code:

```python
from openelectricity import OEClient

# Using environment variable OPENELECTRICITY_API_KEY
with OEClient() as client:
    # API calls will be implemented here
    pass

# Or provide API key directly (not recommended!)
client = OEClient(api_key="your-api-key")
```

For async usage:

```python
from openelectricity import AsyncOEClient
import asyncio

async def main():
    async with AsyncOEClient() as client:
        # API calls will be implemented here
        pass

asyncio.run(main())
```

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
