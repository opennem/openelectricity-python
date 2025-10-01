# OpenElectricity Test Suite

This directory contains the test suite for the OpenElectricity Python client library.

## Test Configuration

The test suite uses pytest fixtures to automatically load API keys and configure clients. The fixtures are defined in `conftest.py`.

### Available Fixtures

- `openelectricity_api_key`: Provides the API key from environment or .env file
- `openelectricity_client`: Provides a configured OEClient instance
- `openelectricity_async_client`: Provides a configured AsyncOEClient instance

### Setting Up API Keys

To run tests that require API access, you need to set up your API key:

1. **Option 1: Environment Variable**
   ```bash
   export OPENELECTRICITY_API_KEY=your_api_key_here
   ```

2. **Option 2: .env File**
   Create a `.env` file in the project root:
   ```
   OPENELECTRICITY_API_KEY=your_api_key_here
   ```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test categories
uv run pytest tests/models/          # Model validation tests
uv run pytest tests/test_client.py   # Client tests
uv run pytest tests/test_*pyspark*   # PySpark integration tests

# Run with verbose output
uv run pytest tests/ -v

# Run only tests that don't require API access
uv run pytest tests/models/ tests/test_client.py::test_facility_response_parsing
```

### Test Categories

- **Model Tests** (`tests/models/`): Test Pydantic model validation and parsing
- **Client Tests** (`tests/test_client.py`): Test API client functionality
- **PySpark Tests** (`tests/test_*pyspark*`): Test PySpark DataFrame conversion
- **Integration Tests**: Test end-to-end functionality

### Test Behavior

- Tests that require API access will be **skipped** if no API key is available
- Tests that require PySpark will be **skipped** if PySpark is not installed
- All tests include proper error handling and graceful degradation

### Custom Markers

The test suite defines custom pytest markers:

- `@pytest.mark.api`: Tests that require API access
- `@pytest.mark.pyspark`: Tests that require PySpark
- `@pytest.mark.integration`: Integration tests

### Fixture Usage

```python
def test_example(openelectricity_client):
    """Example test using the client fixture."""
    response = openelectricity_client.get_facilities()
    assert response is not None
```

The fixtures automatically handle:
- Loading API keys from environment or .env file
- Creating configured client instances
- Graceful skipping when dependencies are unavailable
