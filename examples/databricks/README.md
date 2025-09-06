# Databricks Examples

This folder contains examples and utilities specifically designed for use with Databricks environments.

## Files

### `openelectricity_etl.py`
A comprehensive ETL (Extract, Transform, Load) module for fetching and processing OpenElectricity data in Databricks environments.

**Key Features:**
- Automatic API key retrieval from Databricks secrets
- PySpark DataFrame creation with proper column naming
- Network data fetching (power, energy, market value, emissions)
- Facility data fetching for multiple facilities
- Automatic Spark session management
- Optimized for Databricks environments

**Usage:**
```python
from examples.databricks.openelectricity_etl import get_network_data, get_facility_data

# Fetch network data
df = get_network_data(
    network="NEM",
    interval="5m", 
    days_back=1
)

# Fetch facility data
facility_dfs = get_facility_data(
    network="NEM",
    facility_codes=["BAYSW", "LONSDALE"],
    interval="5m",
    days_back=7
)
```

### `upload_wheel_to_volume.py`
Utility script for uploading Python wheel files to Databricks Unity Catalog volumes.

**Features:**
- Upload wheel files to specified Unity Catalog volumes
- Automatic file path management
- Overwrite protection
- Error handling and logging

**Usage:**
```python
from examples.databricks.upload_wheel_to_volume import upload_wheel_to_volume

upload_wheel_to_volume(
    wheel_path="dist/openelectricity-0.7.2-py3-none-any.whl",
    catalog_name="your_catalog",
    schema_name="your_schema", 
    volume_name="your_volume"
)
```

### `test_api_fields.py`
Testing script for validating OpenElectricity API field structures and responses.

**Purpose:**
- Validate API response schemas
- Test field presence and data types
- Debug API integration issues
- Ensure data quality and consistency

**Usage:**
```python
python examples/databricks/test_api_fields.py
```

## Requirements

These examples require:
- Databricks environment with Unity Catalog access
- PySpark (for ETL functionality)
- OpenElectricity API key (stored in Databricks secrets)
- Databricks SDK for Python

## Setup

1. **Install Dependencies:**
   ```bash
   pip install databricks-sdk pyspark
   ```

2. **Configure Secrets:**
   - Store your OpenElectricity API key in Databricks secrets
   - Default scope: `daveok`
   - Default key: `openelectricity_api_key`

3. **Unity Catalog Setup:**
   - Create a catalog, schema, and volume for storing wheel files
   - Ensure proper permissions are set

## Notes

- These examples are specifically designed for Databricks environments
- They include Databricks-specific features like Unity Catalog integration
- Error handling is optimized for Databricks logging and monitoring
- Performance is tuned for Databricks Spark clusters
