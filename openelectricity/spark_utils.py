"""
OpenElectricity Spark Utilities

This module provides clean, reusable functions for Spark session management
following Databricks best practices. It ensures consistent Spark session handling
across the SDK whether running in Databricks or local environments.

Key Features:
- Automatic detection of Databricks vs local environment
- Consistent Spark session configuration
- Proper error handling and logging
- Easy to test and maintain
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_spark_session() -> "SparkSession":
    """
    """
    try:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()

def create_spark_dataframe(data, schema=None, spark_session=None) -> "Optional[DataFrame]":
    """
    Create a PySpark DataFrame from data with automatic Spark session management.
    
    This function handles the creation of a Spark session if one is not provided,
    making it easy to convert data to PySpark DataFrames without managing sessions manually.
    
    Args:
        data: Data to convert (list of records, pandas DataFrame, etc.)
        schema: Optional schema for the DataFrame
        spark_session: Optional existing Spark session
        app_name: Name for the Spark application if creating a new session
        
    Returns:
        PySpark DataFrame or None if conversion fails
        
    Example:
        >>> records = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]
        >>> df = create_spark_dataframe(records)
        >>> print(f"Created DataFrame with {df.count()} rows")
    """
    try:
        from pyspark.sql import DataFrame
        
        # Use provided session or create new one
        if spark_session is None:
            spark_session = get_spark_session()
        
        # Create DataFrame
        if schema:
            return spark_session.createDataFrame(data, schema)
        else:
            return spark_session.createDataFrame(data)
            
    except ImportError:
        logger.warning("PySpark not available. Install with: uv add 'openelectricity[analysis]'")
        return None
    except Exception as e:
        logger.error(f"Error creating PySpark DataFrame: {e}")
        return None


def is_spark_available() -> bool:
    """
    Check if PySpark is available in the current environment.
    
    Returns:
        bool: True if PySpark can be imported, False otherwise
        
    Example:
        >>> if is_spark_available():
        ...     print("PySpark is ready to use")
        ... else:
        ...     print("PySpark not available")
    """
    try:
        import pyspark
        return True
    except ImportError:
        return False


def get_spark_version() -> Optional[str]:
    """
    Get the version of PySpark if available.
    
    Returns:
        str: PySpark version string or None if not available
        
    Example:
        >>> version = get_spark_version()
        >>> print(f"PySpark version: {version}")
    """
    try:
        import pyspark
        return pyspark.__version__
    except ImportError:
        return None


def create_spark_dataframe_with_schema(data, schema, spark_session=None):
    """
    Create a PySpark DataFrame with explicit schema for better performance.
    
    Args:
        data: List of dictionaries or similar data structure
        schema: PySpark schema (StructType)
        spark_session: Optional PySpark session. If not provided, will create one.
        app_name: Name for the Spark application if creating a new session.
        
    Returns:
        PySpark DataFrame with explicit schema
    """
    if spark_session is None:
        spark_session = get_spark_session()
    
    return spark_session.createDataFrame(data, schema=schema)


def pydantic_field_to_spark_type(field_info, field_name: str):
    """
    Map a Pydantic field to the appropriate Spark type.
    
    Args:
        field_info: Pydantic field info from model fields
        field_name: Name of the field
        
    Returns:
        Appropriate PySpark data type
    """
    from pyspark.sql.types import StringType, DoubleType, IntegerType, BooleanType, TimestampType
    from typing import get_origin, get_args
    import datetime
    from enum import Enum
    
    # Get the annotation (type) from the field
    annotation = field_info.annotation
    
    # Handle Union types (like str | None)
    origin = get_origin(annotation)
    if origin is type(None) or origin is type(type(None)):
        return StringType()
    elif hasattr(annotation, '__origin__') and annotation.__origin__ is type(None):
        return StringType()
    elif origin is not None:
        args = get_args(annotation)
        # For Union types, get the non-None type
        non_none_types = [arg for arg in args if arg is not type(None)]
        if non_none_types:
            annotation = non_none_types[0]
    
    # Map basic Python types
    if annotation == str:
        return StringType()
    elif annotation == int:
        return IntegerType()
    elif annotation == float:
        return DoubleType()
    elif annotation == bool:
        return BooleanType()
    elif annotation == datetime.datetime or annotation is datetime.datetime:
        return TimestampType()  # Store as timestamp with UTC conversion
    
    # Handle Enum types (including custom enums)
    if hasattr(annotation, '__bases__') and any(issubclass(base, Enum) for base in annotation.__bases__):
        return StringType()
    
    # Handle List types
    if origin == list:
        return StringType()  # Store lists as JSON strings for now
    
    # Default to string for unknown types
    return StringType()


def create_schema_from_pydantic_model(model_class, flattened: bool = False):
    """
    Create a Spark schema directly from a Pydantic model class.
    
    Args:
        model_class: Pydantic model class
        flattened: Whether this is for flattened data (like facilities with units)
        
    Returns:
        PySpark StructType schema
    """
    from pyspark.sql.types import StructType, StructField
    
    schema_fields = []
    
    # Get model fields
    for field_name, field_info in model_class.model_fields.items():
        spark_type = pydantic_field_to_spark_type(field_info, field_name)
        schema_fields.append(StructField(field_name, spark_type, True))  # Allow nulls
    
    return StructType(schema_fields)


def create_facilities_flattened_schema():
    """
    Create a Spark schema for flattened facilities data (facility + unit fields).
    
    Returns:
        PySpark StructType schema optimized for facilities with units
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, TimestampType
    
    # Define schema based on the flattened structure from facilities to_pyspark
    # This includes fields from both Facility and FacilityUnit models
    schema_fields = [
        # Facility fields
        StructField('code', StringType(), True),
        StructField('name', StringType(), True), 
        StructField('network_id', StringType(), True),
        StructField('network_region', StringType(), True),
        StructField('description', StringType(), True),
        StructField('fueltech_id', StringType(), True),
        StructField('status_id', StringType(), True),
        StructField('capacity_registered', DoubleType(), True),  # Keep as number
        StructField('emissions_factor_co2', DoubleType(), True),  # Keep as number
        StructField('data_first_seen', TimestampType(), True),  # UTC timestamp
        StructField('data_last_seen', TimestampType(), True),   # UTC timestamp
        StructField('dispatch_type', StringType(), True),
    ]
    
    return StructType(schema_fields)


def infer_schema_from_data(data, sample_size: int = 100):
    """
    Infer PySpark schema from data with support for variant types.
    
    Args:
        data: List of dictionaries or similar data structure
        sample_size: Number of records to sample for schema inference
        
    Returns:
        PySpark StructType schema
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, VariantType, BooleanType, IntegerType
    
    if not data:
        return StructType()
    
    # Sample data for schema inference
    sample_data = data[:min(sample_size, len(data))]
    
    # Analyze field types across the sample
    field_types = {}
    for record in sample_data:
        for key, value in record.items():
            if key not in field_types:
                field_types[key] = set()
            
            if value is None:
                field_types[key].add(type(None))
            else:
                field_types[key].add(type(value))
    
    # Create schema fields
    schema_fields = []
    for field_name, types in field_types.items():
        # Remove None type for schema definition
        types.discard(type(None))
        
        if not types:
            # All values are None, default to StringType
            schema_fields.append(StructField(field_name, StringType(), True))
        elif len(types) == 1:
            # Single type, use appropriate PySpark type
            value_type = list(types)[0]
            if value_type in (int, float):
                schema_fields.append(StructField(field_name, DoubleType(), True))
            elif value_type == str:
                schema_fields.append(StructField(field_name, StringType(), True))
            elif value_type == bool:
                schema_fields.append(StructField(field_name, BooleanType(), True))
            elif 'datetime' in str(value_type) or 'Timestamp' in str(value_type):
                # Handle datetime/timestamp types - store as TimestampType with UTC conversion
                schema_fields.append(StructField(field_name, TimestampType(), True))
            else:
                # Use string type for safety
                schema_fields.append(StructField(field_name, StringType(), True))
        else:
            # Multiple types, use string type for compatibility
            schema_fields.append(StructField(field_name, StringType(), True))
    
    return StructType(schema_fields)


def create_facility_timeseries_schema():
    """
    Create a static, optimized Spark schema for FACILITY timeseries data.
    
    This schema is specifically designed for facility data which includes
    facility-specific fields and DataMetric types.
    
    Returns:
        PySpark StructType schema optimized for facility timeseries data
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
    
    schema_fields = [
        # Core time and grouping fields
        StructField('interval', TimestampType(), True),  # Network timezone datetime
        StructField('network_id', StringType(), True),   # Network identifier
        StructField('network_region', StringType(), True), # Region within network
        StructField('facility_code', StringType(), True),  # Facility identifier
        StructField('unit_code', StringType(), True),      # Unit identifier
        StructField('fueltech_id', StringType(), True),    # Fuel technology
        StructField('status_id', StringType(), True),      # Unit status
        
        # DataMetric fields (facility-specific)
        StructField('power', DoubleType(), True),          # Power generation
        StructField('energy', DoubleType(), True),         # Energy production
        StructField('market_value', DoubleType(), True),   # Market value
        StructField('emissions', DoubleType(), True),      # Emissions data
        StructField('renewable_proportion', DoubleType(), True), # Renewable proportion
        
        # Additional metadata fields
        StructField('unit_capacity', DoubleType(), True),  # Unit capacity
        StructField('unit_efficiency', DoubleType(), True), # Unit efficiency
    ]
    
    return StructType(schema_fields)


def create_market_timeseries_schema():
    """
    Create a static, optimized Spark schema for MARKET timeseries data.
    
    This schema is specifically designed for market data which includes
    market-specific fields and MarketMetric types.
    
    Returns:
        PySpark StructType schema optimized for market timeseries data
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
    
    schema_fields = [
        # Core time and grouping fields
        StructField('interval', TimestampType(), True),  # Network timezone datetime
        StructField('network_id', StringType(), True),   # Network identifier
        StructField('network_region', StringType(), True), # Region within network
        
        # MarketMetric fields (market-specific)
        StructField('price', DoubleType(), True),         # Price data
        StructField('demand', DoubleType(), True),        # Demand data
        StructField('demand_energy', DoubleType(), True), # Demand energy
        StructField('curtailment', DoubleType(), True),   # General curtailment
        StructField('curtailment_energy', DoubleType(), True), # Curtailment energy
        StructField('curtailment_solar', DoubleType(), True),  # Solar curtailment
        StructField('curtailment_solar_energy', DoubleType(), True), # Solar curtailment energy
        StructField('curtailment_wind', DoubleType(), True),   # Wind curtailment
        StructField('curtailment_wind_energy', DoubleType(), True), # Wind curtailment energy
        
        # Additional metadata fields
        StructField('primary_grouping', StringType(), True), # Primary grouping (fueltech, status, etc.)
    ]
    
    return StructType(schema_fields)


def create_network_timeseries_schema():
    """
    Create a static, optimized Spark schema for NETWORK timeseries data.
    
    This schema is specifically designed for network data which includes
    network-wide aggregations and DataMetric types.
    
    Returns:
        PySpark StructType schema optimized for network timeseries data
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
    
    schema_fields = [
        # Core time and grouping fields
        StructField('interval', TimestampType(), True),  # Network timezone datetime
        StructField('network_id', StringType(), True),   # Network identifier
        StructField('network_region', StringType(), True), # Region within network
        
        # DataMetric fields (network-wide)
        StructField('power', DoubleType(), True),          # Network power
        StructField('energy', DoubleType(), True),         # Network energy
        StructField('market_value', DoubleType(), True),   # Network market value
        StructField('emissions', DoubleType(), True),      # Network emissions
        StructField('renewable_proportion', DoubleType(), True), # Network renewable proportion
        
        # Network grouping fields
        StructField('primary_grouping', StringType(), True),   # Primary grouping (fueltech, status, etc.)
        StructField('secondary_grouping', StringType(), True), # Secondary grouping
    ]
    
    return StructType(schema_fields)


# Legacy function for backward compatibility - now delegates to facility schema
def create_timeseries_schema():
    """
    Create a static, optimized Spark schema for timeseries data.
    
    This function is maintained for backward compatibility but now
    delegates to the facility-specific schema as that's most common.
    
    Returns:
        PySpark StructType schema optimized for facility timeseries data
    """
    return create_facility_timeseries_schema()


def create_minimal_facility_timeseries_schema() -> "StructType":
    """
    Create a minimal Spark schema for FACILITY timeseries data with essential fields.
    
    This function always includes the 7 essential fields for facility data,
    ensuring consistent schema structure regardless of data content.
    
    Args:
        records: List of timeseries records to analyze
        
    Returns:
        PySpark StructType schema with the 7 essential facility fields in specific order
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
    
    # Always include these 7 essential fields for facility data in exact order
    essential_fields = [
        ('interval', TimestampType()),
        ('network_region', StringType()),
        ('power', DoubleType()),
        ('energy', DoubleType()),
        ('emissions', DoubleType()),
        ('market_value', DoubleType()),
        ('facility_code', StringType()),
    ]
    
    # Build schema with all essential fields in the specified order
    schema_fields = []
    for field_name, field_type in essential_fields:
        schema_fields.append(StructField(field_name, field_type, True))
    
    return StructType(schema_fields)


def detect_timeseries_schema(records: list[dict]) -> "StructType":
    """
    Automatically detect the appropriate Spark schema based on the data content.
    
    This function analyzes the records to determine whether they contain
    facility, market, or network data and returns the appropriate schema.
    
    Args:
        records: List of timeseries records
        
    Returns:
        PySpark StructType schema appropriate for the data type
    """
    if not records:
        return create_minimal_facility_timeseries_schema()  # Default fallback
    
    # Get all unique field names from the records
    all_fields = set()
    for record in records:
        all_fields.update(record.keys())
    
    # Check for market-specific fields
    market_fields = {'price', 'demand', 'demand_energy', 'curtailment', 
                     'curtailment_energy', 'curtailment_solar', 'curtailment_wind'}
    if market_fields.intersection(all_fields):
        return create_market_timeseries_schema()
    
    # Check for network-specific fields
    network_fields = {'primary_grouping', 'secondary_grouping'}
    if network_fields.intersection(all_fields):
        return create_network_timeseries_schema()
    
    # For facility data, use minimal schema based on actual data
    return create_minimal_facility_timeseries_schema()


def create_timeseries_dataframe_batched(records: list[dict], spark_session, batch_size: int = 10000) -> "DataFrame":
    """
    Create PySpark DataFrame from timeseries records using batched processing for large datasets.
    
    This function processes data in batches to optimize memory usage and performance
    for very large datasets while maintaining data accuracy.
    
    Args:
        records: List of timeseries records
        spark_session: PySpark session
        batch_size: Number of records to process per batch
        
    Returns:
        PySpark DataFrame with all records
    """
    from pyspark.sql.types import StructType
    
    if not records:
        return None
    
    # Get the optimized schema
    schema = detect_timeseries_schema(records)
    
    # For small datasets, use the fast single-pass method
    if len(records) <= batch_size:
        cleaned_records = clean_timeseries_records_fast(records)
        return spark_session.createDataFrame(cleaned_records, schema=schema)
    
    # For large datasets, process in batches
    dataframes = []
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        cleaned_batch = clean_timeseries_records_fast(batch)
        batch_df = spark_session.createDataFrame(cleaned_batch, schema=schema)
        dataframes.append(batch_df)
    
    # Union all batches
    if len(dataframes) == 1:
        return dataframes[0]
    else:
        # Use reduce to union all dataframes
        from functools import reduce
        return reduce(lambda df1, df2: df1.union(df2), dataframes)


def clean_timeseries_records_fast(records: list[dict]) -> list[dict]:
    """
    Fast, optimized cleaning of timeseries records for PySpark conversion.
    
    This function processes data in batches and uses type-specific optimizations
    to minimize object creation and improve performance.
    
    Args:
        records: List of raw timeseries records
        
    Returns:
        List of cleaned records ready for PySpark DataFrame creation
    """
    if not records:
        return []
    
    import datetime as dt
    
    # Pre-define the set of metric fields for faster lookups
    metric_fields = {'POWER', 'ENERGY', 'MARKET_VALUE', 'EMISSIONS', 'PRICE', 'DEMAND', 'VALUE'}
    
    # Process records in batches for better performance
    cleaned_records = []
    
    for record in records:
        # Create new record dict (minimal object creation)
        cleaned_record = {}
        
        for key, value in record.items():
            if value is None:
                cleaned_record[key] = None
                continue
                
            # Fast type checking using isinstance (faster than hasattr)
            if isinstance(value, dt.datetime):
                # Handle datetime conversion to UTC
                if value.tzinfo is not None:
                    cleaned_record[key] = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
                else:
                    cleaned_record[key] = value  # Already naive, assume UTC
            elif hasattr(value, 'value'):  # Enum objects
                cleaned_record[key] = str(value)
            elif isinstance(value, bool):
                cleaned_record[key] = value
            elif isinstance(value, (int, float)):
                # Convert integers to float for metric fields (better for Spark operations)
                if key in metric_fields:
                    cleaned_record[key] = float(value)
                else:
                    cleaned_record[key] = value
            else:
                # For strings and other types, pass through
                cleaned_record[key] = value
        
        cleaned_records.append(cleaned_record)
    
    return cleaned_records
