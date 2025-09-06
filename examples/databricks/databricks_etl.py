"""
OpenNEM SDK Utilities for Databricks

This module provides clean, reusable functions for OpenNEM API operations
following Databricks best practices. Extract common SDK calls from notebooks
into maintainable, testable utility functions.

Key Features:
- Simple, focused functions for each data type
- Proper error handling and logging
- Databricks-friendly parameter patterns
- Clean separation of concerns
- Easy to test and maintain

Usage:
    from openelectricity import get_market_data, get_network_data
    
    # Simple data fetch
    df = get_market_data(
        api_key="your_key",
        network="NEM",
        interval="5m",
        days_back=1
    )
    
    # Save to table
    save_to_table(df, "bronze_nem_market", catalog, schema)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union

# Optional PySpark imports
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import col, when, lit
    PYSPARK_AVAILABLE = True
except ImportError:
    SparkSession = None
    DataFrame = None
    col = None
    when = None
    lit = None
    PYSPARK_AVAILABLE = False

from openelectricity import OEClient
from openelectricity.types import (
    DataMetric, 
    MarketMetric, 
    DataInterval, 
    NetworkCode,
    DataPrimaryGrouping,
    DataSecondaryGrouping
)


# Configure logging for Databricks
logger = logging.getLogger(__name__)

# Quiet down all OpenElectricity related loggers
logging.getLogger("openelectricity.client").setLevel(logging.ERROR)
logging.getLogger("openelectricity.client.http").setLevel(logging.ERROR)
logging.getLogger("openelectricity").setLevel(logging.ERROR)

# Or set to CRITICAL to see almost nothing
# logging.getLogger("openelectricity.client").setLevel(logging.CRITICAL)


def _get_api_key_from_secrets(secret_scope: str = "daveok", secret_key: str = "openelectricity_api_key") -> str:
    """
    Retrieve OpenNEM API key from Databricks secrets using WorkspaceClient.
    
    Args:
        secret_scope: The name of the secret scope (default: "daveok")
        secret_key: The name of the secret key (default: "openelectricity_api_key")
    
    Returns:
        OpenNEM API key as string
        
    Raises:
        Exception: If unable to retrieve API key from secrets
    """
    try:
        from databricks.sdk import WorkspaceClient
        
        w = WorkspaceClient()
        dbutils = w.dbutils
        
        api_key = dbutils.secrets.get(secret_scope, secret_key)
        logger.info(f"Successfully retrieved API key from secret scope '{secret_scope}'")
        return api_key
        
    except ImportError:
        logger.error("Databricks SDK not available. Please ensure you're running in a Databricks environment.")
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve API key from secret scope '{secret_scope}': {str(e)}")
        raise


def _get_client(api_key: Optional[str] = None, base_url: Optional[str] = None, 
                secret_scope: str = "daveok", secret_key: str = "openelectricity_api_key") -> OEClient:
    """
    Create and return an OpenNEM client instance.
    
    Args:
        api_key: OpenNEM API key (if None, will be retrieved from secrets)
        base_url: Optional custom API base URL
        secret_scope: The name of the secret scope (default: "daveok")
        secret_key: The name of the secret key (default: "openelectricity_api_key")
        
    Returns:
        Configured OEClient instance
    """
    if api_key is None:
        api_key = _get_api_key_from_secrets(secret_scope, secret_key)
    
    return OEClient(api_key=api_key, base_url=base_url)


def _get_spark():
    """
    Get a Spark session that works in both Databricks and local environments.
    
    Returns:
        SparkSession: Configured Spark session
    """
    try:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()


def _calculate_date_range(days_back: int, end_date: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """
    Calculate start and end dates for data fetching.
    
    Args:
        days_back: Number of days to look back
        end_date: End date (defaults to now)
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date


def get_market_data(
    network: NetworkCode,
    interval: DataInterval,
    days_back: int = 1,
    end_date: Optional[datetime] = None,
    primary_grouping: DataPrimaryGrouping = "network_region",
    api_key: Optional[str] = None,
    secret_scope: str = "daveok",
    secret_key: str = "openelectricity_api_key"
) -> DataFrame:
    """
    Fetch market data (price, demand, demand energy) from OpenNEM API.
    
    This function simplifies the common pattern of fetching market data
    and automatically handles date calculations and response conversion.
    If no API key is provided, it will be automatically retrieved from Databricks secrets.
    
    Args:
        network: Network code (NEM, WEM, AU)
        interval: Data interval (5m, 1h, 1d, etc.)
        days_back: Number of days to look back from end_date
        end_date: End date for data fetch (defaults to now)
        primary_grouping: Primary grouping for data aggregation
        api_key: OpenNEM API key (optional, will be retrieved from secrets if not provided)
        secret_scope: The name of the secret scope (default: "daveok")
        secret_key: The name of the secret key (default: "openelectricity_api_key")
        
    Returns:
        PySpark DataFrame with market data and proper column naming
        
    Raises:
        Exception: If API call fails
        
    Example:
        >>> df = get_market_data(
        ...     network="NEM",
        ...     interval="5m",
        ...     days_back=1
        ... )
        >>> print(df.columns)
        ['interval', 'price_dollar_MWh', 'demand_MW', 'demand_energy_MWh', ...]
    """
    start_date, end_date = _calculate_date_range(days_back, end_date)
    
    try:
        with _get_client(api_key, None, secret_scope, secret_key) as client:
            logger.info(f"Fetching market data for {network} network ({interval} intervals)")
            
            response = client.get_market(
                network_code=network,
                metrics=[
                    MarketMetric.PRICE,
                    MarketMetric.DEMAND,
                    MarketMetric.DEMAND_ENERGY
                ],
                interval=interval,
                date_start=start_date,
                date_end=end_date,
                primary_grouping=primary_grouping,
            )
            
            # Convert to Spark DataFrame using native to_pandas method
            pd_df = response.to_pandas()
            units = response.get_metric_units()
            spark = _get_spark()
            spark_df = spark.createDataFrame(pd_df)
            
            # Rename columns to be more descriptive using PySpark operations
            spark_df = spark_df.withColumnRenamed('price', 'price_dollar_MWh')
            spark_df = spark_df.withColumnRenamed('demand', 'demand_MW')
            spark_df = spark_df.withColumnRenamed('demand_energy', 'demand_energy_GWh')
            
            
            logger.info(f"Successfully fetched market data records")
            return spark_df
            
    except Exception as e:
        logger.error(f"Failed to fetch market data for {network}: {str(e)}")
        raise


def get_network_data(
    network: NetworkCode,
    interval: DataInterval,
    days_back: int = 1,
    end_date: Optional[datetime] = None,
    primary_grouping: DataPrimaryGrouping = "network_region",
    secondary_grouping: DataSecondaryGrouping = "fueltech_group",
    api_key: Optional[str] = None
) -> Optional[DataFrame]:
    """
    Fetch network data (power, energy, market value, emissions) from OpenNEM API.
    
    If no API key is provided, it will be automatically retrieved from Databricks secrets.
    
    Args:
        network: Network code (NEM, WEM, AU)
        interval: Data interval (5m, 1h, 1d, etc.)
        days_back: Number of days to look back from end_date
        end_date: End date for data fetch (defaults to now)
        primary_grouping: Primary grouping for data aggregation
        secondary_grouping: Secondary grouping for data aggregation
        api_key: OpenNEM API key (optional, will be retrieved from secrets if not provided)
        
    Returns:
        PySpark DataFrame with network data and proper column naming
        
    Example:
        >>> df = get_network_data(
        ...     network="NEM",
        ...     interval="5m",
        ...     days_back=1
        ... )
    """
    if not PYSPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required for get_network_data. Install it with: uv add 'openelectricity[analysis]'"
        )
    
    start_date, end_date = _calculate_date_range(days_back, end_date)
    
    try:
        with _get_client(api_key) as client:
            logger.info(f"Fetching network data for {network} network ({interval} intervals)")
            
            response = client.get_network_data(
                network_code=network,
                metrics=[
                    DataMetric.POWER,
                    DataMetric.ENERGY,
                    DataMetric.MARKET_VALUE,
                    DataMetric.EMISSIONS
                ],
                interval=interval,
                date_start=start_date,
                date_end=end_date,
                primary_grouping=primary_grouping,
                secondary_grouping=secondary_grouping,
            )
            
            # Convert to Spark DataFrame using native to_pandas method
            pd_df = response.to_pandas()
            units = response.get_metric_units()
            spark = _get_spark()
            spark_df = spark.createDataFrame(pd_df)
            
            # Rename columns using PySpark operations
            power_unit = units.get("power", "")
            energy_unit = units.get("energy", "")
            emissions_unit = units.get("emissions", "")
            
            if power_unit:
                spark_df = spark_df.withColumnRenamed('power', f'power_{power_unit}')
            if energy_unit:
                spark_df = spark_df.withColumnRenamed('energy', f'energy_{energy_unit}')
            if emissions_unit:
                spark_df = spark_df.withColumnRenamed('emissions', f'emissions_{emissions_unit}')
            
            # Always rename market_value
            spark_df = spark_df.withColumnRenamed('market_value', 'market_value_aud')
            
            logger.info(f"Successfully fetched network data records")
            return spark_df
            
    except Exception as e:
        logger.error(f"Failed to fetch network data for {network}: {str(e)}")
        raise


def get_facility_data(
    network: NetworkCode,
    facility_codes: Union[str, List[str]],
    interval: DataInterval,
    days_back: int = 7,
    end_date: Optional[datetime] = None,
    api_key: Optional[str] = None
) -> Dict[str, Optional[DataFrame]]:
    """
    Fetch facility data for one or more facilities from OpenNEM API.
    
    If no API key is provided, it will be automatically retrieved from Databricks secrets.
    
    Args:
        network: Network code (NEM, WEM, AU)
        facility_codes: Single facility code or list of facility codes
        interval: Data interval (5m, 1h, 1d, etc.)
        days_back: Number of days to look back from end_date
        end_date: End date for data fetch (defaults to now)
        api_key: OpenNEM API key (optional, will be retrieved from secrets if not provided)
        
    Returns:
        Dictionary mapping facility codes to PySpark DataFrames with facility data
        
    Example:
        >>> facility_dfs = get_facility_data(
        ...     network="NEM",
        ...     facility_codes=["BAYSW", "LONSDALE"],
        ...     interval="5m",
        ...     days_back=7
        ... )
        >>> print(f"Got data for {len(facility_dfs)} facilities")

    """
    if not PYSPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required for get_facility_data. Install it with: uv add 'openelectricity[analysis]'"
        )
    
    start_date, end_date = _calculate_date_range(days_back, end_date)
    
    # Normalize facility codes to list
    if isinstance(facility_codes, str):
        facility_codes = [facility_codes]
    
    facility_dataframes = {}
    
    try:
        with _get_client(api_key) as client:
            logger.info(f"Fetching facility data for {len(facility_codes)} facilities")
            
            for facility_code in facility_codes:
                try:
                    logger.info(f"Fetching data for facility: {facility_code}")
                    
                    response = client.get_facility_data(
                        network_code=network,
                        facility_code=facility_code,
                        metrics=[
                            DataMetric.POWER,
                            DataMetric.ENERGY,
                            DataMetric.MARKET_VALUE,
                            DataMetric.EMISSIONS,
                        ],
                        interval=interval,
                        date_start=start_date,
                        date_end=end_date,
                    )
                    
                    # Convert to Spark DataFrame using native to_spark method
                    pd_df = response.to_pandas()
                    units = response.get_metric_units()
                    spark = _get_spark()
                    spark_df = spark.createDataFrame(pd_df)
                    
                    # Rename columns using PySpark operations
                    power_unit = units.get("power", "")
                    energy_unit = units.get("energy", "")
                    emissions_unit = units.get("emissions", "")
                    
                    if power_unit:
                        spark_df = spark_df.withColumnRenamed('power', f'power_{power_unit}')
                    if energy_unit:
                        spark_df = spark_df.withColumnRenamed('energy', f'energy_{energy_unit}')
                    if emissions_unit:
                        spark_df = spark_df.withColumnRenamed('emissions', f'emissions_{emissions_unit}')
                    
                    # Always rename market_value
                    spark_df = spark_df.withColumnRenamed('market_value', 'market_value_aud')
                    
                    # Add facility identifier using PySpark operations
                    spark_df = spark_df.withColumn("facility_code", lit(facility_code))
                    
                    facility_dataframes[facility_code] = spark_df
                    logger.info(f"Successfully fetched records for {facility_code}")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch data for facility {facility_code}: {str(e)}")
                    continue
            
            successful_count = len(facility_dataframes)
            logger.info(f"Successfully fetched data for {successful_count}/{len(facility_codes)} facilities")
            
            return facility_dataframes
            
    except Exception as e:
        logger.error(f"Failed to initialize facility data fetch: {str(e)}")
        raise


def get_facilities_metadata(api_key: Optional[str] = None) -> DataFrame:
    """
    Fetch facilities metadata (dimension table) from OpenNEM API.
    
    This function retrieves static facility information including
    facility names, locations, fuel types, and capacities.
    
    If no API key is provided, it will be automatically retrieved from Databricks secrets.
    
    Args:
        api_key: OpenNEM API key (optional, will be retrieved from secrets if not provided)
        
    Returns:
        PySpark DataFrame with facilities metadata
        
    Example:
        >>> facilities_df = get_facilities_metadata()
        >>> print(f"Got facilities metadata")
    """
    try:
        with _get_client(api_key) as client:
            logger.info("Fetching facilities metadata")
            
            response = client.get_facilities()
            pd_df = response.to_pandas()
            spark = _get_spark()
            spark_df = spark.createDataFrame(pd_df)
            
            logger.info(f"Successfully fetched facilities metadata")
            return spark_df
            
    except Exception as e:
        logger.error(f"Failed to fetch facilities metadata: {str(e)}")
        raise





def get_spark() -> SparkSession:
    """
    Get a Spark session that works in both Databricks and local environments.
    
    Returns:
        SparkSession: Configured Spark session
        
    Raises:
        Exception: If unable to create Spark session
    """
    return _get_spark()


def save_to_table(
    df: DataFrame,
    table_name: str,
    catalog: str,
    schema: str,
    mode: str = "append",
    **options
) -> None:
    """
    Save PySpark DataFrame to Databricks table with consistent naming and options.
    
    This function provides a standardized way to save data to tables
    with proper error handling and logging.
    
    Args:
        df: DataFrame to save
        table_name: Base table name (will be prefixed with catalog.schema)
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        mode: Write mode (append, overwrite, error, ignore)
        **options: Additional write options as keyword arguments (e.g., readChangeFeed="true")
        
    Example:
        >>> save_to_table(
        ...     df=market_df,
        ...     table_name="bronze_nem_market",
        ...     catalog="your_catalog",
        ...     schema="openelectricity",
        ...     readChangeFeed="true",
        ...     mergeSchema="true"
        ... )
        
        >>> # Or pass options as a dictionary
        >>> options = {"readChangeFeed": "true", "mergeSchema": "true"}
        >>> save_to_table(
        ...     df=market_df,
        ...     table_name="bronze_nem_market",
        ...     catalog="your_catalog",
        ...     schema="openelectricity",
        ...     **options
        ... )
    """
    # Automatically add readChangeFeed option for all tables
    options["readChangeFeed"] = "true"
    options["compression"] = "zstd"
    options["delta.columnMapping.mode"] = "name"
    
    full_table_name = f"{catalog}.{schema}.{table_name}"
    
    try:
        # Build write operation with proper Spark syntax
        writer = df.write.mode(mode).options(**options)
        
        # Save to table
        writer.saveAsTable(full_table_name)
        
        logger.info(f"Successfully saved records to {full_table_name}")
        
    except Exception as e:
        logger.error(f"Failed to save data to {full_table_name}: {str(e)}")
        raise


# Convenience functions for common table patterns

def save_market_data(
    df: DataFrame,
    network: str,
    catalog: str,
    schema: str,
    mode: str = "append"
) -> None:
    """
    Save market data to the standard market table.
    
    Args:
        df: Market data DataFrame
        network: Network name (NEM, WEM, AU)
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        mode: Write mode
    """
    table_name = f"bronze_{network.lower()}_market"
    save_to_table(df, table_name, catalog, schema, mode)


def save_network_data(
    df: DataFrame,
    network: str,
    catalog: str,
    schema: str,
    mode: str = "append"
) -> None:
    """
    Save network data to the standard network table.
    
    Args:
        df: Network data DataFrame
        network: Network name (NEM, WEM, AU)
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        mode: Write mode
    """
    table_name = f"bronze_{network.lower()}_network"
    save_to_table(df, table_name, catalog, schema, mode)


def save_facility_data(
    df: DataFrame,
    network: str,
    catalog: str,
    schema: str,
    mode: str = "append"
) -> None:
    """
    Save facility data to the standard facility table.
    
    Args:
        df: Facility data DataFrame
        network: Network name (NEM, WEM, AU)
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        mode: Write mode
    """
    table_name = f"bronze_{network}_facility_generation"
    save_to_table(df, table_name, catalog, schema, mode)
