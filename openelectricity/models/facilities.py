"""
Facility models for the OpenElectricity API.

This module contains models related to facility data and responses.
"""

from datetime import datetime
import datetime as dt
from typing import Any

from pydantic import BaseModel, Field

from openelectricity.models.base import APIResponse
from openelectricity.types import NetworkCode, UnitFueltechType, UnitStatusType


def convert_field_value(key: str, value):
    """
    Convert field values with appropriate types for Spark compatibility.
    
    Args:
        key: Field name to help determine appropriate conversion
        value: Field value to convert
        
    Returns:
        Converted value optimized for Spark
    """
    if value is None:
        return None
    elif hasattr(value, 'value'):  # Enum objects
        return str(value)
    elif hasattr(value, 'isoformat'):  # Datetime objects
        # Convert timezone-aware datetime to UTC for TimestampType compatibility
        if hasattr(value, 'tzinfo') and value.tzinfo is not None:
            # Convert timezone-aware datetime to UTC
            return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        else:
            return value  # Already naive datetime, assume UTC
    elif isinstance(value, bool):
        return value  # Keep booleans
    elif isinstance(value, (int, float)) and key in ['capacity_registered', 'emissions_factor_co2']:
        return float(value)  # Keep numeric fields as numbers
    elif isinstance(value, (int, float)):
        return value  # Keep other numbers as-is
    else:
        return str(value)  # Convert everything else to string for safety


class FacilityUnit(BaseModel):
    """A unit within a facility."""

    code: str = Field(..., description="Unit code")
    fueltech_id: UnitFueltechType = Field(..., description="Fuel technology type")
    status_id: UnitStatusType = Field(..., description="Unit status")
    capacity_registered: float = Field(..., description="Registered capacity in MW")
    capacity_maximum: float | None = Field(None, description="Maximum capacity in MW")
    capacity_storage: float | None = Field(None, description="Storage capacity in MWh")
    emissions_factor_co2: float | None = Field(None, description="CO2 emissions factor")
    data_first_seen: datetime | None = Field(None, description="When data was first seen for this unit")
    data_last_seen: datetime | None = Field(None, description="When data was last seen for this unit")
    dispatch_type: str = Field(..., description="Dispatch type")


class FacilityLocation(BaseModel):
    """Location coordinates for a facility."""

    lat: float = Field(..., description="Latitude")
    lng: float = Field(..., description="Longitude")


class Facility(BaseModel):
    """A facility in the OpenElectricity system."""

    code: str = Field(..., description="Facility code")
    name: str = Field(..., description="Facility name")
    network_id: NetworkCode = Field(..., description="Network code")
    network_region: str = Field(..., description="Network region")
    description: str | None = Field(None, description="Facility description")
    npi_id: str | None = Field(None, description="NPI facility ID")
    location: FacilityLocation | None = Field(None, description="Facility location coordinates")
    units: list[FacilityUnit] = Field(..., description="Units within the facility")


class FacilityResponse(APIResponse[Facility]):
    """Response model for facility endpoints."""

    data: list[Facility]

    def to_records(self) -> list[dict[str, Any]]:
        """
        Convert facility data into a list of records suitable for data analysis.
        
        Each record represents a unit within a facility, with facility information
        flattened into each unit record.
        
        Returns:
            List of dictionaries with the following schema:
            - facility_code: str
            - facility_name: str  
            - network_id: str
            - network_region: str
            - description: str
            - unit_code: str
            - fueltech_id: str
            - status_id: str
            - capacity_registered: float
            - emissions_factor_co2: float
            - dispatch_type: str
            - data_first_seen: datetime
            - data_last_seen: datetime
        """
        if not self.data:
            return []

        records = []
        
        for facility in self.data:
            # Convert facility to dict
            facility_dict = facility.model_dump()
            
            # Get facility-level fields
            facility_code = facility_dict.get('code')
            facility_name = facility_dict.get('name')
            network_id = facility_dict.get('network_id')
            network_region = facility_dict.get('network_region')
            description = facility_dict.get('description')
            
            # Process each unit in the facility
            units = facility_dict.get('units', [])
            for unit in units:
                # Convert unit to dict (handle both Pydantic models and dicts)
                if hasattr(unit, 'model_dump'):
                    unit_dict = unit.model_dump()
                else:
                    unit_dict = unit  # Already a dict
                
                # Create record with specified schema
                fueltech_value = unit_dict.get('fueltech_id')
                if hasattr(fueltech_value, 'value'):
                    fueltech_value = fueltech_value.value
                elif fueltech_value is not None:
                    fueltech_value = str(fueltech_value)
                
                status_value = unit_dict.get('status_id')
                if hasattr(status_value, 'value'):
                    status_value = status_value.value
                elif status_value is not None:
                    status_value = str(status_value)
                
                record = {
                    "facility_code": facility_code,
                    "facility_name": facility_name,
                    "network_id": network_id,
                    "network_region": network_region,
                    "description": description,
                    "unit_code": unit_dict.get('code'),
                    "fueltech_id": fueltech_value,
                    "status_id": status_value,
                    "capacity_registered": unit_dict.get('capacity_registered'),
                    "emissions_factor_co2": unit_dict.get('emissions_factor_co2'),
                    "dispatch_type": unit_dict.get('dispatch_type'),
                    "data_first_seen": unit_dict.get('data_first_seen'),
                    "data_last_seen": unit_dict.get('data_last_seen')
                }
                
                records.append(record)
        
        return records

    def to_pyspark(self, spark_session=None, app_name: str = "OpenElectricity") -> "Optional['DataFrame']":  # noqa: F821
        """
        Convert facility data into a PySpark DataFrame.

        Args:
            spark_session: Optional PySpark session. If not provided, will try to create one.
            app_name: Name for the Spark application if creating a new session.

        Returns:
            A PySpark DataFrame containing the facility data, or None if PySpark is not available
        """
        try:
            from openelectricity.spark_utils import create_spark_dataframe
            
            # Convert facilities to list of dictionaries
            if not self.data:
                return None
            
            # Debug logging to understand data structure
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Converting {len(self.data)} facilities to PySpark DataFrame")
            if self.data:
                logger.debug(f"First facility type: {type(self.data[0])}")
                if hasattr(self.data[0], 'units'):
                    logger.debug(f"First facility units type: {type(self.data[0].units)}")
                    if self.data[0].units:
                        logger.debug(f"First unit type: {type(self.data[0].units[0])}")
                
            # Convert each facility to dict, handling nested units
            records = []
            for i, facility in enumerate(self.data):
                try:
                    # Convert facility to dict
                    facility_dict = facility.model_dump()
                    
                    # Handle units - create separate records for each unit
                    units = facility_dict.get('units', [])
                    if units and isinstance(units, list):
                        for j, unit in enumerate(units):
                            try:
                                # Create combined record
                                record = {}
                                
                                # Add facility fields (excluding units) with proper type preservation
                                for key, value in facility_dict.items():
                                    if key != 'units':
                                        record[key] = convert_field_value(key, value)
                                
                                # Add unit fields with proper type preservation
                                for key, value in unit.items():
                                    record[key] = convert_field_value(key, value)
                                
                                records.append(record)
                                
                            except Exception as unit_error:
                                logger.warning(f"Error processing unit {j} of facility {i}: {unit_error}")
                                continue
                    else:
                        # No units, just add facility data
                        record = {}
                        for key, value in facility_dict.items():
                            if key != 'units':
                                record[key] = convert_field_value(key, value)
                        records.append(record)
                        
                except Exception as facility_error:
                    logger.warning(f"Error processing facility {i}: {facility_error}")
                    continue
            
            # Debug: Check if we have any records and their structure
            logger.debug(f"Created {len(records)} records for PySpark conversion")
            if records:
                logger.debug(f"First record keys: {list(records[0].keys())}")
                logger.debug(f"First record sample: {str(records[0])[:200]}...")
            
            # Try to create DataFrame using predefined schema optimized for facilities
            try:
                if spark_session is None:
                    from openelectricity.spark_utils import get_spark_session
                    spark_session = get_spark_session()
                
                # Use predefined schema aligned with Pydantic models for better performance
                from openelectricity.spark_utils import create_facilities_flattened_schema
                
                facilities_schema = create_facilities_flattened_schema()
                
                logger.debug(f"Creating PySpark DataFrame with {len(records)} records using predefined schema")
                logger.debug(f"Schema aligned with Pydantic models: {facilities_schema}")
                
                # Create DataFrame with predefined schema
                df = spark_session.createDataFrame(records, schema=facilities_schema)
                logger.debug(f"Successfully created PySpark DataFrame with {len(records)} records")
                return df
                
            except Exception as spark_error:
                logger.error(f"Error creating PySpark DataFrame: {spark_error}")
                import traceback
                logger.debug(f"Full error traceback: {traceback.format_exc()}")
                logger.info("Falling back to None - use to_pandas() for facilities data")
                return None
            
        except ImportError:
            # Log warning but don't raise error to maintain compatibility
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("PySpark not available. Install with: uv add 'openelectricity[analysis]'")
            return None
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error converting to PySpark DataFrame: {e}")
            return None

    def to_pandas(self) -> "pd.DataFrame":  # noqa: F821
        """
        Convert facility data into a Pandas DataFrame.

        Returns:
            A Pandas DataFrame containing the facility data with the same schema as to_records():
            - facility_code: str
            - facility_name: str  
            - network_id: str
            - network_region: str
            - description: str
            - unit_code: str
            - fueltech_id: str
            - capacity_registered: float
            - emissions_factor_co2: float
            - dispatch_type: str
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required for DataFrame conversion. Install it with: uv add 'openelectricity[analysis]'"
            ) from None

        # Use to_records() to ensure consistent schema
        records = self.to_records()
        
        return pd.DataFrame(records)
