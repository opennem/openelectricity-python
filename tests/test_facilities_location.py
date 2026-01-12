"""
Test location data extraction in facility models.
"""

import pytest
from datetime import datetime

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from openelectricity.models.facilities import (
    Facility,
    FacilityLocation,
    FacilityUnit,
    FacilityResponse,
)
from openelectricity.types import UnitFueltechType, UnitStatusType


@pytest.fixture
def facility_with_location():
    """Create a facility with location data for testing."""
    return Facility(
        code="TEST01",
        name="Test Power Station",
        network_id="NEM",
        network_region="NSW1",
        description="Test facility",
        location=FacilityLocation(lat=-32.393502, lng=150.953963),
        units=[
            FacilityUnit(
                code="TEST01_U1",
                fueltech_id=UnitFueltechType.COAL_BLACK,
                status_id=UnitStatusType.OPERATING,
                capacity_registered=500.0,
                dispatch_type="GENERATOR",
            ),
            FacilityUnit(
                code="TEST01_U2",
                fueltech_id=UnitFueltechType.COAL_BLACK,
                status_id=UnitStatusType.OPERATING,
                capacity_registered=500.0,
                dispatch_type="GENERATOR",
            ),
        ],
    )


@pytest.fixture
def facility_without_location():
    """Create a facility without location data for testing."""
    return Facility(
        code="TEST02",
        name="Test Facility No Location",
        network_id="NEM",
        network_region="QLD1",
        description="Test facility without location",
        location=None,
        units=[
            FacilityUnit(
                code="TEST02_U1",
                fueltech_id=UnitFueltechType.WIND,
                status_id=UnitStatusType.OPERATING,
                capacity_registered=100.0,
                dispatch_type="GENERATOR",
            ),
        ],
    )


class TestFacilityLocationExtraction:
    """Test location data extraction in facility models."""

    def test_to_records_includes_location(self, facility_with_location):
        """Test that to_records() includes latitude and longitude."""
        response = FacilityResponse(
            version="1.0",
            created_at=datetime.now(),
            data=[facility_with_location]
        )
        
        records = response.to_records()
        
        # Should have 2 records (one per unit)
        assert len(records) == 2
        
        # Both records should have location data
        for record in records:
            assert "latitude" in record
            assert "longitude" in record
            assert record["latitude"] == -32.393502
            assert record["longitude"] == 150.953963
    
    def test_to_records_handles_missing_location(self, facility_without_location):
        """Test that to_records() handles facilities without location gracefully."""
        response = FacilityResponse(
            version="1.0",
            created_at=datetime.now(),
            data=[facility_without_location]
        )
        
        records = response.to_records()
        
        # Should have 1 record (one unit)
        assert len(records) == 1
        
        # Record should have None for location fields
        record = records[0]
        assert "latitude" in record
        assert "longitude" in record
        assert record["latitude"] is None
        assert record["longitude"] is None
    
    def test_to_records_with_mixed_locations(self, facility_with_location, facility_without_location):
        """Test that to_records() handles mix of facilities with and without locations."""
        response = FacilityResponse(
            version="1.0",
            created_at=datetime.now(),
            data=[facility_with_location, facility_without_location]
        )
        
        records = response.to_records()
        
        # Should have 3 records (2 from first facility, 1 from second)
        assert len(records) == 3
        
        # First two records should have location
        assert records[0]["latitude"] == -32.393502
        assert records[0]["longitude"] == 150.953963
        assert records[1]["latitude"] == -32.393502
        assert records[1]["longitude"] == 150.953963
        
        # Third record should have None
        assert records[2]["latitude"] is None
        assert records[2]["longitude"] is None
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not installed")
    def test_to_pandas_includes_location(self, facility_with_location):
        """Test that to_pandas() includes latitude and longitude columns."""
        response = FacilityResponse(
            version="1.0",
            created_at=datetime.now(),
            data=[facility_with_location]
        )
        
        df = response.to_pandas()
        
        # Check columns exist
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        
        # Check data types
        assert df["latitude"].dtype in ["float64", "float32"]
        assert df["longitude"].dtype in ["float64", "float32"]
        
        # Check values
        assert all(df["latitude"] == -32.393502)
        assert all(df["longitude"] == 150.953963)
        
        # Check no nulls for facility with location
        assert df["latitude"].notna().all()
        assert df["longitude"].notna().all()
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not installed")
    def test_to_pandas_handles_missing_location(self, facility_without_location):
        """Test that to_pandas() handles missing location gracefully."""
        response = FacilityResponse(
            version="1.0",
            created_at=datetime.now(),
            data=[facility_without_location]
        )
        
        df = response.to_pandas()
        
        # Columns should exist
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        
        # Values should be null
        assert df["latitude"].isna().all()
        assert df["longitude"].isna().all()
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not installed")
    def test_to_pandas_location_column_order(self, facility_with_location):
        """Test that latitude and longitude appear in expected position."""
        response = FacilityResponse(
            version="1.0",
            created_at=datetime.now(),
            data=[facility_with_location]
        )
        
        df = response.to_pandas()
        columns = df.columns.tolist()
        
        # Location columns should appear after description and before unit_code
        lat_idx = columns.index("latitude")
        lng_idx = columns.index("longitude")
        desc_idx = columns.index("description")
        unit_idx = columns.index("unit_code")
        
        assert desc_idx < lat_idx < unit_idx
        assert desc_idx < lng_idx < unit_idx
        assert lat_idx < lng_idx  # latitude before longitude
    
    def test_location_object_structure(self):
        """Test that FacilityLocation model works correctly."""
        location = FacilityLocation(lat=-33.8688, lng=151.2093)
        
        assert location.lat == -33.8688
        assert location.lng == 151.2093
        
        # Test model_dump
        location_dict = location.model_dump()
        assert location_dict["lat"] == -33.8688
        assert location_dict["lng"] == 151.2093
    
    def test_facility_with_location_model_dump(self, facility_with_location):
        """Test that facility.model_dump() properly includes nested location."""
        facility_dict = facility_with_location.model_dump()
        
        assert "location" in facility_dict
        assert facility_dict["location"] is not None
        assert facility_dict["location"]["lat"] == -32.393502
        assert facility_dict["location"]["lng"] == 150.953963
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not installed")
    def test_location_data_types_in_dataframe(self, facility_with_location):
        """Test that location data maintains correct numeric types."""
        response = FacilityResponse(
            version="1.0",
            created_at=datetime.now(),
            data=[facility_with_location]
        )
        
        df = response.to_pandas()
        
        # Check that lat/lng are numeric, not string
        assert pd.api.types.is_numeric_dtype(df["latitude"])
        assert pd.api.types.is_numeric_dtype(df["longitude"])
        
        # Check values can be used in numeric operations
        lat_mean = df["latitude"].mean()
        lng_mean = df["longitude"].mean()
        
        assert isinstance(lat_mean, (float, int))
        assert isinstance(lng_mean, (float, int))
        assert lat_mean == pytest.approx(-32.393502)
        assert lng_mean == pytest.approx(150.953963)

