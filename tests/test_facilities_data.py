#!/usr/bin/env python3
"""
Test script to debug the raw response from get_facilities.
"""

import os
import json
import pytest
from datetime import datetime
from dotenv import load_dotenv
from openelectricity import OEClient
from openelectricity.types import UnitStatusType, UnitFueltechType

# Load environment variables
load_dotenv()


@pytest.fixture
def client():
    """Create a client for testing."""
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        pytest.skip("OPENELECTRICITY_API_KEY environment variable not set")
    return OEClient(api_key=api_key)


@pytest.fixture
def sample_facilities_raw_response():
    """Sample raw response from get_facilities API for testing."""
    return {
        "version": "4.2.4",
        "created_at": "2025-09-04T23:38:13+10:00",
        "success": True,
        "error": None,
        "data": [
            {
                "code": "BAYSW",
                "name": "Bayswater",
                "network_id": "NEM",
                "network_region": "NSW1",
                "description": "<p>Bayswater Power Station is a bituminous (black) coal-powered thermal power station with four 660 megawatts (890,000 hp) Tokyo Shibaura Electric (Japan) steam driven turbo alternators for a combined capacity of 2,640 megawatts (3,540,000 hp). Commissioned between 1985 and 1986, the station is located 16 kilometres (10 mi) from Muswellbrook, and 28 km (17 mi) from Singleton in the Hunter Region of New South Wales, Australia.</p>",
                "units": [
                    {
                        "code": "BW02",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 660.0,
                        "emissions_factor_co2": 0.8919,
                        "data_first_seen": "1998-12-07T02:50:00+10:00",
                        "data_last_seen": "2025-09-04T23:35:00+10:00",
                        "dispatch_type": "GENERATOR"
                    },
                    {
                        "code": "BW03",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 660.0,
                        "emissions_factor_co2": 0.8919,
                        "data_first_seen": "1998-12-07T02:50:00+10:00",
                        "data_last_seen": "2025-09-04T23:35:00+10:00",
                        "dispatch_type": "GENERATOR"
                    },
                    {
                        "code": "BW01",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 660.0,
                        "emissions_factor_co2": 0.8919,
                        "data_first_seen": "1998-12-07T02:50:00+10:00",
                        "data_last_seen": "2025-09-04T23:35:00+10:00",
                        "dispatch_type": "GENERATOR"
                    },
                    {
                        "code": "BW04",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 660.0,
                        "emissions_factor_co2": 0.8919,
                        "data_first_seen": "1998-12-07T02:50:00+10:00",
                        "data_last_seen": "2025-09-04T23:35:00+10:00",
                        "dispatch_type": "GENERATOR"
                    }
                ]
            },
            {
                "code": "ERARING",
                "name": "Eraring",
                "network_id": "NEM",
                "network_region": "NSW1",
                "description": "<p>Eraring Power Station is a coal fired electricity power station with four 720 MW Toshiba steam driven turbo-alternators for a combined capacity of 2,880 MW. The station is located near the township of Dora Creek, on the western shore of Lake Macquarie, New South Wales, Australia and is owned and operated by Origin Energy. It is Australia's largest power station.</p>",
                "units": [
                    {
                        "code": "ER04",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 720.0,
                        "emissions_factor_co2": 0.892,
                        "data_first_seen": "1998-12-07T09:30:00+10:00",
                        "data_last_seen": "2025-08-28T11:15:00+10:00",
                        "dispatch_type": "GENERATOR"
                    },
                    {
                        "code": "ER03",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 720.0,
                        "emissions_factor_co2": 0.892,
                        "data_first_seen": "1998-12-07T02:50:00+10:00",
                        "data_last_seen": "2025-09-04T23:35:00+10:00",
                        "dispatch_type": "GENERATOR"
                    },
                    {
                        "code": "ER01",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 720.0,
                        "emissions_factor_co2": 0.892,
                        "data_first_seen": "1998-12-07T02:50:00+10:00",
                        "data_last_seen": "2025-09-04T23:35:00+10:00",
                        "dispatch_type": "GENERATOR"
                    },
                    {
                        "code": "ER02",
                        "fueltech_id": "coal_black",
                        "status_id": "operating",
                        "capacity_registered": 720.0,
                        "emissions_factor_co2": 0.892,
                        "data_first_seen": "1998-12-07T02:50:00+10:00",
                        "data_last_seen": "2025-08-22T01:00:00+10:00",
                        "dispatch_type": "GENERATOR"
                    }
                ]
            },
            {
                "code": "SAPHWF1",
                "name": "Sapphire",
                "network_id": "NEM",
                "network_region": "NSW1",
                "description": "<p>Sapphire Wind Farm is a wind farm in the Australian state of New South Wales. When it was built in 2018, it was the largest wind farm in New South Wales. It is in the New England region of northern New South Wales, 28 kilometres (17 mi) east of Inverell and 18 kilometres (11 mi) west of Glen Innes.</p>",
                "units": [
                    {
                        "code": "SAPHWF1",
                        "fueltech_id": "wind",
                        "status_id": "operating",
                        "capacity_registered": 270.0,
                        "data_first_seen": "2018-02-01T19:30:00+10:00",
                        "data_last_seen": "2025-09-04T21:05:00+10:00",
                        "dispatch_type": "GENERATOR"
                    }
                ]
            },
            {
                "code": "ROYALLA",
                "name": "Royalla",
                "network_id": "NEM",
                "network_region": "NSW1",
                "description": "<p>Located 23 kilometers south of the capital Canberra, at the time of completion, the Royalla Solar Farm was once the largest photovoltaic plant in Australia with 20 MW rated capacity (24 MWp) and around 82,000 solar panels installed on 41 kilometers of fixed structures.</p>",
                "units": [
                    {
                        "code": "ROYALLA1",
                        "fueltech_id": "solar_utility",
                        "status_id": "operating",
                        "capacity_registered": 20.0,
                        "data_first_seen": "2016-04-23T07:00:00+10:00",
                        "data_last_seen": "2025-09-04T23:35:00+10:00",
                        "dispatch_type": "GENERATOR"
                    }
                ]
            },
            {
                "code": "TALWA1",
                "name": "Tallawarra",
                "network_id": "NEM",
                "network_region": "NSW1",
                "description": "<p>Tallawarra Power Station is a 435-megawatt (583,000 hp) combined cycle natural gas power station in the city of Wollongong, New South Wales, Australia. Owned and operated by EnergyAustralia, the station is the first of its type in New South Wales and produces electricity for the state during periods of high demand.</p>",
                "units": [
                    {
                        "code": "TALWA1",
                        "fueltech_id": "gas_ccgt",
                        "status_id": "operating",
                        "capacity_registered": 440.0,
                        "emissions_factor_co2": 0.3718,
                        "data_first_seen": "2008-10-14T08:10:00+10:00",
                        "data_last_seen": "2025-09-01T00:45:00+10:00",
                        "dispatch_type": "GENERATOR"
                    }
                ]
            }
        ],
        "total_records": 5
    }


def test_sample_facilities_raw_response_structure(sample_facilities_raw_response):
    """Test the structure of the sample facilities raw response."""
    
    print("\n🔍 Testing Sample Facilities Raw Response Structure")
    print("=" * 60)
    
    # Basic response structure validation
    assert "version" in sample_facilities_raw_response
    assert "created_at" in sample_facilities_raw_response
    assert "success" in sample_facilities_raw_response
    assert "data" in sample_facilities_raw_response
    assert "total_records" in sample_facilities_raw_response
    
    print(f"✅ Response metadata:")
    print(f"   Version: {sample_facilities_raw_response['version']}")
    print(f"   Created at: {sample_facilities_raw_response['created_at']}")
    print(f"   Success: {sample_facilities_raw_response['success']}")
    print(f"   Total records: {sample_facilities_raw_response['total_records']}")
    print(f"   Number of facilities: {len(sample_facilities_raw_response['data'])}")
    
    # Validate each facility
    for i, facility in enumerate(sample_facilities_raw_response['data']):
        print(f"\n📋 Facility {i+1}: {facility['code']} - {facility['name']}")
        print(f"   Network: {facility['network_id']}")
        print(f"   Region: {facility['network_region']}")
        print(f"   Units: {len(facility['units'])}")
        
        # Validate facility structure
        assert "code" in facility
        assert "name" in facility
        assert "network_id" in facility
        assert "network_region" in facility
        assert "description" in facility
        assert "units" in facility
        assert isinstance(facility['units'], list)
        
        # Validate each unit
        for j, unit in enumerate(facility['units']):
            print(f"     Unit {j+1}: {unit['code']} - {unit['fueltech_id']}")
            print(f"       Status: {unit['status_id']}")
            print(f"       Capacity: {unit['capacity_registered']} MW")
            print(f"       Dispatch type: {unit['dispatch_type']}")
            
            # Validate unit structure
            assert "code" in unit
            assert "fueltech_id" in unit
            assert "status_id" in unit
            assert "capacity_registered" in unit
            assert "dispatch_type" in unit
    
    print(f"\n✅ Sample response structure validation passed!")


def test_facilities_response_parsing(sample_facilities_raw_response):
    """Test parsing the raw response into FacilityResponse objects."""
    
    print("\n🔍 Testing Facilities Response Parsing")
    print("=" * 60)
    
    from openelectricity.models.facilities import FacilityResponse
    
    # Parse the raw response
    response = FacilityResponse.model_validate(sample_facilities_raw_response)
    
    print(f"✅ Successfully parsed response:")
    print(f"   Version: {response.version}")
    print(f"   Created at: {response.created_at}")
    print(f"   Success: {response.success}")
    print(f"   Total records: {response.total_records}")
    print(f"   Number of facilities: {len(response.data)}")
    
    # Validate each facility object
    for i, facility in enumerate(response.data):
        print(f"\n📋 Facility {i+1}: {facility.code} - {facility.name}")
        print(f"   Network: {facility.network_id}")
        print(f"   Region: {facility.network_region}")
        print(f"   Units: {len(facility.units)}")
        
        # Validate facility object
        assert facility.code is not None
        assert facility.name is not None
        assert facility.network_id is not None
        assert facility.network_region is not None
        assert facility.units is not None
        assert isinstance(facility.units, list)
        
        # Validate each unit object
        for j, unit in enumerate(facility.units):
            print(f"     Unit {j+1}: {unit.code} - {unit.fueltech_id}")
            print(f"       Status: {unit.status_id}")
            print(f"       Capacity: {unit.capacity_registered} MW")
            print(f"       Dispatch type: {unit.dispatch_type}")
            
            # Validate unit object
            assert unit.code is not None
            assert unit.fueltech_id is not None
            assert unit.status_id is not None
            assert unit.capacity_registered is not None
            assert unit.dispatch_type is not None
    
    print(f"\n✅ Response parsing validation passed!")


def test_facilities_to_records_schema(sample_facilities_raw_response):
    """Test that to_records() returns records with the correct schema."""
    
    print("\n🔍 Testing Facilities to_records() Schema")
    print("=" * 60)
    
    from openelectricity.models.facilities import FacilityResponse
    
    # Parse the raw response
    response = FacilityResponse.model_validate(sample_facilities_raw_response)
    
    # Get records from to_records()
    records = response.to_records()
    
    print(f"✅ Generated {len(records)} records from to_records()")
    
    # Expected schema fields
    expected_fields = {
        "facility_code",
        "facility_name", 
        "network_id",
        "network_region",
        "description",
        "unit_code",
        "fueltech_id",
        "status_id",
        "capacity_registered",
        "emissions_factor_co2",
        "dispatch_type",
        "data_first_seen",
        "data_last_seen"
    }
    
    # Validate schema
    if records:
        first_record = records[0]
        actual_fields = set(first_record.keys())
        
        print(f"\n📋 Schema validation:")
        print(f"   Expected fields: {sorted(expected_fields)}")
        print(f"   Actual fields: {sorted(actual_fields)}")
        
        # Check that all expected fields are present
        missing_fields = expected_fields - actual_fields
        extra_fields = actual_fields - expected_fields
        
        if missing_fields:
            print(f"   ❌ Missing fields: {missing_fields}")
        if extra_fields:
            print(f"   ⚠️  Extra fields: {extra_fields}")
        
        assert missing_fields == set(), f"Missing required fields: {missing_fields}"
        
        # Validate data types for each field
        print(f"\n🔍 Data type validation:")
        for field, value in first_record.items():
            if field in ["facility_code", "facility_name", "network_id", "network_region", "description", "unit_code", "fueltech_id", "status_id", "dispatch_type"]:
                assert isinstance(value, str) or value is None, f"Field {field} should be str or None, got {type(value)}"
                print(f"   {field}: {type(value).__name__} = {str(value)[:50]}...")
            elif field in ["capacity_registered", "emissions_factor_co2"]:
                assert isinstance(value, (int, float)) or value is None, f"Field {field} should be numeric or None, got {type(value)}"
                print(f"   {field}: {type(value).__name__} = {value}")
            elif field in ["data_first_seen", "data_last_seen"]:
                assert isinstance(value, (datetime, type(None))), f"Field {field} should be datetime or None, got {type(value)}"
                print(f"   {field}: {type(value).__name__} = {value}")
        
        # Show sample records
        print(f"\n📄 Sample records:")
        for i, record in enumerate(records[:3]):  # Show first 3 records
            print(f"   Record {i+1}:")
            for field in sorted(expected_fields):
                value = record.get(field)
                if field in ["facility_code", "unit_code"]:
                    print(f"     {field}: {value}")
                elif field in ["capacity_registered", "emissions_factor_co2"]:
                    print(f"     {field}: {value}")
                else:
                    print(f"     {field}: {str(value)[:30]}...")
            print()
    
    # Validate record count matches expected
    expected_record_count = sum(len(facility.units) for facility in response.data)
    assert len(records) == expected_record_count, f"Expected {expected_record_count} records, got {len(records)}"
    
    print(f"\n✅ Schema validation passed!")
    print(f"   Total records: {len(records)}")
    print(f"   Expected records: {expected_record_count}")


def test_facilities_data_analysis(sample_facilities_raw_response):
    """Test analyzing the facilities data for insights."""
    
    print("\n🔍 Testing Facilities Data Analysis")
    print("=" * 60)
    
    # Analyze the sample data
    facilities = sample_facilities_raw_response['data']
    
    # Count by fueltech
    fueltech_counts = {}
    total_capacity = 0
    
    for facility in facilities:
        for unit in facility['units']:
            fueltech = unit['fueltech_id']
            capacity = unit['capacity_registered']
            
            if fueltech not in fueltech_counts:
                fueltech_counts[fueltech] = {'count': 0, 'capacity': 0}
            
            fueltech_counts[fueltech]['count'] += 1
            fueltech_counts[fueltech]['capacity'] += capacity
            total_capacity += capacity
    
    print(f"📊 Analysis Results:")
    print(f"   Total facilities: {len(facilities)}")
    print(f"   Total units: {sum(len(f['units']) for f in facilities)}")
    print(f"   Total capacity: {total_capacity:.1f} MW")
    
    print(f"\n🔧 Fueltech breakdown:")
    for fueltech, data in fueltech_counts.items():
        print(f"   {fueltech}: {data['count']} units, {data['capacity']:.1f} MW")
    
    # Count by region
    region_counts = {}
    for facility in facilities:
        region = facility['network_region']
        region_counts[region] = region_counts.get(region, 0) + 1
    
    print(f"\n🌍 Region breakdown:")
    for region, count in region_counts.items():
        print(f"   {region}: {count} facilities")
    
    # Count by status
    status_counts = {}
    for facility in facilities:
        for unit in facility['units']:
            status = unit['status_id']
            status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\n📈 Status breakdown:")
    for status, count in status_counts.items():
        print(f"   {status}: {count} units")
    
    print(f"\n✅ Data analysis completed!")


def test_facilities_to_pandas_schema(sample_facilities_raw_response):
    """Test that to_pandas() returns the same schema as to_records()."""
    
    print("\n🔍 Testing Facilities to_pandas() Schema")
    print("=" * 60)
    
    from openelectricity.models.facilities import FacilityResponse
    
    # Parse the raw response
    response = FacilityResponse.model_validate(sample_facilities_raw_response)
    
    # Get records from to_records()
    records = response.to_records()
    
    # Get DataFrame from to_pandas()
    df = response.to_pandas()
    
    print(f"✅ Generated {len(records)} records and {len(df)} DataFrame rows")
    
    # Expected schema fields
    expected_fields = {
        "facility_code",
        "facility_name", 
        "network_id",
        "network_region",
        "description",
        "unit_code",
        "fueltech_id",
        "status_id",
        "capacity_registered",
        "emissions_factor_co2",
        "dispatch_type",
        "data_first_seen",
        "data_last_seen"
    }
    
    # Validate DataFrame schema
    if not df.empty:
        actual_fields = set(df.columns)
        
        print(f"\n📋 DataFrame schema validation:")
        print(f"   Expected fields: {sorted(expected_fields)}")
        print(f"   Actual fields: {sorted(actual_fields)}")
        
        # Check that all expected fields are present
        missing_fields = expected_fields - actual_fields
        extra_fields = actual_fields - expected_fields
        
        if missing_fields:
            print(f"   ❌ Missing fields: {missing_fields}")
        if extra_fields:
            print(f"   ⚠️  Extra fields: {extra_fields}")
        
        assert missing_fields == set(), f"Missing required fields: {missing_fields}"
        
        # Validate data types for each field
        print(f"\n🔍 DataFrame data type validation:")
        for field in sorted(expected_fields):
            dtype = str(df[field].dtype)
            if field in ["facility_code", "facility_name", "network_id", "network_region", "description", "unit_code", "fueltech_id", "status_id", "dispatch_type"]:
                assert "object" in dtype, f"Field {field} should be object dtype, got {dtype}"
                print(f"   {field}: {dtype}")
            elif field in ["capacity_registered", "emissions_factor_co2"]:
                assert "float" in dtype, f"Field {field} should be float dtype, got {dtype}"
                print(f"   {field}: {dtype}")
            elif field in ["data_first_seen", "data_last_seen"]:
                # Datetime fields can be object or datetime64
                assert "object" in dtype or "datetime" in dtype, f"Field {field} should be object or datetime dtype, got {dtype}"
                print(f"   {field}: {dtype}")
        
        # Show sample DataFrame rows
        print(f"\n📄 Sample DataFrame rows:")
        for i in range(min(3, len(df))):
            print(f"   Row {i+1}:")
            for field in sorted(expected_fields):
                value = df.iloc[i][field]
                if field in ["facility_code", "unit_code"]:
                    print(f"     {field}: {value}")
                elif field in ["capacity_registered", "emissions_factor_co2"]:
                    print(f"     {field}: {value}")
                else:
                    print(f"     {field}: {str(value)[:30]}...")
            print()
    
    # Validate that records and DataFrame have same data
    assert len(records) == len(df), f"Records count ({len(records)}) doesn't match DataFrame rows ({len(df)})"
    
    # Compare first few records
    if records and not df.empty:
        print(f"\n🔍 Comparing records vs DataFrame data:")
        for i in range(min(3, len(records))):
            record = records[i]
            df_row = df.iloc[i]
            
            print(f"   Record {i+1} comparison:")
            for field in sorted(expected_fields):
                record_value = record.get(field)
                df_value = df_row[field]
                
                if field in ["facility_code", "unit_code"]:
                    print(f"     {field}: {record_value} == {df_value}")
                    assert record_value == df_value, f"Value mismatch for {field}: {record_value} != {df_value}"
                elif field in ["capacity_registered", "emissions_factor_co2"]:
                    print(f"     {field}: {record_value} == {df_value}")
                    assert record_value == df_value, f"Value mismatch for {field}: {record_value} != {df_value}"
    
    print(f"\n✅ DataFrame schema validation passed!")
    print(f"   Total records: {len(records)}")
    print(f"   DataFrame rows: {len(df)}")


def test_facilities_unit_splitting(sample_facilities_raw_response):
    """Test that each unit gets its own row with facility information duplicated."""
    
    print("\n🔍 Testing Facilities Unit Splitting")
    print("=" * 60)
    
    from openelectricity.models.facilities import FacilityResponse
    
    # Parse the raw response
    response = FacilityResponse.model_validate(sample_facilities_raw_response)
    
    # Get records from to_records()
    records = response.to_records()
    
    print(f"✅ Generated {len(records)} records from {len(response.data)} facilities")
    
    # Count units per facility
    facility_unit_counts = {}
    for facility in response.data:
        facility_unit_counts[facility.code] = len(facility.units)
    
    print(f"\n📊 Facility breakdown:")
    for facility_code, unit_count in facility_unit_counts.items():
        print(f"   {facility_code}: {unit_count} units")
    
    # Verify that each unit gets its own record
    print(f"\n📄 Unit splitting verification:")
    current_facility = None
    unit_count = 0
    
    for i, record in enumerate(records):
        facility_code = record['facility_code']
        unit_code = record['unit_code']
        
        if facility_code != current_facility:
            if current_facility:
                print(f"   {current_facility}: {unit_count} units processed")
            current_facility = facility_code
            unit_count = 0
        
        unit_count += 1
        print(f"   Record {i+1}: {facility_code} -> {unit_code}")
    
    if current_facility:
        print(f"   {current_facility}: {unit_count} units processed")
    
    # Verify record count matches expected
    expected_records = sum(len(facility.units) for facility in response.data)
    assert len(records) == expected_records, f"Expected {expected_records} records, got {len(records)}"
    
    # Show detailed example for ERARING facility
    print(f"\n🔍 Detailed example - ERARING facility:")
    eraring_records = [r for r in records if r['facility_code'] == 'ERARING']
    
    if eraring_records:
        print(f"   ERARING has {len(eraring_records)} units:")
        for i, record in enumerate(eraring_records):
            print(f"     Unit {i+1}: {record['unit_code']}")
            print(f"       Facility: {record['facility_code']} ({record['facility_name']})")
            print(f"       Fueltech: {record['fueltech_id']}")
            print(f"       Capacity: {record['capacity_registered']} MW")
            print(f"       Region: {record['network_region']}")
            print()
    
    print(f"\n✅ Unit splitting validation passed!")
    print(f"   Total facilities: {len(response.data)}")
    print(f"   Total units: {expected_records}")
    print(f"   Total records: {len(records)}")


def test_facilities_pandas_dataframe_output(sample_facilities_raw_response):
    """Test and print the pandas DataFrame output to show the expected structure."""
    
    print("\n🔍 Testing Facilities Pandas DataFrame Output")
    print("=" * 60)
    
    from openelectricity.models.facilities import FacilityResponse
    import pandas as pd
    
    # Parse the raw response
    response = FacilityResponse.model_validate(sample_facilities_raw_response)
    
    # Get DataFrame from to_pandas()
    df = response.to_pandas()
    
    print(f"✅ Generated pandas DataFrame:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Data types:")
    for col in df.columns:
        print(f"     {col}: {df[col].dtype}")
    
    print(f"\n📋 Full DataFrame:")
    print(df.to_string(index=False))
    
    # Show a cleaner version with truncated description
    print(f"\n📋 Clean DataFrame (truncated description):")
    df_clean = df.copy()
    df_clean['description'] = df_clean['description'].str[:50] + "..."
    print(df_clean.to_string(index=False))
    
    # Show expected output structure
    print(f"\n🎯 Expected Output Structure:")
    print("Each row should represent one unit with facility info duplicated:")
    print("Row 1: BAYSW facility, BW02 unit")
    print("Row 2: BAYSW facility, BW03 unit") 
    print("Row 3: BAYSW facility, BW01 unit")
    print("Row 4: BAYSW facility, BW04 unit")
    print("Row 5: ERARING facility, ER04 unit")
    print("Row 6: ERARING facility, ER03 unit")
    print("Row 7: ERARING facility, ER01 unit")
    print("Row 8: ERARING facility, ER02 unit")
    print("Row 9: SAPHWF1 facility, SAPHWF1 unit")
    print("Row 10: ROYALLA facility, ROYALLA1 unit")
    print("Row 11: TALWA1 facility, TALWA1 unit")
    
    # Assertions to verify the structure
    print(f"\n🔍 Assertions:")
    
    # Check DataFrame shape
    expected_rows = sum(len(facility.units) for facility in response.data)
    expected_cols = 13  # The 13 fields in our schema
    assert df.shape == (expected_rows, expected_cols), f"Expected shape ({expected_rows}, {expected_cols}), got {df.shape}"
    print(f"   ✅ Shape: {df.shape} (correct)")
    
    # Check columns
    expected_columns = {
        "facility_code", "facility_name", "network_id", "network_region", "description",
        "unit_code", "fueltech_id", "status_id", "capacity_registered", "emissions_factor_co2", "dispatch_type",
        "data_first_seen", "data_last_seen"
    }
    actual_columns = set(df.columns)
    assert actual_columns == expected_columns, f"Expected columns {expected_columns}, got {actual_columns}"
    print(f"   ✅ Columns: {sorted(actual_columns)} (correct)")
    
    # Check specific values for ERARING facility
    eraring_rows = df[df['facility_code'] == 'ERARING']
    assert len(eraring_rows) == 4, f"Expected 4 ERARING units, got {len(eraring_rows)}"
    print(f"   ✅ ERARING has {len(eraring_rows)} units (correct)")
    
    # Check that facility info is duplicated correctly
    for _, row in eraring_rows.iterrows():
        assert row['facility_code'] == 'ERARING'
        assert row['facility_name'] == 'Eraring'
        assert row['network_id'] == 'NEM'
        assert row['network_region'] == 'NSW1'
    print(f"   ✅ ERARING facility info duplicated correctly")
    
    # Check unit codes for ERARING
    eraring_unit_codes = set(eraring_rows['unit_code'])
    expected_eraring_units = {'ER04', 'ER03', 'ER01', 'ER02'}
    assert eraring_unit_codes == expected_eraring_units, f"Expected ERARING units {expected_eraring_units}, got {eraring_unit_codes}"
    print(f"   ✅ ERARING unit codes: {sorted(eraring_unit_codes)} (correct)")
    
    # Check data types
    string_columns = {"facility_code", "facility_name", "network_id", "network_region", "description", "unit_code", "fueltech_id", "status_id", "dispatch_type"}
    float_columns = {"capacity_registered", "emissions_factor_co2"}
    
    for col in string_columns:
        assert str(df[col].dtype) == 'object', f"Column {col} should be object dtype, got {df[col].dtype}"
    print(f"   ✅ String columns have correct dtype")
    
    for col in float_columns:
        assert 'float' in str(df[col].dtype), f"Column {col} should be float dtype, got {df[col].dtype}"
    print(f"   ✅ Float columns have correct dtype")
    
    # Show summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Total facilities: {df['facility_code'].nunique()}")
    print(f"   Total units: {len(df)}")
    print(f"   Total capacity: {df['capacity_registered'].sum():.1f} MW")
    
    # Fueltech breakdown
    fueltech_counts = df['fueltech_id'].value_counts()
    print(f"   Fueltech breakdown:")
    for fueltech, count in fueltech_counts.items():
        capacity = df[df['fueltech_id'] == fueltech]['capacity_registered'].sum()
        print(f"     {fueltech}: {count} units, {capacity:.1f} MW")
    
    print(f"\n✅ All assertions passed!")
    print(f"   DataFrame structure is correct")
    print(f"   Each unit gets its own row with facility info duplicated")