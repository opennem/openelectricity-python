#!/usr/bin/env python
"""
Script to query the OpenElectricity API and inspect actual field structure.
"""

import os
from datetime import datetime, timedelta, timezone
from openelectricity import OEClient
from openelectricity.types import DataMetric

def test_facility_api_fields():
    """Test the facility API to see what fields are actually returned."""
    
    # Get API key
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        print("❌ OPENELECTRICITY_API_KEY environment variable not set")
        print("Set it with: export OPENELECTRICITY_API_KEY='your-key-here'")
        return
    
    # Create client
    try:
        client = OEClient(api_key=api_key)
        print("✅ Client created successfully")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return
    
    # Try different parameter combinations to find what works
    test_combinations = [
        {
            "name": "Basic power only",
            "params": {
                "network_code": "NEM",
                "facility_code": "YALLOURN",
                "metrics": [DataMetric.POWER],
                "interval": "1d",
                "date_start": datetime.now(timezone.utc) - timedelta(days=1),
                "date_end": datetime.now(timezone.utc)
            }
        },
        {
            "name": "Recent data with multiple metrics",
            "params": {
                "network_code": "NEM",
                "facility_code": "YALLOURN",
                "metrics": [DataMetric.POWER, DataMetric.ENERGY, DataMetric.POWER, DataMetric.MARKET_VALUE, DataMetric.EMISSIONS],
                "interval": "1d",
                "date_start": datetime.now(timezone.utc) - timedelta(days=1),
                "date_end": datetime.now(timezone.utc)
            }
        }
    ]
    
    for i, combo in enumerate(test_combinations, 1):
        print(f"\n Test {i}: {combo['name']}")
        print("Parameters:")
        for key, value in combo['params'].items():
            print(f"   {key}: {value}")
        
        try:
            print("Making API call...")
            response = client.get_facility_data(**combo['params'])
            
            if response and response.data:
                print(f"✅ SUCCESS! Got {len(response.data)} data series")
                
                # Inspect the first data series
                first_series = response.data[0]
                print(f"📋 First data series structure:")
                print(f"   Type: {type(first_series)}")
                print(f"   Attributes: {dir(first_series)}")
                
                # Get records
                print("Converting to records...")
                records = response.to_records()
                
                if records:
                    print(f"✅ Generated {len(records)} records")
                    
                    # Inspect first record
                    first_record = records[0]
                    print(f"🔍 First record structure:")
                    print(f"   Keys: {list(first_record.keys())}")
                    print(f"   Values: {first_record}")
                    
                    # Show all fields that are present
                    print(f" All fields present in first record:")
                    for key, value in first_record.items():
                        print(f"   |-- {key}: {type(value).__name__} = {value}")
                    
                    # Try PySpark conversion
                    print("Testing PySpark conversion...")
                    try:
                        spark_df = response.to_pandas()
                        if spark_df:
                            print(spark_df.head())
                        else:
                            print("❌ PySpark conversion returned None")
                    except Exception as e:
                        print(f"❌ PySpark conversion failed: {e}")
                    
                    # We found working data, no need to try more combinations
                    break
                    
                else:
                    print("❌ No records generated")
                    
            else:
                print("❌ No data in response")
                print(f"Response: {response}")
                
        except Exception as e:
            print(f"❌ API call failed: {e}")
            
            # Try to get more specific error information
            if hasattr(e, 'status_code'):
                print(f"   HTTP Status: {e.status_code}")
            if hasattr(e, 'detail'):
                print(f"   Error Detail: {e.detail}")
            if hasattr(e, 'response'):
                print(f"   Response: {e.response}")
            
            # If it's a 400 error, try to understand why
            if hasattr(e, 'status_code') and e.status_code == 400:
                print("   🔍 400 Bad Request - checking parameter validity...")
                
                # Check if the issue might be with the facility code
                if 'facility_code' in combo['params']:
                    print(f"   💡 Facility code '{combo['params']['facility_code']}' might be invalid")
                
                # Check if the issue might be with the date range
                if 'date_start' in combo['params'] and 'date_end' in combo['params']:
                    print(f"   💡 Date range might be invalid: {combo['params']['date_start']} to {combo['params']['date_end']}")
                
                # Check if the issue might be with the interval
                if 'interval' in combo['params']:
                    print(f"    Interval '{combo['params']['interval']}' might be invalid")
        
        print("-" * 50)
    
    # Clean up
    try:
        client.close()
        print("\n🧹 Client closed")
    except:
        pass

def test_available_facilities():
    """Test to see what facilities are available."""
    
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        print("❌ No API key available for facility discovery")
        return
    
    client = OEClient(api_key=api_key)
    
    print("\n🔍 Discovering available facilities...")
    
    # Try to get a list of facilities or test common ones
    common_facilities = [
        "BAYSW", "LIDDELL", "ERARING", "BUNGALOW", "KALGOORLIE",
        "COLLINSVILLE", "GLADSTONE", "STANWELL", "TARONG", "YALLOURN"
    ]
    
    working_facilities = []
    
    for facility in common_facilities:
        print(f"   Testing {facility}...")
        try:
            response = client.get_facility_data(
                network_code="NEM",
                facility_code=facility,
                metrics=[DataMetric.POWER],
                interval="1d"
            )
            
            if response and response.data:
                print(f"   ✅ {facility}: Working")
                working_facilities.append(facility)
            else:
                print(f"   ❌ {facility}: No data")
                
        except Exception as e:
            print(f"   ❌ {facility}: {e}")
    
    if working_facilities:
        print(f"\n✅ Working facilities found: {working_facilities}")
    else:
        print("\n❌ No working facilities found")
    
    client.close()

if __name__ == "__main__":
    print(" Starting API field inspection...")
    print("=" * 50)
    
    test_facility_api_fields()
    
    print("\n" + "=" * 50)
    print("✅ API field inspection complete!")
