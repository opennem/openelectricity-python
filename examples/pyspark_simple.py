#!/usr/bin/env python
"""
Simple PySpark Example with OpenElectricity

This example demonstrates the new to_pyspark functionality
that automatically handles Spark session creation for both
Databricks and local environments.

PySpark is completely optional - the SDK works without it!
"""

from openelectricity import OEClient
from openelectricity.types import MarketMetric
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def demonstrate_without_pyspark():
    """Demonstrate SDK functionality without PySpark."""
    print("📊 OpenElectricity SDK Demo (No PySpark)")
    print("=" * 50)
    print("This shows how the SDK works without PySpark installed")
    
    # Initialize the client
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        print("❌ OPENELECTRICITY_API_KEY environment variable not set")
        return
    
    client = OEClient(api_key=api_key)
    
    # Fetch market data
    print("\n📊 Fetching market data...")
    try:
        response = client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.PRICE, MarketMetric.DEMAND],
            interval="1h",
            date_start=datetime.now() - timedelta(days=1),
            date_end=datetime.now(),
            primary_grouping="network_region"
        )
        print(f"✅ Fetched {len(response.data)} time series")
        
        # Try to convert to PySpark (will return None)
        print("\n🔄 Attempting PySpark conversion...")
        spark_df = response.to_pyspark()
        
        if spark_df is None:
            print("ℹ️  PySpark not available - to_pyspark() returned None")
            print("   This is expected behavior when PySpark isn't installed")
            
            # Fall back to pandas (which is usually available)
            print("\n🔄 Falling back to pandas...")
            try:
                pandas_df = response.to_pandas()
                print("✅ Successfully created pandas DataFrame!")
                print(f"   Shape: {pandas_df.shape}")
                print(f"   Columns: {', '.join(pandas_df.columns)}")
                
                # Show sample data
                print("\n📋 Sample data:")
                print(pandas_df.head())
                
            except ImportError:
                print("ℹ️  Pandas also not available")
                print("   Raw data is still accessible via response.data")
                
        else:
            print("✅ PySpark DataFrame created successfully!")
            
    except Exception as e:
        print(f"❌ Error during data fetch: {e}")
    
    # Test facilities data
    print("\n🏭 Testing facilities data...")
    try:
        facilities_response = client.get_facilities(network_region="NSW1")
        print(f"✅ Fetched {len(facilities_response.data)} facilities")
        
        # Try PySpark conversion
        facilities_df = facilities_response.to_pyspark()
        
        if facilities_df is None:
            print("ℹ️  PySpark not available for facilities")
            
            # Try pandas fallback
            try:
                pandas_facilities = facilities_response.to_pandas()
                print("✅ Successfully created facilities pandas DataFrame!")
                print(f"   Shape: {pandas_facilities.shape}")
                print(f"   Columns: {', '.join(pandas_facilities.columns)}")
                
            except ImportError:
                print("ℹ️  Pandas not available for facilities")
                
        else:
            print("✅ PySpark facilities DataFrame created successfully!")
            
    except Exception as e:
        print(f"❌ Error during facilities fetch: {e}")


def demonstrate_with_pyspark():
    """Demonstrate SDK functionality with PySpark available."""
    print("\n🚀 OpenElectricity SDK Demo (With PySpark)")
    print("=" * 50)
    print("This shows the full PySpark functionality when available")
    
    # Check if PySpark is available
    try:
        import pyspark
        print(f"✅ PySpark {pyspark.__version__} is available")
    except ImportError:
        print("ℹ️  PySpark not available - skipping PySpark demo")
        print("   Install with: uv add 'openelectricity[analysis]' or uv add pyspark")
        return
    
    # Initialize the client
    api_key = os.getenv("OPENELECTRICITY_API_KEY")
    if not api_key:
        print("❌ OPENELECTRICITY_API_KEY environment variable not set")
        return
    
    client = OEClient(api_key=api_key)
    
    # Test Spark session management
    print("\n🔧 Testing Spark session management...")
    try:
        spark = client.get_spark_session("OpenElectricity-Demo")
        print(f"✅ Successfully created Spark session: {spark.conf.get('spark.app.name')}")
        print(f"   Spark version: {spark.version}")
        
        # Check environment type
        try:
            from databricks.connect import DatabricksSession
            print("   Environment: Databricks")
        except ImportError:
            print("   Environment: Local PySpark")
            
    except Exception as e:
        print(f"❌ Failed to create Spark session: {e}")
        return
    
    # Fetch and convert data
    print("\n📊 Fetching and converting market data...")
    try:
        response = client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.PRICE, MarketMetric.DEMAND],
            interval="1h",
            date_start=datetime.now() - timedelta(days=1),
            date_end=datetime.now(),
            primary_grouping="network_region"
        )
        print(f"✅ Fetched {len(response.data)} time series")
        
        # Convert to PySpark DataFrame
        spark_df = response.to_pyspark(spark_session=spark, app_name="OpenElectricity-Conversion")
        
        if spark_df is not None:
            print("✅ Successfully created PySpark DataFrame!")
            print(f"   Schema: {spark_df.schema}")
            print(f"   Row count: {spark_df.count()}")
            print(f"   Columns: {', '.join(spark_df.columns)}")
            
            # Show sample data
            print("\n📋 Sample data:")
            spark_df.show(5, truncate=False)
            
            # Demonstrate some PySpark operations
            print("\n🔍 PySpark Operations:")
            
            # Show data types
            print("📊 Data Types:")
            spark_df.printSchema()
            
            # Show summary statistics
            print("\n📊 Summary Statistics:")
            spark_df.describe().show()
            
            # Filter for specific region
            nsw_data = spark_df.filter(spark_df.network_region == "NSW1")
            print(f"\n🏭 NSW1 data count: {nsw_data.count()}")
            
            # Show price statistics by region
            print("\n💰 Price Statistics by Region:")
            price_stats = spark_df.groupBy("network_region").agg(
                {"price": "avg", "price": "min", "price": "max"}
            ).withColumnRenamed("avg(price)", "avg_price").withColumnRenamed("min(price)", "min_price").withColumnRenamed("max(price)", "max_price")
            price_stats.show()
            
        else:
            print("❌ Failed to create PySpark DataFrame")
            
    except Exception as e:
        print(f"❌ Error during data fetch: {e}")
    
    # Test facilities data
    print("\n🏭 Testing facilities data conversion...")
    try:
        facilities_response = client.get_facilities(network_region="NSW1")
        print(f"✅ Fetched {len(facilities_response.data)} facilities")
        
        facilities_df = facilities_response.to_pyspark(spark_session=spark, app_name="OpenElectricity-Facilities")
        
        if facilities_df is not None:
            print("✅ Successfully created facilities PySpark DataFrame!")
            print(f"   Row count: {facilities_df.count()}")
            print(f"   Columns: {', '.join(facilities_df.columns)}")
            
            # Show sample data
            print("\n📋 Sample facilities data:")
            facilities_df.show(5, truncate=False)
            
        else:
            print("❌ Failed to create facilities PySpark DataFrame")
            
    except Exception as e:
        print(f"❌ Error during facilities fetch: {e}")


def main():
    """Main function to run all demonstrations."""
    print("🎯 OpenElectricity PySpark Integration Demo")
    print("=" * 60)
    print("This demo shows how the SDK works with and without PySpark")
    print("PySpark is completely optional - the SDK works without it!")
    print()
    
    # Always demonstrate core functionality
    demonstrate_without_pyspark()
    
    # Then show PySpark features if available
    demonstrate_with_pyspark()
    
    print("\n🎉 Demo completed!")
    print("\n💡 Key takeaways:")
    print("   - PySpark is completely optional")
    print("   - SDK works seamlessly without PySpark")
    print("   - to_pyspark() returns None when PySpark unavailable")
    print("   - Graceful fallback to pandas or raw data")
    print("   - Install PySpark only when needed")
    print("\n📦 Installation options:")
    print("   - Core SDK: uv add openelectricity")
    print("   - With Analysis: uv add 'openelectricity[analysis]'")
    print("   - Just PySpark: uv add pyspark")


if __name__ == "__main__":
    main()
