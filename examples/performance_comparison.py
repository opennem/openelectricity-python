#!/usr/bin/env python3
"""
Performance comparison between the old async-based synchronous client and the new requests-based client.

This example demonstrates the performance improvements and simplicity of the new synchronous client.
"""

import os
import time
from datetime import datetime, timedelta
from openelectricity.client import OEClient, LegacyOEClient
from openelectricity.types import NetworkCode, DataMetric, DataInterval
from openelectricity.settings_schema import settings

def benchmark_client(client_class, name, api_key):
    """Benchmark a specific client class."""
    print(f"\n🔧 Benchmarking {name}...")
    
    start_time = time.time()
    
    try:
        with client_class(api_key=api_key) as client:
            # Test 1: Get facilities
            t1_start = time.time()
            facilities = client.get_facilities()
            t1_end = time.time()
            print(f"   ✅ get_facilities: {(t1_end - t1_start) * 1000:.2f}ms")
            
            # Test 2: Get network data
            t2_start = time.time()
            network_data = client.get_network_data(
                network_code="NEM",
                metrics=[DataMetric.ENERGY],
                interval="5m"
            )
            t2_end = time.time()
            print(f"   ✅ get_network_data: {(t2_end - t2_start) * 1000:.2f}ms")
            
            # Test 3: Get current user
            t3_start = time.time()
            user = client.get_current_user()
            t3_end = time.time()
            print(f"   ✅ get_current_user: {(t3_end - t3_start) * 1000:.2f}ms")
            
        total_time = time.time() - start_time
        print(f"   📊 Total time: {total_time * 1000:.2f}ms")
        
        return total_time, len(facilities.data), len(network_data.data)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, 0, 0

def main():
    """Main comparison function."""
    
    # Get API key from settings (which loads from .env file)
    api_key = settings.api_key
    if not api_key:
        print("❌ No API key found in .env file or environment")
        return
    
    print("🚀 Performance Comparison: Legacy vs New Synchronous Client")
    print("=" * 60)
    
    # Benchmark the new synchronous client
    new_time, new_facilities, new_data = benchmark_client(
        OEClient, "New Synchronous Client (requests)", api_key
    )
    
    # Benchmark the legacy client
    legacy_time, legacy_facilities, legacy_data = benchmark_client(
        LegacyOEClient, "Legacy Client (aiohttp + asyncio.run)", api_key
    )
    
    # Compare results
    if new_time and legacy_time:
        print("\n📈 Performance Comparison Results:")
        print("=" * 40)
        
        speedup = legacy_time / new_time
        print(f"   New Client Total Time: {new_time * 1000:.2f}ms")
        print(f"   Legacy Client Total Time: {legacy_time * 1000:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x faster")
        
        if speedup > 1:
            print(f"   🎉 New client is {speedup:.2f}x faster!")
        else:
            print(f"   ⚠️  Legacy client is {1/speedup:.2f}x faster")
        
        print(f"\n   Data Retrieved:")
        print(f"   - Facilities: {new_facilities} (new) vs {legacy_facilities} (legacy)")
        print(f"   - Network Data Points: {new_data} (new) vs {legacy_data} (legacy)")
        
        print("\n💡 Key Benefits of New Client:")
        print("   - No event loop overhead")
        print("   - Direct HTTP requests without async wrapper")
        print("   - Better connection pooling")
        print("   - Simpler error handling")
        print("   - More predictable performance")
    
    print("\n✅ Performance comparison completed!")

if __name__ == "__main__":
    main()
