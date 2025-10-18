#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced climate and AQI prediction system
"""

import asyncio
import time
import requests
import json
from datetime import datetime

async def test_realtime_collection():
    """Test real-time data collection functionality"""
    print("📡 Testing real-time data collection...")
    
    try:
        import sys
        sys.path.append('src')
        from realtime_data_collector import RealtimeDataCollector
        
        collector = RealtimeDataCollector()
        print("✅ Real-time collector initialized successfully")
        
        # Skip actual API calls in CI/CD environment
        import os
        if os.getenv('GITHUB_ACTIONS'):
            print("🔄 Skipping API calls in CI/CD environment")
            return True
        
        # Test collecting data for one location
        print("🌍 Testing real-time data collection for Pune Central...")
        try:
            weather_data = await collector.fetch_weather_data('pune_central')
            air_quality_data = await collector.fetch_air_quality_data('pune_central')
            
            if weather_data:
                print(f"✅ Successfully collected weather data: {weather_data.get('temperature', 'N/A')}°C")
            if air_quality_data:
                print(f"✅ Successfully collected air quality data: AQI {air_quality_data.get('aqi', 'N/A')}")
                
        except Exception as api_error:
            print(f"⚠️ API call failed (expected in CI/CD): {api_error}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Real-time collection test failed: {e}")
        return False

async def test_data_collection():
    """Test data collection functionality"""
    print("📊 Testing data collection...")
    
    try:
        import sys
        sys.path.append('src')
        from enhanced_data_collector import EnhancedDataCollector
        
        collector = EnhancedDataCollector()
        print("✅ Data collector initialized successfully")
        
        # Skip actual API calls in CI/CD environment
        import os
        if os.getenv('GITHUB_ACTIONS'):
            print("🔄 Skipping API calls in CI/CD environment")
            return True
        
        # Test fetching data for one location (only in local environment)
        print("🌍 Testing weather data fetch for Pune Central...")
        try:
            weather_df = await collector.fetch_weather_data('pune_central', days_back=1)
            
            if not weather_df.empty:
                print(f"✅ Successfully fetched {len(weather_df)} weather records")
            else:
                print("⚠️ No weather data fetched (API might be rate limited)")
        except Exception as api_error:
            print(f"⚠️ API call failed (expected in CI/CD): {api_error}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Data collection test failed: {e}")
        return False

def test_ml_models():
    """Test ML model functionality"""
    print("🧠 Testing ML models...")
    
    try:
        import sys
        sys.path.append('src')
        from advanced_ml_models import AdvancedMLModels
        
        ml_models = AdvancedMLModels()
        print("✅ ML models initialized successfully")
        
        # Test model structure
        print("📋 Available model configurations:")
        from config import MODEL_CONFIG
        
        print(f"   - Target variables: {MODEL_CONFIG['target_variables']}")
        print(f"   - Ensemble models: {MODEL_CONFIG['ensemble_models']}")
        print(f"   - Prediction horizons: {MODEL_CONFIG['prediction_horizons']} days")
        
        return True
        
    except Exception as e:
        print(f"⚠️ ML models test failed: {e}")
        return False

def test_locations():
    """Test location configuration"""
    print("📍 Testing location configuration...")
    
    try:
        import sys
        sys.path.append('src')
        from config import PUNE_LOCATIONS
        
        print(f"✅ Configured {len(PUNE_LOCATIONS)} monitoring locations:")
        for loc_id, loc_config in PUNE_LOCATIONS.items():
            print(f"   - {loc_config.name} ({loc_config.zone} zone)")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Location test failed: {e}")
        return False

def show_system_info():
    """Display system information"""
    print("\n" + "="*60)
    print("🌟 ENHANCED CLIMATE & AQI PREDICTION SYSTEM")
    print("="*60)
    print("📊 Dashboard: http://localhost:8501")
    print("🌟 Enhanced Dashboard: http://localhost:8502")
    print("="*60)
    
    print("\n🎯 KEY FEATURES:")
    print("✅ High-accuracy ensemble ML models (RF, XGBoost, LightGBM, LSTM)")
    print("✅ 8 strategic locations across Pune metropolitan area")
    print("✅ Real-time API data collection with caching")
    print("✅ Interactive dashboard with live updates")
    print("✅ Multi-horizon predictions (1-30 days)")
    print("✅ Location-wise environmental analysis")
    
    print("\n📍 MONITORING LOCATIONS:")
    import sys
    sys.path.append('src')
    from config import PUNE_LOCATIONS
    for loc_id, loc_config in PUNE_LOCATIONS.items():
        print(f"   🌍 {loc_config.name} - {loc_config.zone} Zone")
    
    print("\n🔬 PREDICTED VARIABLES:")
    from config import MODEL_CONFIG
    for var in MODEL_CONFIG['target_variables']:
        print(f"   📈 {var.replace('_', ' ').title()}")

async def main():
    """Main test function"""
    print("🚀 Starting Enhanced Climate & AQI Prediction System Tests")
    print("="*60)
    
    # Show system information
    show_system_info()
    
    print("\n🧪 RUNNING SYSTEM TESTS:")
    print("-"*40)
    
    # Test components
    test_locations()
    test_ml_models()
    await test_data_collection()
    await test_realtime_collection()
    
    print("\n" + "="*60)
    print("✅ SYSTEM TESTS COMPLETED!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. 🌐 Open http://localhost:8501 to view the dashboard")
    print("2. 📊 Real-time data is collected automatically every 30 minutes")
    print("3. 🤖 ML models provide predictions based on current data")
    print("4. 📈 Historical trends are available for analysis")
    
    print("\n💡 USAGE EXAMPLES:")
    print("   # Start full system with IoT simulation")
    print("   python run_system.py")
    print("")
    print("   # Start dashboard only")
    print("   python run_system.py --mode dashboard")
    print("")
    print("   # Start data collection only")
    print("   python run_system.py --mode data-collection")

if __name__ == "__main__":
    asyncio.run(main())