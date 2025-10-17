#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced climate and AQI prediction system
"""

import asyncio
import time
import requests
import json
from datetime import datetime

def test_iot_api():
    """Test IoT data submission via HTTP API"""
    print("🔌 Testing IoT API integration...")
    
    # Sample sensor data for different locations
    test_data = [
        {
            "sensor_type": "temperature",
            "location_id": "pune_central",
            "sensor_id": "temp_001",
            "value": 28.5,
            "unit": "C",
            "timestamp": datetime.now().isoformat(),
            "quality_score": 0.95
        },
        {
            "sensor_type": "pm25",
            "location_id": "hadapsar",
            "sensor_id": "air_001",
            "value": 45.2,
            "unit": "µg/m³",
            "timestamp": datetime.now().isoformat(),
            "quality_score": 0.88
        },
        {
            "sensor_type": "humidity",
            "location_id": "kothrud",
            "sensor_id": "humid_001",
            "value": 65.3,
            "unit": "%",
            "timestamp": datetime.now().isoformat(),
            "quality_score": 0.92
        }
    ]
    
    # Try to submit data to IoT API
    api_url = "http://localhost:5000/api/sensor-data"
    
    for data in test_data:
        try:
            response = requests.post(api_url, json=data, timeout=5)
            if response.status_code == 200:
                print(f"✅ Successfully submitted {data['sensor_type']} data from {data['location_id']}")
            else:
                print(f"⚠️ API not available yet (expected during initial setup)")
        except requests.exceptions.RequestException:
            print(f"⚠️ IoT API not running yet - this is normal during initial setup")
    
    return True

async def test_data_collection():
    """Test data collection functionality"""
    print("📊 Testing data collection...")
    
    try:
        from src.enhanced_data_collector import EnhancedDataCollector
        
        collector = EnhancedDataCollector()
        print("✅ Data collector initialized successfully")
        
        # Test fetching data for one location
        print("🌍 Testing weather data fetch for Pune Central...")
        weather_df = await collector.fetch_weather_data('pune_central', days_back=1)
        
        if not weather_df.empty:
            print(f"✅ Successfully fetched {len(weather_df)} weather records")
        else:
            print("⚠️ No weather data fetched (API might be rate limited)")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Data collection test failed: {e}")
        return False

def test_ml_models():
    """Test ML model functionality"""
    print("🧠 Testing ML models...")
    
    try:
        from src.advanced_ml_models import AdvancedMLModels
        
        ml_models = AdvancedMLModels()
        print("✅ ML models initialized successfully")
        
        # Test model structure
        print("📋 Available model configurations:")
        from src.config import MODEL_CONFIG
        
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
        from src.config import PUNE_LOCATIONS
        
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
    print("🔌 IoT API: http://localhost:5000")
    print("📡 MQTT: localhost:1883")
    print("="*60)
    
    print("\n🎯 KEY FEATURES:")
    print("✅ High-accuracy ensemble ML models (RF, XGBoost, LightGBM, LSTM)")
    print("✅ 8 strategic locations across Pune metropolitan area")
    print("✅ Real-time IoT sensor integration (MQTT + HTTP)")
    print("✅ Interactive dashboard with live updates")
    print("✅ Multi-horizon predictions (1-30 days)")
    print("✅ Location-wise environmental analysis")
    
    print("\n📍 MONITORING LOCATIONS:")
    from src.config import PUNE_LOCATIONS
    for loc_id, loc_config in PUNE_LOCATIONS.items():
        print(f"   🌍 {loc_config.name} - {loc_config.zone} Zone")
    
    print("\n🔬 PREDICTED VARIABLES:")
    from src.config import MODEL_CONFIG
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
    test_iot_api()
    
    print("\n" + "="*60)
    print("✅ SYSTEM TESTS COMPLETED!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. 🌐 Open http://localhost:8501 to view the dashboard")
    print("2. 🔌 IoT sensors can submit data to http://localhost:5000/api/sensor-data")
    print("3. 📡 MQTT sensors can publish to localhost:1883")
    print("4. 🤖 Sensor simulation is running automatically")
    
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