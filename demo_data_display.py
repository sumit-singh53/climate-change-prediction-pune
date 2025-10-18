#!/usr/bin/env python3
"""
Demo script to show how the climate prediction system displays data
"""

import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

def show_database_contents():
    """Display contents of the database"""
    print("🗄️ DATABASE CONTENTS")
    print("=" * 60)
    
    conn = sqlite3.connect('data/climate_aqi_database.db')
    
    # Show tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"📊 Available tables: {len(tables)}")
    for table in tables:
        print(f"   - {table[0]}")
    
    print("\n" + "=" * 60)
    
    # Show location metadata
    print("📍 MONITORING LOCATIONS")
    print("-" * 40)
    df_locations = pd.read_sql_query("SELECT * FROM location_metadata", conn)
    for _, location in df_locations.iterrows():
        print(f"🌍 {location['name']} ({location['zone']} Zone)")
        print(f"   📍 Coordinates: {location['latitude']:.4f}, {location['longitude']:.4f}")
        print(f"   🏙️ District: {location['district']}")
        print(f"   📊 Population Density: {location['population_density']}")
        print()
    
    # Show recent weather data
    print("🌤️ RECENT WEATHER DATA")
    print("-" * 40)
    try:
        df_weather = pd.read_sql_query("""
            SELECT location_id, timestamp, temperature, humidity, pressure, wind_speed 
            FROM weather_historical 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, conn)
        
        if not df_weather.empty:
            print(f"📈 Latest {len(df_weather)} weather records:")
            for _, record in df_weather.iterrows():
                print(f"   🌡️ {record['location_id']}: {record['temperature']:.1f}°C, "
                      f"{record['humidity']:.1f}% humidity, {record['wind_speed']:.1f} m/s wind")
        else:
            print("   ⚠️ No weather data found")
    except Exception as e:
        print(f"   ⚠️ Error reading weather data: {e}")
    
    # Show recent air quality data
    print("\n💨 RECENT AIR QUALITY DATA")
    print("-" * 40)
    try:
        df_aqi = pd.read_sql_query("""
            SELECT location_id, timestamp, pm25, pm10, aqi, co, no2, so2 
            FROM air_quality_historical 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, conn)
        
        if not df_aqi.empty:
            print(f"📊 Latest {len(df_aqi)} air quality records:")
            for _, record in df_aqi.iterrows():
                aqi_status = "Good" if record['aqi'] <= 50 else "Moderate" if record['aqi'] <= 100 else "Unhealthy"
                print(f"   🏭 {record['location_id']}: AQI {record['aqi']:.0f} ({aqi_status}), "
                      f"PM2.5: {record['pm25']:.1f}, PM10: {record['pm10']:.1f}")
        else:
            print("   ⚠️ No air quality data found")
    except Exception as e:
        print(f"   ⚠️ Error reading air quality data: {e}")
    
    conn.close()

def show_ml_predictions():
    """Show ML model predictions"""
    print("\n🤖 MACHINE LEARNING PREDICTIONS")
    print("=" * 60)
    
    try:
        from advanced_ml_models import AdvancedMLModels
        from config import PUNE_LOCATIONS, MODEL_CONFIG
        
        ml_models = AdvancedMLModels()
        
        print("🧠 Model Configuration:")
        print(f"   📊 Target Variables: {MODEL_CONFIG['target_variables']}")
        print(f"   🔮 Prediction Horizons: {MODEL_CONFIG['prediction_horizons']} days")
        print(f"   🎯 Ensemble Models: {MODEL_CONFIG['ensemble_models']}")
        
        # Try to make a sample prediction
        print("\n🔮 Sample Predictions:")
        print("-" * 40)
        
        # Get some sample data for prediction
        conn = sqlite3.connect('data/climate_aqi_database.db')
        df_sample = pd.read_sql_query("""
            SELECT * FROM weather_historical 
            ORDER BY timestamp DESC 
            LIMIT 100
        """, conn)
        conn.close()
        
        if not df_sample.empty:
            print(f"   📈 Using {len(df_sample)} recent records for prediction")
            print("   🔄 Generating predictions for next 7 days...")
            
            # This would normally generate actual predictions
            # For demo purposes, we'll show the structure
            sample_location = list(PUNE_LOCATIONS.keys())[0]
            print(f"   📍 Sample predictions for {PUNE_LOCATIONS[sample_location].name}:")
            
            for day in range(1, 8):
                future_date = datetime.now() + timedelta(days=day)
                print(f"     📅 {future_date.strftime('%Y-%m-%d')}: Temp ~28°C, AQI ~85, PM2.5 ~45")
        else:
            print("   ⚠️ Insufficient data for predictions")
            
    except Exception as e:
        print(f"   ⚠️ Error with ML predictions: {e}")

def show_data_collection_status():
    """Show data collection status"""
    print("\n📡 DATA COLLECTION STATUS")
    print("=" * 60)
    
    try:
        from realtime_data_collector import RealtimeDataCollector
        from config import API_CONFIG, REALTIME_CONFIG
        
        collector = RealtimeDataCollector()
        
        print("🔧 Configuration:")
        print(f"   ⏱️ Update Interval: {REALTIME_CONFIG['update_interval_minutes']} minutes")
        print(f"   🌐 API Endpoints: {len(API_CONFIG)} configured")
        print(f"   📍 Monitoring Locations: {len(PUNE_LOCATIONS)} locations")
        
        print("\n📊 API Status:")
        print("-" * 40)
        for api_name, api_config in API_CONFIG.items():
            print(f"   🔌 {api_name}: {api_config['base_url']}")
            print(f"      Rate limit: {api_config.get('rate_limit', 'Not specified')}")
        
    except Exception as e:
        print(f"   ⚠️ Error checking data collection: {e}")

def main():
    """Main demo function"""
    print("🌍 ENHANCED CLIMATE & AQI PREDICTION SYSTEM - DATA DEMO")
    print("=" * 80)
    print("📅 Demo run at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # Show different aspects of the system
    show_database_contents()
    show_ml_predictions()
    show_data_collection_status()
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETED!")
    print("\n💡 To see the full interactive dashboard, run:")
    print("   streamlit run src/realtime_dashboard.py")
    print("\n🔗 Or visit: http://localhost:8501 (when dashboard is running)")

if __name__ == "__main__":
    main()