#!/usr/bin/env python3
"""
Reset Database Script
Clears and recreates the database with correct schema
"""

import os
import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from config import DATABASE_CONFIG

def reset_database():
    """Reset the database with clean schema"""
    db_path = DATABASE_CONFIG["sqlite_path"]
    
    print(f"🗄️ Resetting database: {db_path}")
    
    # Create data directory if it doesn't exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
        print("✅ Removed existing database")
    
    # Create new database with correct schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables with correct schema
    print("📊 Creating database tables...")
    
    # Location metadata table
    cursor.execute("""
        CREATE TABLE location_metadata (
            location_id TEXT PRIMARY KEY,
            name TEXT,
            latitude REAL,
            longitude REAL,
            district TEXT,
            zone TEXT,
            elevation REAL,
            population_density REAL,
            last_updated DATETIME
        )
    """)
    
    # Historical weather data table (comprehensive schema)
    cursor.execute("""
        CREATE TABLE weather_historical (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            location_id TEXT,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            wind_speed REAL,
            wind_direction REAL,
            precipitation REAL,
            solar_radiation REAL,
            uv_index REAL,
            visibility REAL,
            cloud_cover REAL,
            data_source TEXT,
            quality_score REAL
        )
    """)
    
    # Real-time weather data table (same schema for compatibility)
    cursor.execute("""
        CREATE TABLE realtime_weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            location_id TEXT,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            wind_speed REAL,
            wind_direction REAL,
            precipitation REAL,
            solar_radiation REAL,
            uv_index REAL,
            visibility REAL,
            cloud_cover REAL,
            data_source TEXT,
            quality_score REAL
        )
    """)
    
    # Historical air quality data table (comprehensive schema)
    cursor.execute("""
        CREATE TABLE air_quality_historical (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            location_id TEXT,
            pm25 REAL,
            pm10 REAL,
            no2 REAL,
            so2 REAL,
            co REAL,
            o3 REAL,
            aqi REAL,
            dominant_pollutant TEXT,
            data_source TEXT,
            quality_score REAL
        )
    """)
    
    # Real-time air quality data table (same schema for compatibility)
    cursor.execute("""
        CREATE TABLE realtime_air_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            location_id TEXT,
            pm25 REAL,
            pm10 REAL,
            no2 REAL,
            so2 REAL,
            co REAL,
            o3 REAL,
            aqi REAL,
            dominant_pollutant TEXT,
            data_source TEXT,
            quality_score REAL
        )
    """)
    
    # Data collection log table
    cursor.execute("""
        CREATE TABLE collection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            location_id TEXT,
            data_type TEXT,
            status TEXT,
            records_collected INTEGER,
            error_message TEXT
        )
    """)
    
    print("📊 Generating comprehensive historical climate data...")
    
    # Import and run the comprehensive data generator
    import sys
    sys.path.append('src')
    
    try:
        from comprehensive_data_generator import generate_comprehensive_climate_data
        weather_count, aqi_count = generate_comprehensive_climate_data()
        print(f"✅ Generated {weather_count:,} weather records and {aqi_count:,} AQI records")
    except Exception as e:
        print(f"⚠️ Error generating comprehensive data: {e}")
        print("📊 Adding basic sample data instead...")
        
        from datetime import datetime, timedelta
        import random
        
        # Generate sample weather data for the last 24 hours as fallback
        base_time = datetime.now()
        for i in range(24):
            timestamp = base_time - timedelta(hours=i)
            
            # Realistic Pune weather data
            temp = 25 + random.uniform(-5, 10)  # 20-35°C range
            humidity = 60 + random.uniform(-20, 30)  # 40-90% range
            pressure = 1013 + random.uniform(-10, 10)  # Normal atmospheric pressure
            wind_speed = 5 + random.uniform(0, 15)  # 5-20 km/h
            
            cursor.execute("""
                INSERT INTO weather_historical 
                (timestamp, location_id, temperature, humidity, pressure, wind_speed, 
                 wind_direction, precipitation, solar_radiation, uv_index, visibility, 
                 cloud_cover, data_source, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat(),
                'pune_central',
                temp,
                humidity,
                pressure,
                wind_speed,
                random.uniform(0, 360),
                random.uniform(0, 5),
                random.uniform(10, 25),
                random.uniform(0, 11),
                random.uniform(5, 15),
                random.uniform(0, 100),
                'sample_data',
                0.95
            ))
        
        # Generate sample air quality data
        for i in range(24):
            timestamp = base_time - timedelta(hours=i)
            
            # Realistic Pune AQI data
            pm25 = 30 + random.uniform(0, 70)  # PM2.5 levels
            pm10 = pm25 * 1.5 + random.uniform(0, 20)  # PM10 typically higher
            aqi = min(500, max(0, pm25 * 2 + random.uniform(-20, 40)))  # Rough AQI calculation
            
            cursor.execute("""
                INSERT INTO air_quality_historical 
                (timestamp, location_id, pm25, pm10, no2, so2, co, o3, aqi, 
                 dominant_pollutant, data_source, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat(),
                'pune_central',
                pm25,
                pm10,
                random.uniform(10, 50),  # NO2
                random.uniform(5, 25),   # SO2
                random.uniform(0.5, 2.0), # CO
                random.uniform(20, 80),  # O3
                aqi,
                'PM2.5' if pm25 > pm10/2 else 'PM10',
                'sample_data',
                0.95
            ))
    
    conn.commit()
    conn.close()
    
    print("✅ Database reset completed successfully!")
    print(f"📍 Database location: {db_path}")
    print(f"📊 Added sample data: 24 hours of weather and air quality records")

if __name__ == "__main__":
    reset_database()