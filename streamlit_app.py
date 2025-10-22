#!/usr/bin/env python3
"""
Streamlit Cloud Entry Point
Main application for Pune Climate Change Dashboard
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add src to path
sys.path.append('src')

# Suppress warnings for cleaner deployment
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for production
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def setup_database_if_needed():
    """Setup database with sample data if not exists"""
    db_path = "data/climate_aqi_database.db"
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    if not Path(db_path).exists():
        print("Setting up database for first run...")
        
        # Import and run database setup
        try:
            import subprocess
            subprocess.run([sys.executable, "reset_database.py"], check=True)
        except Exception as e:
            print(f"Database setup error: {e}")
            # Create minimal database as fallback
            create_minimal_database()

def create_minimal_database():
    """Create minimal database with sample data"""
    from datetime import datetime, timedelta
    import random
    
    db_path = "data/climate_aqi_database.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_historical (
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
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS air_quality_historical (
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
    
    # Add sample data for demo
    locations = ['pune_central', 'hadapsar', 'kothrud']
    base_date = datetime(2020, 1, 1)
    
    for i in range(365 * 3):  # 3 years of data
        date = base_date + timedelta(days=i)
        
        for location in locations:
            # Weather data
            temp = 25 + random.uniform(-8, 12)
            humidity = 60 + random.uniform(-30, 30)
            
            cursor.execute("""
                INSERT INTO weather_historical 
                (timestamp, location_id, temperature, humidity, pressure, wind_speed, 
                 wind_direction, precipitation, solar_radiation, uv_index, visibility, 
                 cloud_cover, data_source, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date.isoformat(), location, temp, humidity,
                1013 + random.uniform(-10, 10), 10 + random.uniform(-5, 10),
                random.uniform(0, 360), random.uniform(0, 20),
                15 + random.uniform(-5, 10), random.uniform(0, 11),
                10 + random.uniform(-5, 5), random.uniform(0, 100),
                'demo_data', 0.9
            ))
            
            # AQI data
            aqi = 80 + random.uniform(-40, 80)
            pm25 = aqi * 0.4 + random.uniform(-10, 10)
            
            cursor.execute("""
                INSERT INTO air_quality_historical 
                (timestamp, location_id, pm25, pm10, no2, so2, co, o3, aqi, 
                 dominant_pollutant, data_source, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date.isoformat(), location, max(0, pm25), max(0, pm25 * 1.5),
                random.uniform(10, 40), random.uniform(5, 20),
                random.uniform(0.5, 2), random.uniform(20, 60),
                max(0, aqi), 'PM2.5', 'demo_data', 0.9
            ))
    
    conn.commit()
    conn.close()
    print(f"Created demo database with {len(locations) * 365 * 3} records per table")

if __name__ == "__main__":
    # Setup database
    setup_database_if_needed()
    
    # Import and run the main dashboard
    try:
        from src.climate_dashboard import main
        main()
    except ImportError:
        from climate_dashboard import main
        main()