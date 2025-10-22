#!/usr/bin/env python3
"""
Comprehensive Climate Dashboard Launcher
Launches the complete climate change dashboard with historical data and predictions
"""

import os
import sys
import subprocess

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress other warnings
import warnings
warnings.filterwarnings('ignore')

def setup_and_run():
    """Setup data and run the comprehensive climate dashboard"""
    print("🌍 PUNE CLIMATE CHANGE DASHBOARD")
    print("=" * 50)
    
    # Check if database exists and has data
    from pathlib import Path
    db_path = Path("data/climate_aqi_database.db")
    
    if not db_path.exists():
        print("📊 Database not found. Creating with comprehensive climate data...")
        subprocess.run([sys.executable, "reset_database.py"])
    else:
        # Check if database has sufficient data
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM weather_historical")
            weather_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM air_quality_historical")
            aqi_count = cursor.fetchone()[0]
            
            conn.close()
            
            if weather_count < 1000 or aqi_count < 1000:
                print(f"📊 Insufficient data found (Weather: {weather_count}, AQI: {aqi_count})")
                print("🔄 Regenerating comprehensive climate data...")
                subprocess.run([sys.executable, "reset_database.py"])
            else:
                print(f"✅ Database ready (Weather: {weather_count:,}, AQI: {aqi_count:,} records)")
        
        except Exception as e:
            print(f"⚠️ Database check failed: {e}")
            print("🔄 Recreating database...")
            subprocess.run([sys.executable, "reset_database.py"])
    
    print("\n🚀 Starting Climate Dashboard...")
    print("📊 Features:")
    print("   • Historical climate trends (2000-2024)")
    print("   • Temperature, rainfall, and AQI analysis")
    print("   • Machine learning predictions (2025-2050)")
    print("   • Interactive visualizations")
    print("   • Climate change impact assessment")
    
    print(f"\n🌐 Dashboard will be available at: http://localhost:8501")
    print("⏳ Please wait for the system to initialize...")
    
    # Run the comprehensive climate dashboard
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        'src/climate_dashboard.py',
        '--server.headless', 'true',
        '--server.port', '8501',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    setup_and_run()