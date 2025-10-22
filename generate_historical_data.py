#!/usr/bin/env python3
"""
Generate Large Historical Climate Data for Pune
Creates comprehensive climate data from 2000 to current date for detailed analysis
"""

import sys
import os
sys.path.append('src')

from comprehensive_data_generator import generate_comprehensive_climate_data

def main():
    """Generate comprehensive historical climate data"""
    print("🌍 GENERATING COMPREHENSIVE HISTORICAL CLIMATE DATA")
    print("=" * 60)
    print("📊 Data Range: 2000 - 2025 (25+ years)")
    print("📍 Locations: 8 areas across Pune")
    print("🔢 Expected Records: ~75,000 weather + ~75,000 AQI")
    print("⏳ This may take a few minutes...")
    print()
    
    try:
        weather_count, aqi_count = generate_comprehensive_climate_data()
        
        print("\n" + "=" * 60)
        print("🎉 DATA GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Weather Records: {weather_count:,}")
        print(f"🏭 AQI Records: {aqi_count:,}")
        print(f"📅 Years Covered: 25+ years (2000-2025)")
        print(f"📍 Locations: 8 Pune areas")
        print(f"💾 Database: data/climate_aqi_database.db")
        print()
        print("✅ Ready for climate analysis and predictions!")
        print("🚀 Run: python run_climate_dashboard.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("💡 Please check the database schema and try again")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)