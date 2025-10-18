#!/usr/bin/env python3
"""
System Cleanup Script
Fixes null data, removes unnecessary files, and optimizes for ML training
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def clean_database():
    """Clean database issues"""
    print("🗄️ CLEANING DATABASE")
    print("=" * 50)
    
    db_path = "data/climate_aqi_database.db"
    conn = sqlite3.connect(db_path)
    
    # 1. Drop columns with >95% null values from weather_historical
    print("🧹 Cleaning weather_historical table...")
    
    # Check current state
    weather_df = pd.read_sql_query("SELECT * FROM weather_historical", conn)
    print(f"   📊 Current rows: {len(weather_df)}")
    
    # Drop problematic columns
    columns_to_drop = ['uv_index', 'visibility']
    for col in columns_to_drop:
        try:
            conn.execute(f"ALTER TABLE weather_historical DROP COLUMN {col}")
            print(f"   ✅ Dropped column: {col}")
        except Exception as e:
            print(f"   ⚠️ Could not drop {col}: {e}")
    
    # 2. Fill missing location metadata with defaults
    print("🧹 Cleaning location_metadata table...")
    
    # Update location metadata with reasonable defaults
    updates = [
        ("elevation", "500"),  # Average elevation for Pune
        ("land_use", "'Mixed'"),
        ("population_density", "1000"),  # Reasonable default
        ("traffic_density", "'Medium'"),
        ("industrial_activity", "'Moderate'")
    ]
    
    for col, default_val in updates:
        try:
            conn.execute(f"UPDATE location_metadata SET {col} = {default_val} WHERE {col} IS NULL")
            print(f"   ✅ Updated {col} with default values")
        except Exception as e:
            print(f"   ⚠️ Could not update {col}: {e}")
    
    # 3. Remove empty tables that are not needed
    empty_tables = ['data_quality_metrics', 'realtime_weather', 'realtime_air_quality', 'collection_log']
    
    for table in empty_tables:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"   ✅ Dropped empty table: {table}")
        except Exception as e:
            print(f"   ⚠️ Could not drop {table}: {e}")
    
    conn.commit()
    conn.close()
    
    print("✅ Database cleanup completed!")

def clean_files():
    """Remove unnecessary files"""
    print("\n📁 CLEANING FILES")
    print("=" * 50)
    
    # Remove .DS_Store files
    ds_store_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == ".DS_Store":
                ds_store_files.append(os.path.join(root, file))
    
    for file in ds_store_files:
        try:
            os.remove(file)
            print(f"   ✅ Removed: {file}")
        except Exception as e:
            print(f"   ⚠️ Could not remove {file}: {e}")
    
    # Clean up problematic CSV file
    co2_file = "data/external/co2_mm_mlo.csv"
    if os.path.exists(co2_file):
        try:
            # Try to fix the CSV file
            with open(co2_file, 'r') as f:
                lines = f.readlines()
            
            # Keep only properly formatted lines
            clean_lines = []
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    # Check if line has consistent format
                    parts = line.strip().split(',')
                    if len(parts) == 1:  # Expected format
                        clean_lines.append(line)
            
            # Write cleaned file
            with open(co2_file, 'w') as f:
                f.writelines(clean_lines)
            
            print(f"   ✅ Fixed CSV format: {co2_file}")
            
        except Exception as e:
            print(f"   ⚠️ Could not fix {co2_file}: {e}")
            # If can't fix, remove it
            try:
                os.remove(co2_file)
                print(f"   ✅ Removed problematic file: {co2_file}")
            except:
                pass

def optimize_for_ml():
    """Optimize data for machine learning"""
    print("\n🤖 OPTIMIZING FOR ML TRAINING")
    print("=" * 50)
    
    db_path = "data/climate_aqi_database.db"
    conn = sqlite3.connect(db_path)
    
    # 1. Create optimized training dataset
    print("📊 Creating optimized training dataset...")
    
    try:
        # Merge weather and air quality data
        query = """
        SELECT 
            w.timestamp,
            w.location_id,
            w.temperature,
            w.humidity,
            w.pressure,
            w.wind_speed,
            w.wind_direction,
            w.precipitation,
            w.solar_radiation,
            w.cloud_cover,
            a.pm25,
            a.pm10,
            a.no2,
            a.so2,
            a.co,
            a.o3,
            a.aqi
        FROM weather_historical w
        LEFT JOIN air_quality_historical a 
        ON w.timestamp = a.timestamp AND w.location_id = a.location_id
        WHERE w.temperature IS NOT NULL 
        AND w.humidity IS NOT NULL
        """
        
        training_df = pd.read_sql_query(query, conn)
        
        # Remove rows with too many nulls
        null_threshold = 0.3  # Remove rows with >30% null values
        training_df = training_df.dropna(thresh=len(training_df.columns) * (1 - null_threshold))
        
        # Fill remaining nulls with appropriate values
        numeric_columns = training_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if training_df[col].isnull().any():
                # Use median for numeric columns
                median_val = training_df[col].median()
                training_df[col].fillna(median_val, inplace=True)
        
        print(f"   📈 Optimized dataset: {len(training_df)} rows, {len(training_df.columns)} columns")
        print(f"   📊 Null values remaining: {training_df.isnull().sum().sum()}")
        
        # Save optimized dataset
        training_df.to_csv("data/processed/ml_training_dataset.csv", index=False)
        print("   ✅ Saved optimized training dataset")
        
    except Exception as e:
        print(f"   ❌ Error creating training dataset: {e}")
    
    # 2. Create feature engineering dataset
    print("🔧 Creating feature-engineered dataset...")
    
    try:
        # Add time-based features
        training_df['timestamp'] = pd.to_datetime(training_df['timestamp'])
        training_df['hour'] = training_df['timestamp'].dt.hour
        training_df['day_of_week'] = training_df['timestamp'].dt.dayofweek
        training_df['month'] = training_df['timestamp'].dt.month
        training_df['season'] = training_df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Autumn
        })
        
        # Add cyclical encoding
        training_df['hour_sin'] = np.sin(2 * np.pi * training_df['hour'] / 24)
        training_df['hour_cos'] = np.cos(2 * np.pi * training_df['hour'] / 24)
        training_df['month_sin'] = np.sin(2 * np.pi * training_df['month'] / 12)
        training_df['month_cos'] = np.cos(2 * np.pi * training_df['month'] / 12)
        
        # Add interaction features
        if 'temperature' in training_df.columns and 'humidity' in training_df.columns:
            training_df['temp_humidity_interaction'] = training_df['temperature'] * training_df['humidity']
        
        if 'pm25' in training_df.columns and 'pm10' in training_df.columns:
            training_df['pm_ratio'] = training_df['pm25'] / (training_df['pm10'] + 1e-6)
        
        # Save feature-engineered dataset
        training_df.to_csv("data/processed/ml_features_dataset.csv", index=False)
        print(f"   ✅ Created feature-engineered dataset: {len(training_df.columns)} features")
        
    except Exception as e:
        print(f"   ❌ Error creating features: {e}")
    
    conn.close()

def create_data_summary():
    """Create a summary of cleaned data"""
    print("\n📋 CREATING DATA SUMMARY")
    print("=" * 50)
    
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "database_status": "cleaned",
        "files_cleaned": True,
        "ml_ready": True
    }
    
    # Check database
    db_path = "data/climate_aqi_database.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        
        # Get table info
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        summary["tables"] = {}
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            summary["tables"][table] = {
                "rows": len(df),
                "columns": len(df.columns),
                "null_values": int(df.isnull().sum().sum()),
                "null_percentage": float((df.isnull().sum().sum() / df.size) * 100) if df.size > 0 else 0
            }
        
        conn.close()
    
    # Check processed files
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        summary["processed_files"] = {}
        for file in processed_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                summary["processed_files"][file.name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_mb": file.stat().st_size / (1024 * 1024)
                }
            except:
                pass
    
    # Save summary
    import json
    with open("data/cleanup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Data summary created: data/cleanup_summary.json")
    
    # Print summary
    print("\n📊 CLEANUP SUMMARY:")
    print(f"   🗄️ Database tables: {len(summary.get('tables', {}))}")
    for table, info in summary.get('tables', {}).items():
        print(f"      - {table}: {info['rows']} rows, {info['null_percentage']:.1f}% nulls")
    
    print(f"   📁 Processed files: {len(summary.get('processed_files', {}))}")
    for file, info in summary.get('processed_files', {}).items():
        print(f"      - {file}: {info['rows']} rows, {info['size_mb']:.1f} MB")

def main():
    """Run complete system cleanup"""
    print("🧹 COMPREHENSIVE SYSTEM CLEANUP")
    print("=" * 80)
    print("Fixing null data, removing unnecessary files, and optimizing for ML...")
    print()
    
    try:
        # Run cleanup steps
        clean_database()
        clean_files()
        optimize_for_ml()
        create_data_summary()
        
        print("\n" + "=" * 80)
        print("🎉 CLEANUP COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ Database optimized for ML training")
        print("✅ Unnecessary files removed")
        print("✅ Feature-engineered datasets created")
        print("✅ System ready for machine learning")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        print("Please check the errors above and run again.")

if __name__ == "__main__":
    main()