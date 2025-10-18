#!/usr/bin/env python3
"""
Comprehensive System Audit Script
Checks for null data, unnecessary files, and potential ML issues
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import json

def check_database_quality():
    """Check database for null values and data quality issues"""
    print("🗄️ DATABASE QUALITY AUDIT")
    print("=" * 60)
    
    db_path = "data/climate_aqi_database.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Get all tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    print(f"📊 Found {len(tables)} tables: {tables}")
    print()
    
    issues_found = []
    
    for table in tables:
        print(f"🔍 Analyzing table: {table}")
        
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            
            if df.empty:
                print(f"   ⚠️ Table {table} is EMPTY")
                issues_found.append(f"Empty table: {table}")
                continue
            
            print(f"   📈 Rows: {len(df)}")
            print(f"   📊 Columns: {len(df.columns)}")
            
            # Check for null values
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            
            if total_nulls > 0:
                print(f"   ⚠️ NULL VALUES FOUND: {total_nulls} total")
                for col, null_count in null_counts.items():
                    if null_count > 0:
                        percentage = (null_count / len(df)) * 100
                        print(f"      - {col}: {null_count} nulls ({percentage:.1f}%)")
                        if percentage > 50:
                            issues_found.append(f"High null rate in {table}.{col}: {percentage:.1f}%")
            else:
                print("   ✅ No null values found")
            
            # Check for duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                print(f"   ⚠️ DUPLICATE ROWS: {duplicates}")
                issues_found.append(f"Duplicate rows in {table}: {duplicates}")
            else:
                print("   ✅ No duplicate rows")
            
            # Check data types
            print(f"   📋 Data types:")
            for col, dtype in df.dtypes.items():
                print(f"      - {col}: {dtype}")
                
                # Check for mixed data types in numeric columns
                if col in ['temperature', 'humidity', 'pm25', 'pm10', 'aqi', 'pressure', 'wind_speed']:
                    non_numeric = pd.to_numeric(df[col], errors='coerce').isnull().sum()
                    if non_numeric > 0:
                        print(f"      ⚠️ Non-numeric values in {col}: {non_numeric}")
                        issues_found.append(f"Non-numeric values in {table}.{col}: {non_numeric}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error analyzing {table}: {e}")
            issues_found.append(f"Error analyzing {table}: {e}")
    
    conn.close()
    
    return issues_found

def check_file_system():
    """Check for unnecessary files and potential issues"""
    print("📁 FILE SYSTEM AUDIT")
    print("=" * 60)
    
    issues_found = []
    unnecessary_files = []
    
    # Check for common unnecessary files
    unwanted_patterns = [
        "*.tmp", "*.temp", "*.log", "*.bak", "*.old", 
        "*.pyc", "__pycache__", ".DS_Store", "Thumbs.db",
        "*.swp", "*.swo", "*.orig", "*.rej"
    ]
    
    # Scan directory
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            # Check for very large files (>100MB)
            if file_size > 100 * 1024 * 1024:
                print(f"⚠️ Large file found: {file_path} ({file_size / (1024*1024):.1f} MB)")
                issues_found.append(f"Large file: {file_path}")
            
            # Check for unnecessary files
            for pattern in unwanted_patterns:
                if pattern.startswith("*.") and file.endswith(pattern[1:]):
                    unnecessary_files.append(file_path)
                elif pattern in file:
                    unnecessary_files.append(file_path)
    
    if unnecessary_files:
        print(f"🗑️ Found {len(unnecessary_files)} unnecessary files:")
        for file in unnecessary_files[:10]:  # Show first 10
            print(f"   - {file}")
        if len(unnecessary_files) > 10:
            print(f"   ... and {len(unnecessary_files) - 10} more")
    else:
        print("✅ No unnecessary files found")
    
    return issues_found, unnecessary_files

def check_python_files():
    """Check Python files for potential issues"""
    print("\n🐍 PYTHON FILES AUDIT")
    print("=" * 60)
    
    issues_found = []
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📄 Found {len(python_files)} Python files")
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common issues
            lines = content.split('\n')
            
            # Check for very long files
            if len(lines) > 1000:
                print(f"⚠️ Very long file: {py_file} ({len(lines)} lines)")
                issues_found.append(f"Long file: {py_file}")
            
            # Check for TODO/FIXME comments
            todos = [i for i, line in enumerate(lines) if 'TODO' in line.upper() or 'FIXME' in line.upper()]
            if todos:
                print(f"📝 TODOs/FIXMEs in {py_file}: {len(todos)}")
            
            # Check for hardcoded paths
            hardcoded_paths = [i for i, line in enumerate(lines) if ('C:\\' in line or '/home/' in line) and not line.strip().startswith('#')]
            if hardcoded_paths:
                print(f"⚠️ Hardcoded paths in {py_file}: {len(hardcoded_paths)}")
                issues_found.append(f"Hardcoded paths in {py_file}")
            
        except Exception as e:
            print(f"❌ Error reading {py_file}: {e}")
            issues_found.append(f"Error reading {py_file}: {e}")
    
    return issues_found

def check_data_files():
    """Check data files for issues"""
    print("\n📊 DATA FILES AUDIT")
    print("=" * 60)
    
    issues_found = []
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return ["Data directory missing"]
    
    # Check each data subdirectory
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            print(f"📁 Checking {subdir.name}/")
            
            files = list(subdir.glob("*"))
            print(f"   📄 Files: {len(files)}")
            
            for file in files:
                if file.suffix in ['.csv', '.json', '.txt']:
                    try:
                        file_size = file.stat().st_size
                        print(f"   - {file.name}: {file_size / 1024:.1f} KB")
                        
                        # Check CSV files for data quality
                        if file.suffix == '.csv':
                            df = pd.read_csv(file)
                            null_percentage = (df.isnull().sum().sum() / df.size) * 100
                            if null_percentage > 30:
                                print(f"     ⚠️ High null rate: {null_percentage:.1f}%")
                                issues_found.append(f"High null rate in {file}: {null_percentage:.1f}%")
                            
                    except Exception as e:
                        print(f"     ❌ Error reading {file.name}: {e}")
                        issues_found.append(f"Error reading {file}: {e}")
    
    return issues_found

def check_ml_readiness():
    """Check if data is ready for ML training"""
    print("\n🤖 ML READINESS AUDIT")
    print("=" * 60)
    
    issues_found = []
    
    # Add src to path for imports
    sys.path.append('src')
    
    try:
        from config import PUNE_LOCATIONS, MODEL_CONFIG
        
        print(f"✅ Configuration loaded successfully")
        print(f"   📍 Locations: {len(PUNE_LOCATIONS)}")
        print(f"   🎯 Target variables: {MODEL_CONFIG['target_variables']}")
        
        # Check database for ML readiness
        db_path = "data/climate_aqi_database.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            
            # Check if we have enough data for each target variable
            for target in MODEL_CONFIG['target_variables']:
                # Check weather data
                weather_query = f"SELECT COUNT(*) FROM weather_historical WHERE {target} IS NOT NULL"
                try:
                    weather_count = pd.read_sql_query(weather_query, conn).iloc[0, 0]
                    print(f"   📊 {target} (weather): {weather_count} records")
                    
                    if weather_count < 100:
                        issues_found.append(f"Insufficient {target} data for ML: {weather_count} records")
                except:
                    # Try air quality table
                    aqi_query = f"SELECT COUNT(*) FROM air_quality_historical WHERE {target} IS NOT NULL"
                    try:
                        aqi_count = pd.read_sql_query(aqi_query, conn).iloc[0, 0]
                        print(f"   📊 {target} (aqi): {aqi_count} records")
                        
                        if aqi_count < 100:
                            issues_found.append(f"Insufficient {target} data for ML: {aqi_count} records")
                    except:
                        print(f"   ❌ No data found for {target}")
                        issues_found.append(f"No data found for target variable: {target}")
            
            conn.close()
        else:
            issues_found.append("Database file missing")
            
    except Exception as e:
        print(f"❌ Error checking ML readiness: {e}")
        issues_found.append(f"ML readiness check failed: {e}")
    
    return issues_found

def generate_cleanup_recommendations(all_issues, unnecessary_files):
    """Generate recommendations for cleanup"""
    print("\n🧹 CLEANUP RECOMMENDATIONS")
    print("=" * 60)
    
    if not all_issues and not unnecessary_files:
        print("🎉 No issues found! Your system is clean and ready for ML training.")
        return
    
    print("📋 Issues to address:")
    for i, issue in enumerate(all_issues, 1):
        print(f"   {i}. {issue}")
    
    if unnecessary_files:
        print(f"\n🗑️ Files recommended for deletion ({len(unnecessary_files)} files):")
        for file in unnecessary_files[:5]:  # Show first 5
            print(f"   - {file}")
        if len(unnecessary_files) > 5:
            print(f"   ... and {len(unnecessary_files) - 5} more")
    
    print("\n💡 Recommended actions:")
    print("   1. Clean null values in database tables")
    print("   2. Remove duplicate records")
    print("   3. Delete unnecessary files")
    print("   4. Fix data type inconsistencies")
    print("   5. Ensure sufficient data for ML training")

def main():
    """Run comprehensive system audit"""
    print("🔍 COMPREHENSIVE SYSTEM AUDIT")
    print("=" * 80)
    print("Checking for null data, unnecessary files, and ML readiness issues...")
    print()
    
    all_issues = []
    
    # Run all checks
    db_issues = check_database_quality()
    all_issues.extend(db_issues)
    
    fs_issues, unnecessary_files = check_file_system()
    all_issues.extend(fs_issues)
    
    py_issues = check_python_files()
    all_issues.extend(py_issues)
    
    data_issues = check_data_files()
    all_issues.extend(data_issues)
    
    ml_issues = check_ml_readiness()
    all_issues.extend(ml_issues)
    
    # Generate recommendations
    generate_cleanup_recommendations(all_issues, unnecessary_files)
    
    print("\n" + "=" * 80)
    print(f"📊 AUDIT SUMMARY: {len(all_issues)} issues found")
    print("=" * 80)

if __name__ == "__main__":
    main()