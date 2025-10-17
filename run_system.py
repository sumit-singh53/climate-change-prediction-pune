#!/usr/bin/env python3
"""
Quick Start Script for Enhanced Climate & AQI Prediction System
Run this script to start the complete system with all components
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn',
        'tensorflow', 'xgboost', 'lightgbm', 'aiohttp', 'paho-mqtt', 'flask'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '-r', 'requirements.txt'
        ])
        print("✅ Dependencies installed successfully!")
    else:
        print("✅ All dependencies are installed!")

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'data/iot',
        'data/api',
        'outputs/models',
        'outputs/figures',
        'outputs/logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully!")

def initialize_database():
    """Initialize the database with sample data"""
    print("🗄️ Initializing database...")
    
    try:
        # Import and run data collection to initialize database
        sys.path.append('src')
        from enhanced_data_collector import EnhancedDataCollector
        import asyncio
        
        collector = EnhancedDataCollector()
        
        # Run initial data collection
        print("📊 Collecting initial data...")
        asyncio.run(collector.run_data_collection(days_back=7))
        
        print("✅ Database initialized with sample data!")
        
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
        print("The system will still work, but may have limited initial data.")

def start_system(mode='full', simulate_sensors=True, port=8501):
    """Start the system based on specified mode"""
    print(f"🚀 Starting system in {mode} mode...")
    
    # Change to src directory
    os.chdir('src')
    
    if mode == 'full':
        print("🌟 Starting complete system with all components...")
        print("📊 Dashboard will be available at: http://localhost:8501")
        print("📡 Real-time data collection every 30 minutes")
        print("🤖 ML predictions based on live API data")
        
        cmd = [sys.executable, 'main_orchestrator.py', '--mode', 'full']
        if simulate_sensors:
            cmd.append('--collect-initial-data')
            print("📊 Initial data collection enabled")
    
    elif mode == 'dashboard':
        print("📊 Starting dashboard only...")
        print(f"🌐 Dashboard will be available at: http://localhost:{port}")
        
        cmd = ['streamlit', 'run', 'realtime_dashboard.py', f'--server.port={port}']
    
    elif mode == 'data-collection':
        print("📡 Starting data collection service...")
        cmd = [sys.executable, 'main_orchestrator.py', '--mode', 'data-collection']
    
    elif mode == 'training':
        print("🧠 Starting model training...")
        cmd = [sys.executable, 'main_orchestrator.py', '--mode', 'training']
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested by user")
    except Exception as e:
        print(f"❌ Error starting system: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Enhanced Climate & AQI Prediction System - Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_system.py                          # Start full system with sensor simulation
  python run_system.py --mode dashboard         # Start dashboard only
  python run_system.py --no-simulate            # Start without sensor simulation
  python run_system.py --port 8502              # Use custom port for dashboard
  python run_system.py --skip-setup             # Skip dependency and setup checks
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'dashboard', 'data-collection', 'training'],
        default='full',
        help='System mode to run (default: full)'
    )
    
    parser.add_argument(
        '--no-initial-data',
        action='store_true',
        help='Skip initial data collection (default: enabled)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for dashboard (default: 8501)'
    )
    
    parser.add_argument(
        '--skip-setup',
        action='store_true',
        help='Skip dependency checks and setup'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick start - skip data initialization'
    )
    
    args = parser.parse_args()
    
    print("🌍 Enhanced Climate & AQI Prediction System for Pune")
    print("=" * 60)
    
    if not args.skip_setup:
        # Setup phase
        check_dependencies()
        setup_directories()
        
        if not args.quick:
            initialize_database()
    
    print("\n" + "=" * 60)
    
    # Start system
    start_system(
        mode=args.mode,
        simulate_sensors=not args.no_initial_data,
        port=args.port
    )

if __name__ == "__main__":
    main()