#!/usr/bin/env python3
"""
Setup script for the Pune Climate Prediction System
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'data/api',
        'outputs/models',
        'outputs/figures',
        'outputs/logs',
        'outputs/reports'
    ]
    
    print("📁 Creating directory structure...")
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ Directory structure created!")

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False

def create_env_file():
    """Create environment configuration file"""
    env_content = """# Pune Climate Prediction System Configuration

# Database
DATABASE_PATH=data/climate_aqi_database.db

# API Configuration (add your API keys if needed)
# OPENWEATHER_API_KEY=your_api_key_here
# NASA_API_KEY=your_api_key_here

# Dashboard Configuration
DASHBOARD_PORT=8501
DASHBOARD_HOST=localhost

# Model Configuration
MODEL_SAVE_PATH=outputs/models/
ENABLE_MODEL_OPTIMIZATION=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=outputs/logs/system.log

# Data Collection
DATA_COLLECTION_INTERVAL=3600  # seconds
ENABLE_REAL_TIME_DATA=true
"""
    
    env_path = Path('.env')
    if not env_path.exists():
        print("⚙️ Creating environment configuration...")
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✅ Environment file created: .env")
    else:
        print("ℹ️ Environment file already exists")

def test_installation():
    """Test if the installation is working"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import streamlit
        import pandas
        import numpy
        import sklearn
        import plotly
        
        print("✅ Core packages imported successfully!")
        
        # Test backend imports
        sys.path.append('backend')
        from data_collector import fetch_city_data
        from data_preprocessor import clean_and_preprocess
        from model_trainer import train_model
        
        print("✅ Backend modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🌡️ PUNE CLIMATE PREDICTION SYSTEM - SETUP")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\n📋 Next Steps:")
    print("   1. Run the application: python run_app.py")
    print("   2. Test the backend: python test_backend.py")
    print("   3. Open browser: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Edit .env file to configure settings")
    print("   - Check outputs/logs/ for system logs")
    print("   - Use 'streamlit run app.py' for direct access")
    
    print("\n🌍 Ready to predict Pune's climate future!")

if __name__ == "__main__":
    main()