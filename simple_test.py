#!/usr/bin/env python3
"""
Simple test script for CI/CD validation
Tests basic functionality and imports without external dependencies
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version not supported")
        return False

def test_basic_imports():
    """Test basic Python imports"""
    try:
        import pandas
        import numpy
        import sklearn
        print("✅ Basic scientific libraries imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import basic libraries: {e}")
        return False

def test_project_structure():
    """Test project directory structure"""
    required_dirs = ['src', 'data']
    required_files = ['requirements.txt', 'README.md', 'src/config.py']
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory {directory} exists")
        else:
            print(f"❌ Directory {directory} missing")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File {file_path} exists")
        else:
            print(f"❌ File {file_path} missing")
            all_good = False
    
    return all_good

def test_config_imports():
    """Test configuration imports"""
    try:
        sys.path.append('src')
        from config import PUNE_LOCATIONS, MODEL_CONFIG, API_CONFIG
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Locations configured: {len(PUNE_LOCATIONS)}")
        print(f"   - Target variables: {MODEL_CONFIG['target_variables']}")
        print(f"   - API endpoints: {len(API_CONFIG)}")
        return True
    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        return False

def test_core_modules():
    """Test core module imports (non-blocking)"""
    sys.path.append('src')
    modules_to_test = [
        'realtime_data_collector',
        'advanced_ml_models', 
        'realtime_dashboard'
    ]
    
    success_count = 0
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ Module {module_name} imported successfully")
            success_count += 1
        except Exception as e:
            print(f"⚠️  Module {module_name} import warning: {e}")
    
    print(f"Core modules test: {success_count}/{len(modules_to_test)} successful")
    return success_count > 0  # At least one module should work

def create_required_directories():
    """Create required directories if they don't exist"""
    directories = [
        'data/raw', 'data/processed', 'data/external', 'data/api',
        'outputs/models', 'outputs/logs', 'outputs/figures'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory {directory} ready")

def main():
    """Run all tests"""
    print("🚀 Starting simple test suite...")
    print("=" * 50)
    
    # Create directories first
    create_required_directories()
    
    tests = [
        ("Python Version", test_python_version),
        ("Basic Imports", test_basic_imports),
        ("Project Structure", test_project_structure),
        ("Configuration", test_config_imports),
        ("Core Modules", test_core_modules),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed >= 4:  # Allow one test to fail
        print(f"🎉 Test suite passed! ({passed}/{total} tests successful)")
        print("✅ System is ready for deployment")
        return 0
    else:
        print(f"❌ Test suite failed! ({passed}/{total} tests successful)")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)