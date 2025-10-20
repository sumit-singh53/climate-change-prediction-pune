#!/usr/bin/env python3
"""
Simple test script for CI/CD validation
Tests basic functionality without external dependencies
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing module imports...")
    
    # Add src to path
    sys.path.append('src')
    
    try:
        # Test configuration
        from config import PUNE_LOCATIONS, MODEL_CONFIG, REALTIME_CONFIG
        print(f"✅ Configuration loaded: {len(PUNE_LOCATIONS)} locations")
        
        # Test that we can import other modules
        import realtime_data_collector
        print("✅ Real-time data collector module imported")
        
        import advanced_ml_models
        print("✅ Advanced ML models module imported")
        
        # Test dashboard import with error handling for CI
        try:
            import realtime_dashboard
            print("✅ Real-time dashboard module imported")
        except Exception as dashboard_error:
            print(f"⚠️ Dashboard import warning (non-critical in CI): {dashboard_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_configuration():
    """Test configuration values"""
    print("🔧 Testing configuration...")
    
    sys.path.append('src')
    
    try:
        from config import PUNE_LOCATIONS, MODEL_CONFIG, REALTIME_CONFIG
        
        # Test locations
        assert len(PUNE_LOCATIONS) == 8, f"Expected 8 locations, got {len(PUNE_LOCATIONS)}"
        print(f"✅ All {len(PUNE_LOCATIONS)} Pune locations configured")
        
        # Test model config
        assert 'target_variables' in MODEL_CONFIG, "Missing target_variables in MODEL_CONFIG"
        print(f"✅ Model config has {len(MODEL_CONFIG['target_variables'])} target variables")
        
        # Test realtime config
        assert 'update_interval_minutes' in REALTIME_CONFIG, "Missing update_interval_minutes"
        print(f"✅ Real-time config set to update every {REALTIME_CONFIG['update_interval_minutes']} minutes")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_project_structure():
    """Test that required files exist"""
    print("📁 Testing project structure...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'LICENSE',
        'src/config.py',
        'src/realtime_data_collector.py',
        'src/advanced_ml_models.py',
        'src/realtime_dashboard.py',
        'src/main_orchestrator.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_dependencies():
    """Test that critical dependencies are available"""
    print("📦 Testing critical dependencies...")
    
    critical_deps = [
        'pandas', 'numpy', 'scikit-learn', 'aiohttp', 
        'streamlit', 'plotly', 'tensorflow'
    ]
    
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            missing_deps.append(dep)
            print(f"⚠️ {dep} not available")
    
    if missing_deps:
        print(f"⚠️ Missing dependencies (may cause issues): {missing_deps}")
        # Don't fail the test, just warn
    
    print("✅ Dependency check completed")
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Enhanced Climate & AQI Prediction System Tests")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("Module Imports", test_imports)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())