#!/usr/bin/env python3
"""
Test script for the Pune Climate Dashboard
Verifies all components are working correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from data_collector import fetch_city_data
        from data_preprocessor import clean_and_preprocess
        from model_trainer import train_model
        from predictor import predict_future
        from evaluator import evaluate_model
        from report_generator import generate_report
        print("✅ Backend modules imported successfully")
    except ImportError as e:
        print(f"❌ Backend import failed: {e}")
        return False
    
    try:
        from visualization import create_all_visualizations
        print("✅ Visualization module imported successfully")
    except ImportError as e:
        print(f"❌ Visualization import failed: {e}")
        return False
    
    return True

def test_data_collection():
    """Test data collection functionality"""
    print("\n📊 Testing data collection...")
    
    try:
        from data_collector import fetch_city_data
        
        # Test with small dataset
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(
            fetch_city_data("Pune", 2023, 2024, include_current=False)
        )
        loop.close()
        
        if data.empty:
            print("❌ Data collection returned empty dataset")
            return False
        
        print(f"✅ Data collection successful: {len(data)} records")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        
        return data
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return False

def test_data_preprocessing(data):
    """Test data preprocessing functionality"""
    print("\n🔧 Testing data preprocessing...")
    
    try:
        from data_preprocessor import clean_and_preprocess
        
        results = clean_and_preprocess(
            data,
            target_variables=['temperature', 'rainfall'],
            scaling_method='robust'
        )
        
        if not results or 'final' not in results:
            print("❌ Data preprocessing failed")
            return False
        
        processed_data = results['final']
        print(f"✅ Data preprocessing successful: {len(processed_data)} records")
        print(f"   Features: {len(processed_data.columns)} columns")
        
        return results
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def test_model_training(processed_results):
    """Test model training functionality"""
    print("\n🤖 Testing model training...")
    
    try:
        from model_trainer import train_model
        
        processed_data = processed_results['final']
        
        # Test with simple linear model
        model_info = train_model(
            processed_data,
            'temperature',
            'linear',
            optimize=False
        )
        
        if not model_info or 'model' not in model_info:
            print("❌ Model training failed")
            return False
        
        print(f"✅ Model training successful")
        print(f"   R² Score: {model_info.get('test_r2', 'N/A'):.3f}")
        print(f"   RMSE: {model_info.get('test_rmse', 'N/A'):.3f}")
        
        return model_info
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def test_predictions(model_info, processed_data):
    """Test prediction functionality"""
    print("\n🔮 Testing predictions...")
    
    try:
        from predictor import predict_future
        
        future_years = [2025, 2026, 2027]
        predictions = predict_future(model_info, future_years, processed_data)
        
        if predictions.empty:
            print("❌ Prediction generation failed")
            return False
        
        print(f"✅ Predictions successful: {len(predictions)} records")
        print(f"   Columns: {list(predictions.columns)}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Predictions failed: {e}")
        return False

def test_visualizations(data, predictions):
    """Test visualization functionality"""
    print("\n📈 Testing visualizations...")
    
    try:
        from visualization import create_all_visualizations
        
        pred_dict = {'temperature': predictions} if predictions is not False else None
        
        visuals = create_all_visualizations(data, pred_dict)
        
        if not visuals:
            print("❌ Visualization creation failed")
            return False
        
        print(f"✅ Visualizations successful: {len(visuals)} charts created")
        print(f"   Chart types: {list(visuals.keys())}")
        
        return visuals
        
    except Exception as e:
        print(f"❌ Visualizations failed: {e}")
        return False

def test_report_generation(data, predictions):
    """Test report generation functionality"""
    print("\n📄 Testing report generation...")
    
    try:
        from report_generator import generate_report
        
        pred_dict = {'temperature': predictions} if predictions is not False else None
        
        report_path = generate_report(
            data,
            predictions=pred_dict,
            output_path="test_report.pdf"
        )
        
        if not os.path.exists(report_path):
            print("❌ Report generation failed - file not created")
            return False
        
        file_size = os.path.getsize(report_path)
        print(f"✅ Report generation successful")
        print(f"   File: {report_path}")
        print(f"   Size: {file_size:,} bytes")
        
        # Clean up test file
        try:
            os.remove(report_path)
            print("   Test file cleaned up")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

def test_dashboard_components():
    """Test dashboard-specific components"""
    print("\n🎛️ Testing dashboard components...")
    
    try:
        # Test utility functions from dashboard
        import streamlit_dashboard
        
        # Test session state initialization
        print("✅ Dashboard module loaded successfully")
        
        # Test risk calculation
        risk_score = streamlit_dashboard.calculate_risk_score(2.5, 85, 15)
        print(f"✅ Risk calculation working: score = {risk_score}")
        
        # Test AI insights generation
        sample_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', '2023-12-31', freq='D'),
            'temperature': np.random.normal(25, 5, 1461),
            'rainfall': np.random.exponential(2, 1461),
            'aqi': np.random.normal(75, 20, 1461),
            'year': [d.year for d in pd.date_range('2020-01-01', '2023-12-31', freq='D')]
        })
        
        insights = streamlit_dashboard.generate_ai_insights(sample_data)
        print(f"✅ AI insights generation working: {len(insights)} insights")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard components test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🌡️ PUNE CLIMATE DASHBOARD - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Track test results
    test_results = {}
    
    # Test 1: Module imports
    test_results['imports'] = test_imports()
    
    if not test_results['imports']:
        print("\n❌ Critical import failures - stopping tests")
        return False
    
    # Test 2: Data collection
    data = test_data_collection()
    test_results['data_collection'] = data is not False
    
    if not test_results['data_collection']:
        print("\n❌ Data collection failed - stopping tests")
        return False
    
    # Test 3: Data preprocessing
    processed_results = test_data_preprocessing(data)
    test_results['preprocessing'] = processed_results is not False
    
    if not test_results['preprocessing']:
        print("\n❌ Data preprocessing failed - continuing with limited tests")
        processed_results = None
    
    # Test 4: Model training
    if processed_results:
        model_info = test_model_training(processed_results)
        test_results['model_training'] = model_info is not False
        processed_data = processed_results['final']
    else:
        model_info = False
        test_results['model_training'] = False
        processed_data = data
    
    # Test 5: Predictions
    if model_info:
        predictions = test_predictions(model_info, processed_data)
        test_results['predictions'] = predictions is not False
    else:
        predictions = False
        test_results['predictions'] = False
    
    # Test 6: Visualizations
    visuals = test_visualizations(data, predictions)
    test_results['visualizations'] = visuals is not False
    
    # Test 7: Report generation
    test_results['reports'] = test_report_generation(data, predictions)
    
    # Test 8: Dashboard components
    test_results['dashboard'] = test_dashboard_components()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Dashboard is ready to use.")
        print("\n🚀 To launch the dashboard, run:")
        print("   python run_dashboard.py")
        print("   OR")
        print("   streamlit run streamlit_dashboard.py")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("   Dashboard may have limited functionality.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)