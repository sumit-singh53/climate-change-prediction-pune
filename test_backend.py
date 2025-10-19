#!/usr/bin/env python3
"""
Comprehensive test script for the Pune Climate Prediction System backend
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_data_collection():
    """Test data collection functionality"""
    print("🧪 Testing Data Collection...")
    
    try:
        from data_collector import fetch_city_data
        
        # Test async data collection
        async def run_test():
            data = await fetch_city_data("Pune", 2022, 2024, include_current=True)
            return data
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(run_test())
        loop.close()
        
        # Validate data
        assert not data.empty, "Data should not be empty"
        assert 'date' in data.columns, "Date column missing"
        assert 'temperature' in data.columns, "Temperature column missing"
        assert 'rainfall' in data.columns, "Rainfall column missing"
        assert len(data) > 100, "Should have substantial data"
        
        print(f"   ✅ Data collection successful: {len(data):,} records")
        return data
        
    except Exception as e:
        print(f"   ❌ Data collection failed: {e}")
        return None

def test_data_preprocessing(data):
    """Test data preprocessing functionality"""
    print("🧪 Testing Data Preprocessing...")
    
    if data is None:
        print("   ⏭️ Skipping (no data)")
        return None
    
    try:
        from data_preprocessor import clean_and_preprocess
        
        results = clean_and_preprocess(
            data, 
            target_variables=['temperature', 'rainfall'],
            scaling_method='robust',
            create_features=True
        )
        
        # Validate results
        assert 'final' in results, "Final processed data missing"
        assert not results['final'].empty, "Processed data should not be empty"
        assert len(results['final'].columns) > len(data.columns), "Should have more features"
        
        print(f"   ✅ Preprocessing successful: {len(results['final'])} records, {len(results['final'].columns)} features")
        return results
        
    except Exception as e:
        print(f"   ❌ Preprocessing failed: {e}")
        return None

def test_model_training(processed_data):
    """Test model training functionality"""
    print("🧪 Testing Model Training...")
    
    if processed_data is None:
        print("   ⏭️ Skipping (no processed data)")
        return None
    
    try:
        from model_trainer import train_model
        
        data = processed_data['final']
        models = {}
        
        # Test different model types
        model_types = ['linear', 'random_forest']  # Skip LSTM and Prophet for quick testing
        
        for model_type in model_types:
            try:
                print(f"   🔄 Training {model_type} model...")
                model_info = train_model(
                    data, 
                    target='temperature', 
                    model_type=model_type,
                    optimize=False  # Skip optimization for speed
                )
                
                # Validate model
                assert 'model' in model_info, f"{model_type} model missing"
                assert 'test_r2' in model_info, f"{model_type} R² score missing"
                assert model_info['test_r2'] > 0, f"{model_type} R² should be positive"
                
                models[model_type] = model_info
                print(f"   ✅ {model_type} trained: R² = {model_info['test_r2']:.3f}")
                
            except Exception as e:
                print(f"   ❌ {model_type} training failed: {e}")
                continue
        
        if models:
            print(f"   ✅ Model training successful: {len(models)} models trained")
            return models
        else:
            print("   ❌ No models trained successfully")
            return None
            
    except Exception as e:
        print(f"   ❌ Model training failed: {e}")
        return None

def test_predictions(models, processed_data):
    """Test prediction functionality"""
    print("🧪 Testing Predictions...")
    
    if models is None or processed_data is None:
        print("   ⏭️ Skipping (no models or data)")
        return None
    
    try:
        from predictor import predict_future
        
        data = processed_data['final']
        predictions = {}
        
        for model_type, model_info in models.items():
            try:
                print(f"   🔄 Generating {model_type} predictions...")
                
                pred_df = predict_future(
                    model_info, 
                    future_years=[2025, 2026],
                    historical_df=data
                )
                
                # Validate predictions
                assert not pred_df.empty, f"{model_type} predictions should not be empty"
                assert 'temperature_predicted' in pred_df.columns, f"{model_type} prediction column missing"
                assert len(pred_df) > 300, f"{model_type} should have substantial predictions"
                
                predictions[model_type] = pred_df
                print(f"   ✅ {model_type} predictions: {len(pred_df)} records")
                
            except Exception as e:
                print(f"   ❌ {model_type} prediction failed: {e}")
                continue
        
        if predictions:
            print(f"   ✅ Predictions successful: {len(predictions)} model predictions")
            return predictions
        else:
            print("   ❌ No predictions generated")
            return None
            
    except Exception as e:
        print(f"   ❌ Predictions failed: {e}")
        return None

def test_evaluation(models):
    """Test model evaluation functionality"""
    print("🧪 Testing Model Evaluation...")
    
    if models is None:
        print("   ⏭️ Skipping (no models)")
        return None
    
    try:
        from evaluator import evaluate_model
        
        evaluations = {}
        
        for model_type, model_info in models.items():
            if 'y_test' in model_info and 'y_pred_test' in model_info:
                try:
                    print(f"   🔄 Evaluating {model_type}...")
                    
                    result = evaluate_model(
                        model_info['y_test'], 
                        model_info['y_pred_test'], 
                        model_type
                    )
                    
                    # Validate evaluation
                    assert 'metrics' in result, f"{model_type} metrics missing"
                    assert 'r2' in result['metrics'], f"{model_type} R² metric missing"
                    
                    evaluations[model_type] = result
                    print(f"   ✅ {model_type} evaluated: R² = {result['metrics']['r2']:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ {model_type} evaluation failed: {e}")
                    continue
        
        if evaluations:
            print(f"   ✅ Evaluation successful: {len(evaluations)} models evaluated")
            return evaluations
        else:
            print("   ❌ No evaluations completed")
            return None
            
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        return None

def test_visualizations(data, predictions):
    """Test visualization functionality"""
    print("🧪 Testing Visualizations...")
    
    if data is None:
        print("   ⏭️ Skipping (no data)")
        return None
    
    try:
        from visualizer import generate_visuals
        
        # Prepare predictions for visualization
        viz_predictions = {}
        if predictions:
            for model_type, pred_df in predictions.items():
                viz_predictions['temperature'] = pred_df
                break  # Use first available prediction
        
        visuals = generate_visuals(
            data, 
            predictions=viz_predictions,
            variables=['temperature', 'rainfall']
        )
        
        # Validate visualizations
        assert isinstance(visuals, dict), "Visuals should be a dictionary"
        assert len(visuals) > 0, "Should have at least one visualization"
        
        print(f"   ✅ Visualizations successful: {len(visuals)} plots created")
        return visuals
        
    except Exception as e:
        print(f"   ❌ Visualizations failed: {e}")
        return None

def test_report_generation(data, predictions, evaluations):
    """Test report generation functionality"""
    print("🧪 Testing Report Generation...")
    
    if data is None:
        print("   ⏭️ Skipping (no data)")
        return None
    
    try:
        from report_generator import generate_report
        
        # Prepare model results for report
        model_results = None
        if evaluations:
            model_results = {
                'temperature': {
                    'individual_results': evaluations,
                    'best_model': list(evaluations.keys())[0] if evaluations else 'Unknown'
                }
            }
        
        # Prepare predictions for report
        report_predictions = {}
        if predictions:
            for model_type, pred_df in predictions.items():
                report_predictions['temperature'] = pred_df
                break  # Use first available prediction
        
        report_path = generate_report(
            data=data,
            predictions=report_predictions,
            model_results=model_results,
            output_path="test_climate_report.pdf"
        )
        
        # Validate report
        assert os.path.exists(report_path), "Report file should exist"
        assert os.path.getsize(report_path) > 1000, "Report should have substantial content"
        
        print(f"   ✅ Report generation successful: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"   ❌ Report generation failed: {e}")
        return None

def run_comprehensive_test():
    """Run comprehensive backend test"""
    print("🚀 PUNE CLIMATE PREDICTION SYSTEM - BACKEND TEST")
    print("=" * 60)
    
    # Test results tracking
    test_results = {
        'data_collection': False,
        'preprocessing': False,
        'model_training': False,
        'predictions': False,
        'evaluation': False,
        'visualizations': False,
        'report_generation': False
    }
    
    # 1. Test Data Collection
    data = test_data_collection()
    test_results['data_collection'] = data is not None
    
    # 2. Test Data Preprocessing
    processed_data = test_data_preprocessing(data)
    test_results['preprocessing'] = processed_data is not None
    
    # 3. Test Model Training
    models = test_model_training(processed_data)
    test_results['model_training'] = models is not None
    
    # 4. Test Predictions
    predictions = test_predictions(models, processed_data)
    test_results['predictions'] = predictions is not None
    
    # 5. Test Evaluation
    evaluations = test_evaluation(models)
    test_results['evaluation'] = evaluations is not None
    
    # 6. Test Visualizations
    visuals = test_visualizations(data, predictions)
    test_results['visualizations'] = visuals is not None
    
    # 7. Test Report Generation
    report_path = test_report_generation(data, predictions, evaluations)
    test_results['report_generation'] = report_path is not None
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Backend is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

def main():
    """Main test function"""
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()