#!/usr/bin/env python3
"""
Climate Change Prediction Demonstration
Shows how the system predicts climate change using real data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from data_collector import fetch_city_data
from data_preprocessor import clean_and_preprocess
from model_trainer import train_model
from predictor import predict_future
from evaluator import evaluate_model

def analyze_climate_trends(data):
    """Analyze historical climate trends"""
    print("📊 ANALYZING HISTORICAL CLIMATE TRENDS")
    print("=" * 60)
    
    # Temperature trends
    if 'temperature' in data.columns and 'year' in data.columns:
        yearly_temp = data.groupby('year')['temperature'].mean()
        temp_trend = np.polyfit(yearly_temp.index, yearly_temp.values, 1)[0]
        
        print(f"🌡️ TEMPERATURE ANALYSIS:")
        print(f"   📈 Average temperature: {data['temperature'].mean():.1f}°C")
        print(f"   📊 Temperature range: {data['temperature'].min():.1f}°C to {data['temperature'].max():.1f}°C")
        print(f"   📈 Annual trend: {temp_trend:.3f}°C per year")
        
        if temp_trend > 0.05:
            print(f"   🔥 WARMING DETECTED: {temp_trend:.3f}°C/year increase")
        elif temp_trend < -0.05:
            print(f"   🧊 COOLING DETECTED: {abs(temp_trend):.3f}°C/year decrease")
        else:
            print(f"   ➡️ STABLE: Temperature relatively stable")
    
    # Rainfall trends
    if 'rainfall' in data.columns and 'year' in data.columns:
        yearly_rainfall = data.groupby('year')['rainfall'].sum()
        rainfall_trend = np.polyfit(yearly_rainfall.index, yearly_rainfall.values, 1)[0]
        rainfall_cv = yearly_rainfall.std() / yearly_rainfall.mean()
        
        print(f"\n🌧️ RAINFALL ANALYSIS:")
        print(f"   📊 Average annual rainfall: {yearly_rainfall.mean():.0f}mm")
        print(f"   📈 Rainfall trend: {rainfall_trend:.1f}mm per year")
        print(f"   📊 Variability (CV): {rainfall_cv:.3f}")
        
        if abs(rainfall_trend) > 10:
            direction = "increasing" if rainfall_trend > 0 else "decreasing"
            print(f"   🌧️ CHANGE DETECTED: Rainfall {direction} by {abs(rainfall_trend):.1f}mm/year")
        
        if rainfall_cv > 0.3:
            print(f"   ⚠️ HIGH VARIABILITY: Irregular rainfall patterns detected")
    
    # Air quality trends
    if 'aqi' in data.columns:
        avg_aqi = data['aqi'].mean()
        print(f"\n💨 AIR QUALITY ANALYSIS:")
        print(f"   📊 Average AQI: {avg_aqi:.0f}")
        
        if avg_aqi > 150:
            print(f"   🚨 SEVERE: Very unhealthy air quality")
        elif avg_aqi > 100:
            print(f"   ⚠️ POOR: Unhealthy air quality")
        elif avg_aqi > 50:
            print(f"   🟡 MODERATE: Acceptable air quality")
        else:
            print(f"   ✅ GOOD: Healthy air quality")
    
    return {
        'temp_trend': temp_trend if 'temperature' in data.columns else 0,
        'rainfall_trend': rainfall_trend if 'rainfall' in data.columns else 0,
        'avg_aqi': avg_aqi if 'aqi' in data.columns else 0
    }

def demonstrate_climate_prediction():
    """Main demonstration of climate prediction capabilities"""
    print("🌡️ CLIMATE CHANGE PREDICTION DEMONSTRATION")
    print("=" * 70)
    print("This demo shows how we predict climate change using machine learning")
    print("=" * 70)
    
    # Step 1: Load historical data
    print("\n📊 STEP 1: LOADING HISTORICAL CLIMATE DATA")
    print("-" * 50)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(
        fetch_city_data("Pune", 2020, 2024, include_current=True)
    )
    loop.close()
    
    if data.empty:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(data):,} climate records")
    print(f"📅 Period: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    print(f"📊 Variables: {', '.join([col for col in data.columns if col in ['temperature', 'rainfall', 'humidity', 'aqi']])}")
    
    # Step 2: Analyze historical trends
    print("\n📈 STEP 2: ANALYZING HISTORICAL TRENDS")
    print("-" * 50)
    
    trends = analyze_climate_trends(data)
    
    # Step 3: Prepare data for machine learning
    print("\n🔧 STEP 3: PREPARING DATA FOR MACHINE LEARNING")
    print("-" * 50)
    
    processed_results = clean_and_preprocess(
        data,
        target_variables=['temperature', 'rainfall'],
        scaling_method='robust',
        create_features=True
    )
    
    if not processed_results:
        print("❌ Data preprocessing failed")
        return
    
    processed_data = processed_results['final']
    print(f"✅ Data preprocessed: {len(processed_data)} records, {len(processed_data.columns)} features")
    
    # Step 4: Train prediction models
    print("\n🤖 STEP 4: TRAINING CLIMATE PREDICTION MODELS")
    print("-" * 50)
    
    models = {}
    
    # Train temperature prediction model
    print("🌡️ Training temperature prediction model...")
    temp_model = train_model(
        processed_data,
        'temperature',
        'random_forest',  # Use Random Forest for better performance
        optimize=False
    )
    models['temperature'] = temp_model
    
    print(f"   📊 Temperature Model Performance:")
    print(f"      R² Score: {temp_model.get('test_r2', 0):.3f}")
    print(f"      RMSE: {temp_model.get('test_rmse', 0):.3f}°C")
    print(f"      MAE: {temp_model.get('test_mae', 0):.3f}°C")
    
    # Train rainfall prediction model
    print("\n🌧️ Training rainfall prediction model...")
    rain_model = train_model(
        processed_data,
        'rainfall',
        'random_forest',
        optimize=False
    )
    models['rainfall'] = rain_model
    
    print(f"   📊 Rainfall Model Performance:")
    print(f"      R² Score: {rain_model.get('test_r2', 0):.3f}")
    print(f"      RMSE: {rain_model.get('test_rmse', 0):.3f}mm")
    print(f"      MAE: {rain_model.get('test_mae', 0):.3f}mm")
    
    # Step 5: Generate future predictions
    print("\n🔮 STEP 5: GENERATING CLIMATE PREDICTIONS (2025-2050)")
    print("-" * 50)
    
    future_years = [2025, 2030, 2035, 2040, 2045, 2050]
    predictions = {}
    
    for variable, model in models.items():
        print(f"🎯 Predicting {variable} for {len(future_years)} future periods...")
        
        pred_data = predict_future(model, future_years, processed_data)
        predictions[variable] = pred_data
        
        if not pred_data.empty and f'{variable}_predicted' in pred_data.columns:
            # Calculate prediction summary
            current_avg = data[variable].tail(365).mean()  # Last year average
            future_avg = pred_data[f'{variable}_predicted'].mean()
            change = future_avg - current_avg
            change_pct = (change / current_avg) * 100 if current_avg != 0 else 0
            
            print(f"   📊 {variable.title()} Predictions:")
            print(f"      Current average: {current_avg:.2f}")
            print(f"      Future average: {future_avg:.2f}")
            print(f"      Predicted change: {change:+.2f} ({change_pct:+.1f}%)")
            
            if variable == 'temperature':
                if change > 2:
                    print(f"      🔥 SIGNIFICANT WARMING: {change:.1f}°C increase predicted")
                elif change > 1:
                    print(f"      🌡️ MODERATE WARMING: {change:.1f}°C increase predicted")
                elif change < -1:
                    print(f"      🧊 COOLING: {abs(change):.1f}°C decrease predicted")
                else:
                    print(f"      ➡️ STABLE: Temperature relatively stable")
            
            elif variable == 'rainfall':
                if abs(change_pct) > 20:
                    direction = "increase" if change > 0 else "decrease"
                    print(f"      🌧️ SIGNIFICANT CHANGE: {abs(change_pct):.1f}% {direction} in rainfall")
                elif abs(change_pct) > 10:
                    direction = "increase" if change > 0 else "decrease"
                    print(f"      🌦️ MODERATE CHANGE: {abs(change_pct):.1f}% {direction} in rainfall")
                else:
                    print(f"      ➡️ STABLE: Rainfall patterns relatively stable")
    
    # Step 6: Climate change impact assessment
    print("\n⚠️ STEP 6: CLIMATE CHANGE IMPACT ASSESSMENT")
    print("-" * 50)
    
    # Calculate climate risk score
    risk_score = 0
    risk_factors = []
    
    # Temperature risk
    if 'temperature' in predictions:
        temp_pred = predictions['temperature']
        if not temp_pred.empty and 'temperature_predicted' in temp_pred.columns:
            temp_increase = temp_pred['temperature_predicted'].mean() - data['temperature'].mean()
            if temp_increase > 3:
                risk_score += 40
                risk_factors.append(f"Severe warming: +{temp_increase:.1f}°C")
            elif temp_increase > 2:
                risk_score += 30
                risk_factors.append(f"Significant warming: +{temp_increase:.1f}°C")
            elif temp_increase > 1:
                risk_score += 20
                risk_factors.append(f"Moderate warming: +{temp_increase:.1f}°C")
    
    # Air quality risk
    if 'aqi' in data.columns:
        avg_aqi = data['aqi'].mean()
        if avg_aqi > 150:
            risk_score += 30
            risk_factors.append("Severe air pollution")
        elif avg_aqi > 100:
            risk_score += 20
            risk_factors.append("Poor air quality")
    
    # Rainfall variability risk
    if 'rainfall' in data.columns and 'year' in data.columns:
        yearly_rainfall = data.groupby('year')['rainfall'].sum()
        if len(yearly_rainfall) > 1:
            rainfall_cv = yearly_rainfall.std() / yearly_rainfall.mean()
            if rainfall_cv > 0.4:
                risk_score += 20
                risk_factors.append("High rainfall variability")
    
    # Determine risk level
    if risk_score < 30:
        risk_level = "🟢 LOW"
        risk_color = "Low"
    elif risk_score < 60:
        risk_level = "🟠 MODERATE"
        risk_color = "Moderate"
    else:
        risk_level = "🔴 HIGH"
        risk_color = "High"
    
    print(f"📊 CLIMATE RISK ASSESSMENT:")
    print(f"   Risk Level: {risk_level}")
    print(f"   Risk Score: {risk_score}/100")
    print(f"   Risk Factors:")
    for factor in risk_factors:
        print(f"      • {factor}")
    
    if not risk_factors:
        print(f"      • No significant risk factors identified")
    
    # Step 7: Recommendations
    print("\n💡 STEP 7: CLIMATE ADAPTATION RECOMMENDATIONS")
    print("-" * 50)
    
    recommendations = []
    
    if risk_score >= 60:
        recommendations.extend([
            "🚨 Implement emergency climate action plan",
            "🌳 Massive urban forestry program (plant 50,000+ trees)",
            "💧 Develop comprehensive water management systems",
            "🏥 Establish climate health monitoring and early warning systems"
        ])
    elif risk_score >= 30:
        recommendations.extend([
            "⚠️ Develop proactive climate adaptation strategies",
            "🌱 Increase urban green cover by 25%",
            "🚌 Improve public transportation to reduce emissions",
            "🏢 Implement green building standards"
        ])
    else:
        recommendations.extend([
            "✅ Continue monitoring climate trends",
            "🌿 Maintain existing environmental protection measures",
            "📊 Regular climate risk assessments"
        ])
    
    # Add specific recommendations based on predictions
    if 'temperature' in predictions:
        temp_pred = predictions['temperature']
        if not temp_pred.empty and 'temperature_predicted' in temp_pred.columns:
            temp_increase = temp_pred['temperature_predicted'].mean() - data['temperature'].mean()
            if temp_increase > 1:
                recommendations.append(f"🌡️ Heat mitigation: Increase green cover to offset {temp_increase:.1f}°C warming")
    
    print("📋 RECOMMENDED ACTIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 CLIMATE PREDICTION SUMMARY")
    print("=" * 70)
    
    print(f"📊 Data Analysis: {len(data):,} historical records analyzed")
    print(f"🤖 Models Trained: {len(models)} prediction models")
    print(f"🔮 Predictions: Generated forecasts for {len(future_years)} time periods")
    print(f"⚠️ Risk Level: {risk_color} ({risk_score}/100)")
    print(f"💡 Recommendations: {len(recommendations)} adaptation strategies")
    
    print(f"\n🌍 CONCLUSION:")
    if risk_score >= 60:
        print("🔴 HIGH RISK: Urgent climate action required for Pune")
    elif risk_score >= 30:
        print("🟠 MODERATE RISK: Proactive climate adaptation needed")
    else:
        print("🟢 LOW RISK: Continue monitoring and maintain current measures")
    
    print(f"\n🚀 The climate prediction system successfully demonstrates:")
    print(f"   ✅ Historical trend analysis")
    print(f"   ✅ Machine learning model training")
    print(f"   ✅ Future climate predictions")
    print(f"   ✅ Risk assessment and recommendations")
    
    return {
        'data': data,
        'models': models,
        'predictions': predictions,
        'risk_score': risk_score,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    try:
        results = demonstrate_climate_prediction()
        print(f"\n✅ Climate prediction demonstration completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()