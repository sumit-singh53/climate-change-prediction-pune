#!/usr/bin/env python3
"""
Demo script for the Pune Climate Dashboard
Shows key functionality without full Streamlit interface
"""

import pandas as pd
import numpy as np
import asyncio
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def demo_data_collection():
    """Demo data collection"""
    print("📊 DEMO: Data Collection")
    print("-" * 40)
    
    try:
        from data_collector import fetch_city_data
        
        # Fetch sample data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(
            fetch_city_data("Pune", 2022, 2024, include_current=True)
        )
        loop.close()
        
        print(f"✅ Collected {len(data):,} climate records")
        print(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"🌡️ Temperature range: {data['temperature'].min():.1f}°C to {data['temperature'].max():.1f}°C")
        print(f"🌧️ Rainfall range: {data['rainfall'].min():.1f}mm to {data['rainfall'].max():.1f}mm")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def demo_model_training(data):
    """Demo model training"""
    print("\n🤖 DEMO: Model Training")
    print("-" * 40)
    
    try:
        from data_preprocessor import clean_and_preprocess
        from model_trainer import train_model
        
        # Preprocess data
        print("🔧 Preprocessing data...")
        results = clean_and_preprocess(
            data,
            target_variables=['temperature'],
            scaling_method='robust'
        )
        
        processed_data = results['final']
        print(f"✅ Preprocessed {len(processed_data)} records")
        
        # Train model
        print("🚀 Training Linear Regression model...")
        model_info = train_model(
            processed_data,
            'temperature',
            'linear',
            optimize=False
        )
        
        print(f"✅ Model trained successfully!")
        print(f"📊 R² Score: {model_info.get('test_r2', 0):.3f}")
        print(f"📊 RMSE: {model_info.get('test_rmse', 0):.3f}")
        print(f"📊 MAE: {model_info.get('test_mae', 0):.3f}")
        
        return model_info, processed_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def demo_predictions(model_info, processed_data):
    """Demo predictions"""
    print("\n🔮 DEMO: Future Predictions")
    print("-" * 40)
    
    try:
        from predictor import predict_future
        
        # Generate predictions
        future_years = [2025, 2026, 2027, 2028, 2029, 2030]
        print(f"🎯 Predicting temperature for years: {future_years}")
        
        predictions = predict_future(model_info, future_years, processed_data)
        
        print(f"✅ Generated {len(predictions)} predictions")
        
        # Show sample predictions
        if 'temperature_predicted' in predictions.columns:
            avg_pred = predictions['temperature_predicted'].mean()
            min_pred = predictions['temperature_predicted'].min()
            max_pred = predictions['temperature_predicted'].max()
            
            print(f"🌡️ Average predicted temperature: {avg_pred:.1f}°C")
            print(f"🌡️ Temperature range: {min_pred:.1f}°C to {max_pred:.1f}°C")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def demo_visualizations(data, predictions):
    """Demo visualizations"""
    print("\n📈 DEMO: Visualizations")
    print("-" * 40)
    
    try:
        from visualization import ClimateVisualizationEngine
        
        visualizer = ClimateVisualizationEngine()
        
        # Create visualizations
        pred_dict = {'temperature': predictions} if not predictions.empty else None
        
        print("🎨 Creating interactive visualizations...")
        visuals = visualizer.create_comprehensive_dashboard(data, pred_dict)
        
        print(f"✅ Created {len(visuals)} visualizations:")
        for chart_name in visuals.keys():
            print(f"   📊 {chart_name.replace('_', ' ').title()}")
        
        return visuals
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def demo_ai_insights(data, predictions):
    """Demo AI insights"""
    print("\n🧠 DEMO: AI Insights")
    print("-" * 40)
    
    try:
        # Simple insights generation
        insights = []
        
        # Temperature insights
        if 'temperature' in data.columns:
            temp_mean = data['temperature'].mean()
            insights.append(f"Average temperature in Pune is {temp_mean:.1f}°C")
            
            if 'year' in data.columns:
                yearly_temp = data.groupby('year')['temperature'].mean()
                if len(yearly_temp) > 1:
                    temp_trend = yearly_temp.diff().mean()
                    if temp_trend > 0.1:
                        insights.append(f"Temperature is rising at {temp_trend:.2f}°C per year")
                    elif temp_trend < -0.1:
                        insights.append(f"Temperature is cooling at {abs(temp_trend):.2f}°C per year")
        
        # Rainfall insights
        if 'rainfall' in data.columns:
            total_rainfall = data['rainfall'].sum()
            insights.append(f"Total rainfall recorded: {total_rainfall:.0f}mm")
        
        # AQI insights
        if 'aqi' in data.columns:
            avg_aqi = data['aqi'].mean()
            if avg_aqi > 100:
                insights.append(f"Air quality is concerning with AQI of {avg_aqi:.0f}")
            else:
                insights.append(f"Air quality is moderate with AQI of {avg_aqi:.0f}")
        
        # Prediction insights
        if not predictions.empty and 'temperature_predicted' in predictions.columns:
            future_temp = predictions['temperature_predicted'].mean()
            current_temp = data['temperature'].tail(365).mean()
            change = future_temp - current_temp
            
            if abs(change) > 0.5:
                direction = "increase" if change > 0 else "decrease"
                insights.append(f"Temperature may {direction} by {abs(change):.1f}°C in coming years")
        
        print("🎯 Generated AI Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        return insights
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def demo_risk_assessment(data):
    """Demo risk assessment"""
    print("\n⚠️ DEMO: Climate Risk Assessment")
    print("-" * 40)
    
    try:
        # Calculate risk factors
        risk_score = 0
        risk_factors = []
        
        # Temperature risk
        if 'temperature' in data.columns:
            temp_mean = data['temperature'].mean()
            if temp_mean > 32:
                risk_score += 30
                risk_factors.append("High temperature levels")
            elif temp_mean > 28:
                risk_score += 15
                risk_factors.append("Elevated temperature levels")
        
        # AQI risk
        if 'aqi' in data.columns:
            aqi_mean = data['aqi'].mean()
            if aqi_mean > 100:
                risk_score += 25
                risk_factors.append("Poor air quality")
            elif aqi_mean > 75:
                risk_score += 15
                risk_factors.append("Moderate air quality concerns")
        
        # Rainfall variability risk
        if 'rainfall' in data.columns and 'year' in data.columns:
            yearly_rainfall = data.groupby('year')['rainfall'].sum()
            if len(yearly_rainfall) > 1:
                rainfall_cv = yearly_rainfall.std() / yearly_rainfall.mean()
                if rainfall_cv > 0.3:
                    risk_score += 20
                    risk_factors.append("High rainfall variability")
        
        # Determine risk level
        if risk_score < 25:
            risk_level = "🟢 LOW"
        elif risk_score < 50:
            risk_level = "🟠 MODERATE"
        else:
            risk_level = "🔴 HIGH"
        
        print(f"🎯 Climate Risk Level: {risk_level}")
        print(f"📊 Risk Score: {risk_score}/100")
        
        if risk_factors:
            print("⚠️ Risk Factors:")
            for factor in risk_factors:
                print(f"   • {factor}")
        else:
            print("✅ No significant risk factors identified")
        
        return risk_score, risk_level, risk_factors
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0, "Unknown", []

def demo_report_generation(data, predictions):
    """Demo report generation"""
    print("\n📄 DEMO: Report Generation")
    print("-" * 40)
    
    try:
        from report_generator import generate_report
        
        # Generate report
        pred_dict = {'temperature': predictions} if not predictions.empty else None
        
        print("📝 Generating PDF report...")
        report_path = generate_report(
            data,
            predictions=pred_dict,
            output_path=f"demo_climate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if os.path.exists(report_path):
            file_size = os.path.getsize(report_path)
            print(f"✅ Report generated successfully!")
            print(f"📁 File: {report_path}")
            print(f"📊 Size: {file_size:,} bytes")
            
            return report_path
        else:
            print("❌ Report file not found")
            return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Run complete dashboard demo"""
    print("🌡️ PUNE CLIMATE DASHBOARD - COMPLETE DEMO")
    print("=" * 60)
    print("This demo shows all dashboard features without the web interface")
    print("=" * 60)
    
    # Demo 1: Data Collection
    data = demo_data_collection()
    if data.empty:
        print("❌ Cannot continue without data")
        return
    
    # Demo 2: Model Training
    model_info, processed_data = demo_model_training(data)
    
    # Demo 3: Predictions
    if model_info:
        predictions = demo_predictions(model_info, processed_data)
    else:
        predictions = pd.DataFrame()
    
    # Demo 4: Visualizations
    visuals = demo_visualizations(data, predictions)
    
    # Demo 5: AI Insights
    insights = demo_ai_insights(data, predictions)
    
    # Demo 6: Risk Assessment
    risk_score, risk_level, risk_factors = demo_risk_assessment(data)
    
    # Demo 7: Report Generation
    report_path = demo_report_generation(data, predictions)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE - DASHBOARD FEATURES SUMMARY")
    print("=" * 60)
    
    print(f"📊 Data Records: {len(data):,}")
    print(f"🤖 Model Training: {'✅ Success' if model_info else '❌ Failed'}")
    print(f"🔮 Predictions: {'✅ Generated' if not predictions.empty else '❌ Failed'}")
    print(f"📈 Visualizations: {len(visuals)} charts created")
    print(f"🧠 AI Insights: {len(insights)} insights generated")
    print(f"⚠️ Risk Assessment: {risk_level}")
    print(f"📄 Report: {'✅ Generated' if report_path else '❌ Failed'}")
    
    print("\n🚀 To launch the full interactive dashboard:")
    print("   python run_dashboard.py")
    print("   OR")
    print("   streamlit run streamlit_dashboard.py")
    
    print("\n📚 For detailed usage instructions:")
    print("   See DASHBOARD_README.md")

if __name__ == "__main__":
    main()