#!/usr/bin/env python3
"""
Simple Climate Change Prediction Demo
Demonstrates climate prediction with realistic results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import sys
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from data_collector import fetch_city_data

def create_simple_features(data):
    """Create simple, effective features for prediction"""
    df = data.copy()
    
    # Basic time features
    df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Climate features
    if 'is_monsoon' in df.columns:
        df['is_monsoon'] = df['is_monsoon'].astype(int)
    
    return df

def train_simple_model(data, target):
    """Train a simple but effective model"""
    print(f"🤖 Training model for {target}...")
    
    # Create features
    df = create_simple_features(data)
    
    # Select features
    feature_cols = ['year_norm', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
    
    # Add other climate variables as features (excluding target)
    climate_vars = ['temperature', 'rainfall', 'humidity', 'aqi', 'wind_speed', 'pressure']
    for var in climate_vars:
        if var in df.columns and var != target:
            feature_cols.append(var)
    
    if 'is_monsoon' in df.columns:
        feature_cols.append('is_monsoon')
    
    # Prepare data
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target].fillna(df[target].mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"   📊 Model Performance:")
    print(f"      R² Score: {r2:.3f}")
    print(f"      MAE: {mae:.3f}")
    print(f"      RMSE: {rmse:.3f}")
    
    return {
        'model': model,
        'features': feature_cols,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'target_mean': y.mean(),
        'target_std': y.std()
    }

def predict_future_simple(model_info, data, future_years):
    """Generate realistic future predictions"""
    print(f"🔮 Generating predictions for {len(future_years)} years...")
    
    model = model_info['model']
    features = model_info['features']
    target_mean = model_info['target_mean']
    target_std = model_info['target_std']
    
    # Create future dates
    future_dates = []
    for year in future_years:
        year_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='M')
        future_dates.extend(year_dates)
    
    # Create future dataframe
    future_df = pd.DataFrame({
        'date': future_dates,
        'year': [d.year for d in future_dates],
        'month': [d.month for d in future_dates],
        'day': [15 for _ in future_dates]  # Mid-month
    })
    
    # Add climate change trends
    base_year = data['year'].max()
    years_ahead = future_df['year'] - base_year
    
    # Estimate other climate variables based on historical patterns
    historical_monthly = data.groupby('month').agg({
        'temperature': 'mean',
        'rainfall': 'mean', 
        'humidity': 'mean',
        'aqi': 'mean',
        'wind_speed': 'mean',
        'pressure': 'mean'
    })
    
    for month in range(1, 13):
        month_mask = future_df['month'] == month
        
        # Add historical averages for each month
        for var in ['temperature', 'rainfall', 'humidity', 'aqi', 'wind_speed', 'pressure']:
            if var in historical_monthly.columns:
                base_value = historical_monthly.loc[month, var]
                
                # Add climate change trends
                if var == 'temperature':
                    # Warming trend: 0.02°C per year
                    trend = years_ahead[month_mask] * 0.02
                elif var == 'rainfall':
                    # Slight decrease: -0.5% per year
                    trend = base_value * years_ahead[month_mask] * -0.005
                elif var == 'aqi':
                    # Improvement: -0.5 points per year
                    trend = years_ahead[month_mask] * -0.5
                else:
                    trend = 0
                
                future_df.loc[month_mask, var] = base_value + trend
    
    # Add monsoon indicator
    future_df['is_monsoon'] = future_df['month'].isin([6, 7, 8, 9]).astype(int)
    
    # Create features
    future_df = create_simple_features(future_df)
    
    # Make predictions
    X_future = future_df[features].fillna(0)
    predictions = model.predict(X_future)
    
    # Add some realistic bounds
    predictions = np.clip(predictions, 
                         target_mean - 3*target_std, 
                         target_mean + 3*target_std)
    
    future_df['predicted'] = predictions
    
    return future_df

def demonstrate_climate_prediction():
    """Main demonstration"""
    print("🌡️ SIMPLE CLIMATE CHANGE PREDICTION DEMO")
    print("=" * 60)
    
    # Load data
    print("📊 Loading climate data...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(
        fetch_city_data("Pune", 2020, 2024, include_current=True)
    )
    loop.close()
    
    print(f"✅ Loaded {len(data):,} records from {data['date'].min().year} to {data['date'].max().year}")
    
    # Analyze historical trends
    print("\n📈 Historical Climate Analysis:")
    
    # Temperature trend
    yearly_temp = data.groupby('year')['temperature'].mean()
    temp_trend = np.polyfit(yearly_temp.index, yearly_temp.values, 1)[0]
    print(f"   🌡️ Temperature: {data['temperature'].mean():.1f}°C average, {temp_trend:+.3f}°C/year trend")
    
    # Rainfall trend
    yearly_rain = data.groupby('year')['rainfall'].sum()
    rain_trend = np.polyfit(yearly_rain.index, yearly_rain.values, 1)[0]
    print(f"   🌧️ Rainfall: {yearly_rain.mean():.0f}mm/year average, {rain_trend:+.1f}mm/year trend")
    
    # Train models
    print("\n🤖 Training Prediction Models:")
    
    temp_model = train_simple_model(data, 'temperature')
    rain_model = train_simple_model(data, 'rainfall')
    
    # Generate predictions
    print("\n🔮 Generating Future Predictions:")
    
    future_years = [2025, 2030, 2035, 2040, 2045, 2050]
    
    temp_predictions = predict_future_simple(temp_model, data, future_years)
    rain_predictions = predict_future_simple(rain_model, data, future_years)
    
    # Analyze predictions
    print("\n📊 Climate Change Projections:")
    
    # Temperature analysis
    current_temp = data['temperature'].mean()
    future_temp_2030 = temp_predictions[temp_predictions['year'] == 2030]['predicted'].mean()
    future_temp_2050 = temp_predictions[temp_predictions['year'] == 2050]['predicted'].mean()
    
    temp_change_2030 = future_temp_2030 - current_temp
    temp_change_2050 = future_temp_2050 - current_temp
    
    print(f"   🌡️ Temperature Projections:")
    print(f"      Current (2024): {current_temp:.1f}°C")
    print(f"      2030: {future_temp_2030:.1f}°C ({temp_change_2030:+.1f}°C)")
    print(f"      2050: {future_temp_2050:.1f}°C ({temp_change_2050:+.1f}°C)")
    
    # Rainfall analysis
    current_rain = data.groupby('year')['rainfall'].sum().mean()
    future_rain_2030 = rain_predictions[rain_predictions['year'] == 2030]['predicted'].sum()
    future_rain_2050 = rain_predictions[rain_predictions['year'] == 2050]['predicted'].sum()
    
    rain_change_2030 = ((future_rain_2030 - current_rain) / current_rain) * 100
    rain_change_2050 = ((future_rain_2050 - current_rain) / current_rain) * 100
    
    print(f"   🌧️ Rainfall Projections:")
    print(f"      Current: {current_rain:.0f}mm/year")
    print(f"      2030: {future_rain_2030:.0f}mm/year ({rain_change_2030:+.1f}%)")
    print(f"      2050: {future_rain_2050:.0f}mm/year ({rain_change_2050:+.1f}%)")
    
    # Climate impact assessment
    print("\n⚠️ Climate Impact Assessment:")
    
    risk_score = 0
    impacts = []
    
    if temp_change_2050 > 2:
        risk_score += 40
        impacts.append(f"Severe warming: +{temp_change_2050:.1f}°C by 2050")
    elif temp_change_2050 > 1:
        risk_score += 25
        impacts.append(f"Significant warming: +{temp_change_2050:.1f}°C by 2050")
    elif temp_change_2050 > 0.5:
        risk_score += 15
        impacts.append(f"Moderate warming: +{temp_change_2050:.1f}°C by 2050")
    
    if abs(rain_change_2050) > 20:
        risk_score += 25
        direction = "increase" if rain_change_2050 > 0 else "decrease"
        impacts.append(f"Major rainfall change: {abs(rain_change_2050):.1f}% {direction}")
    elif abs(rain_change_2050) > 10:
        risk_score += 15
        direction = "increase" if rain_change_2050 > 0 else "decrease"
        impacts.append(f"Moderate rainfall change: {abs(rain_change_2050):.1f}% {direction}")
    
    # Air quality
    avg_aqi = data['aqi'].mean()
    if avg_aqi > 100:
        risk_score += 20
        impacts.append(f"Poor air quality: AQI {avg_aqi:.0f}")
    
    # Risk level
    if risk_score < 25:
        risk_level = "🟢 LOW"
    elif risk_score < 50:
        risk_level = "🟠 MODERATE"
    else:
        risk_level = "🔴 HIGH"
    
    print(f"   Risk Level: {risk_level}")
    print(f"   Risk Score: {risk_score}/100")
    print(f"   Key Impacts:")
    for impact in impacts:
        print(f"      • {impact}")
    
    # Recommendations
    print("\n💡 Climate Adaptation Recommendations:")
    
    recommendations = []
    
    if temp_change_2050 > 1:
        recommendations.append("🌳 Increase urban green cover by 30% to mitigate heat")
        recommendations.append("🏢 Implement cool roofing and building standards")
    
    if abs(rain_change_2050) > 10:
        recommendations.append("💧 Develop adaptive water management systems")
        recommendations.append("🌊 Improve flood/drought preparedness")
    
    if avg_aqi > 75:
        recommendations.append("🚗 Promote electric vehicles and clean transport")
        recommendations.append("🏭 Strengthen industrial emission controls")
    
    recommendations.extend([
        "📊 Establish continuous climate monitoring",
        "🎓 Develop community climate awareness programs",
        "🏛️ Integrate climate considerations into urban planning"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    print(f"📊 Analysis: {len(data):,} historical records")
    print(f"🤖 Models: Temperature (R²={temp_model['r2']:.3f}), Rainfall (R²={rain_model['r2']:.3f})")
    print(f"🔮 Projections: {len(future_years)} time periods (2025-2050)")
    print(f"⚠️ Risk: {risk_level.split()[1]} ({risk_score}/100)")
    print(f"💡 Actions: {len(recommendations)} recommendations")
    
    if temp_change_2050 > 1 or abs(rain_change_2050) > 15:
        print(f"\n🌍 CONCLUSION: Climate change impacts expected for Pune")
        print(f"   Proactive adaptation measures recommended")
    else:
        print(f"\n🌍 CONCLUSION: Moderate climate changes expected")
        print(f"   Continue monitoring and maintain preparedness")
    
    return {
        'temp_change_2050': temp_change_2050,
        'rain_change_2050': rain_change_2050,
        'risk_score': risk_score,
        'models': {'temperature': temp_model, 'rainfall': rain_model}
    }

if __name__ == "__main__":
    try:
        results = demonstrate_climate_prediction()
        print(f"\n✅ Climate prediction demonstration completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()