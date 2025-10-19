"""
Prediction Module
Generates climate forecasts for user-selected years (2026–2050)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import joblib
from prophet import Prophet
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ClimatePredictor:
    """Advanced climate prediction system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
    
    def load_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Load a trained model from disk"""
        try:
            if model_type == 'lstm':
                # Load LSTM model and associated info
                model = load_model(f"{model_path}_lstm.h5")
                info = joblib.load(f"{model_path}_lstm_info.pkl")
                return {
                    'model': model,
                    'model_type': 'lstm',
                    'scaler': info['scaler'],
                    'sequence_length': info['sequence_length'],
                    'target': info['target']
                }
            else:
                # Load sklearn/prophet models
                return joblib.load(model_path)
        except Exception as e:
            print(f"❌ Error loading model from {model_path}: {e}")
            return None
    
    def prepare_future_features(self, last_date: datetime, future_years: List[int],
                               historical_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for future predictions"""
        print(f"📅 Preparing features for years: {future_years}")
        
        # Create future date range
        start_date = datetime(min(future_years), 1, 1)
        end_date = datetime(max(future_years), 12, 31)
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize future dataframe
        future_df = pd.DataFrame({'date': future_dates})
        
        # Add basic time features
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['day'] = future_df['date'].dt.day
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['day_of_year'] = future_df['date'].dt.dayofyear
        future_df['week_of_year'] = future_df['date'].dt.isocalendar().week
        future_df['quarter'] = future_df['date'].dt.quarter
        
        # Cyclical encoding
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365)
        future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365)
        future_df['week_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['week_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        
        # Season indicators
        future_df['is_winter'] = future_df['month'].isin([12, 1, 2]).astype(int)
        future_df['is_summer'] = future_df['month'].isin([3, 4, 5]).astype(int)
        future_df['is_monsoon'] = future_df['month'].isin([6, 7, 8, 9]).astype(int)
        future_df['is_post_monsoon'] = future_df['month'].isin([10, 11]).astype(int)
        
        # Long-term trends
        historical_start_year = historical_df['year'].min()
        future_df['years_since_start'] = future_df['year'] - historical_start_year
        future_df['trend'] = np.arange(len(historical_df), len(historical_df) + len(future_df))
        
        # Climate change projections (based on IPCC scenarios)
        # Temperature increase: ~0.2°C per decade
        # Rainfall variability: ±5% per decade
        base_temp_increase = (future_df['year'] - 2024) * 0.02  # 0.2°C per decade
        base_rainfall_change = (future_df['year'] - 2024) * 0.005  # 0.5% per decade
        
        # Estimate future climate variables based on historical patterns
        historical_monthly_stats = historical_df.groupby('month').agg({
            'temperature': ['mean', 'std'],
            'rainfall': ['mean', 'std'],
            'humidity': ['mean', 'std'],
            'wind_speed': ['mean', 'std'],
            'pressure': ['mean', 'std'],
            'aqi': ['mean', 'std'],
            'co2': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        historical_monthly_stats.columns = ['_'.join(col).strip() for col in historical_monthly_stats.columns]
        
        # Add estimated climate variables
        for month in range(1, 13):
            month_mask = future_df['month'] == month
            
            # Temperature with climate change trend
            temp_mean = historical_monthly_stats.loc[month, 'temperature_mean']
            temp_std = historical_monthly_stats.loc[month, 'temperature_std']
            future_df.loc[month_mask, 'temperature'] = (
                temp_mean + base_temp_increase[month_mask] + 
                np.random.normal(0, temp_std, month_mask.sum())
            )
            
            # Rainfall with variability
            rain_mean = historical_monthly_stats.loc[month, 'rainfall_mean']
            rain_std = historical_monthly_stats.loc[month, 'rainfall_std']
            future_df.loc[month_mask, 'rainfall'] = np.maximum(0,
                rain_mean * (1 + base_rainfall_change[month_mask]) + 
                np.random.normal(0, rain_std, month_mask.sum())
            )
            
            # Other variables (with some trends)
            for var in ['humidity', 'wind_speed', 'pressure']:
                var_mean = historical_monthly_stats.loc[month, f'{var}_mean']
                var_std = historical_monthly_stats.loc[month, f'{var}_std']
                future_df.loc[month_mask, var] = (
                    var_mean + np.random.normal(0, var_std, month_mask.sum())
                )
            
            # AQI (assuming gradual improvement due to policies)
            aqi_mean = historical_monthly_stats.loc[month, 'aqi_mean']
            aqi_std = historical_monthly_stats.loc[month, 'aqi_std']
            aqi_improvement = (future_df['year'] - 2024) * -0.5  # 0.5 point improvement per year
            future_df.loc[month_mask, 'aqi'] = np.maximum(20,
                aqi_mean + aqi_improvement[month_mask] + 
                np.random.normal(0, aqi_std, month_mask.sum())
            )
            
            # CO2 (continuing increase)
            co2_mean = historical_monthly_stats.loc[month, 'co2_mean']
            co2_increase = (future_df['year'] - 2024) * 2.5  # 2.5 ppm per year
            future_df.loc[month_mask, 'co2'] = (
                co2_mean + co2_increase[month_mask] + 
                np.random.normal(0, 2, month_mask.sum())
            )
        
        # Add solar radiation (seasonal pattern)
        future_df['solar_radiation'] = (
            600 + 200 * np.sin(2 * np.pi * future_df['day_of_year'] / 365) +
            np.random.normal(0, 50, len(future_df))
        )
        
        # Add season dummies if they exist in historical data
        if 'season' in historical_df.columns:
            season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Summer', 4: 'Summer', 5: 'Summer',
                         6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
                         10: 'Post-Monsoon', 11: 'Post-Monsoon'}
            future_df['season'] = future_df['month'].map(season_map)
            
            # Add season dummies
            season_dummies = pd.get_dummies(future_df['season'], prefix='season')
            future_df = pd.concat([future_df, season_dummies], axis=1)
        
        print(f"✅ Future features prepared: {len(future_df)} days, {len(future_df.columns)} features")
        return future_df
    
    def predict_with_linear_model(self, model_info: Dict[str, Any], 
                                 future_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using linear regression model"""
        model = model_info['model']
        target = model_info['target']
        
        # Get feature columns used during training
        if hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_
        else:
            # Fallback: use all numeric columns except target and date
            feature_cols = [col for col in future_df.columns 
                           if col not in ['date', target, 'season'] and 
                           future_df[col].dtype in ['int64', 'float64']]
        
        # Ensure all required features are present
        missing_features = set(feature_cols) - set(future_df.columns)
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                future_df[feature] = 0
        
        X_future = future_df[feature_cols]
        predictions = model.predict(X_future)
        
        return predictions
    
    def predict_with_random_forest(self, model_info: Dict[str, Any], 
                                  future_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using random forest model"""
        return self.predict_with_linear_model(model_info, future_df)
    
    def predict_with_prophet(self, model_info: Dict[str, Any], 
                            future_years: List[int]) -> pd.DataFrame:
        """Make predictions using Prophet model"""
        model = model_info['model']
        
        # Create future dataframe for Prophet
        start_date = datetime(min(future_years), 1, 1)
        end_date = datetime(max(future_years), 12, 31)
        
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        future_prophet = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = model.predict(future_prophet)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def predict_with_lstm(self, model_info: Dict[str, Any], 
                         historical_df: pd.DataFrame, 
                         future_years: List[int]) -> np.ndarray:
        """Make predictions using LSTM model"""
        model = model_info['model']
        scaler = model_info['scaler']
        sequence_length = model_info['sequence_length']
        target = model_info['target']
        
        # Get historical target values
        historical_values = historical_df[target].values.reshape(-1, 1)
        historical_scaled = scaler.transform(historical_values)
        
        # Calculate number of future days
        start_date = datetime(min(future_years), 1, 1)
        end_date = datetime(max(future_years), 12, 31)
        num_future_days = (end_date - start_date).days + 1
        
        # Make predictions iteratively
        predictions = []
        current_sequence = historical_scaled[-sequence_length:].flatten()
        
        for _ in range(num_future_days):
            # Reshape for LSTM input
            X_input = current_sequence.reshape(1, sequence_length, 1)
            
            # Predict next value
            next_pred = model.predict(X_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform predictions
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_actual = scaler.inverse_transform(predictions_array).flatten()
        
        return predictions_actual


def predict_future(model_info: Dict[str, Any], future_years: List[int], 
                  historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main function to generate climate forecasts for specified years
    
    Args:
        model_info: Dictionary containing trained model information
        future_years: List of years to predict (e.g., [2026, 2027, 2028])
        historical_df: Historical data for context (required for some models)
    
    Returns:
        DataFrame with predictions for the specified years
    """
    print(f"🔮 GENERATING PREDICTIONS FOR YEARS: {future_years}")
    print("=" * 60)
    
    predictor = ClimatePredictor()
    model_type = model_info['model_type']
    target = model_info['target']
    
    # Create date range for predictions
    start_date = datetime(min(future_years), 1, 1)
    end_date = datetime(max(future_years), 12, 31)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"📅 Prediction period: {start_date.date()} to {end_date.date()}")
    print(f"🎯 Target variable: {target}")
    print(f"🤖 Model type: {model_type}")
    
    if model_type == 'prophet':
        # Prophet handles its own feature preparation
        forecast_df = predictor.predict_with_prophet(model_info, future_years)
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'date': forecast_df['ds'],
            'year': forecast_df['ds'].dt.year,
            'month': forecast_df['ds'].dt.month,
            'day': forecast_df['ds'].dt.day,
            f'{target}_predicted': forecast_df['yhat'],
            f'{target}_lower': forecast_df['yhat_lower'],
            f'{target}_upper': forecast_df['yhat_upper'],
            'model_type': model_type,
            'confidence_interval': True
        })
    
    elif model_type == 'lstm':
        if historical_df is None:
            raise ValueError("Historical data required for LSTM predictions")
        
        # LSTM predictions
        predictions = predictor.predict_with_lstm(model_info, historical_df, future_years)
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'date': future_dates,
            'year': [d.year for d in future_dates],
            'month': [d.month for d in future_dates],
            'day': [d.day for d in future_dates],
            f'{target}_predicted': predictions,
            'model_type': model_type,
            'confidence_interval': False
        })
    
    else:
        # Linear regression and Random Forest
        if historical_df is None:
            raise ValueError("Historical data required for feature preparation")
        
        # Prepare future features
        future_df = predictor.prepare_future_features(
            historical_df['date'].max(), future_years, historical_df
        )
        
        # Make predictions
        if model_type == 'linear_regression':
            predictions = predictor.predict_with_linear_model(model_info, future_df)
        elif model_type == 'random_forest':
            predictions = predictor.predict_with_random_forest(model_info, future_df)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'date': future_df['date'],
            'year': future_df['year'],
            'month': future_df['month'],
            'day': future_df['day'],
            f'{target}_predicted': predictions,
            'model_type': model_type,
            'confidence_interval': False
        })
    
    # Add additional information
    result_df['season'] = result_df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Summer', 4: 'Summer', 5: 'Summer',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Post-Monsoon', 11: 'Post-Monsoon'
    })
    
    # Add climate change context
    result_df['climate_change_factor'] = (result_df['year'] - 2024) * 0.02  # Temperature increase
    result_df['prediction_uncertainty'] = 'medium'  # Could be calculated based on model performance
    
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"   📅 Total predictions: {len(result_df):,}")
    print(f"   🎯 Target: {target}")
    print(f"   📈 Prediction range: {result_df[f'{target}_predicted'].min():.2f} to {result_df[f'{target}_predicted'].max():.2f}")
    print(f"   🌡️ Average annual change: {result_df.groupby('year')[f'{target}_predicted'].mean().diff().mean():.3f}")
    
    return result_df


def predict_multiple_targets(models_dict: Dict[str, Dict[str, Any]], 
                           future_years: List[int],
                           historical_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate predictions for multiple targets using multiple models
    
    Args:
        models_dict: Dictionary of models for different targets
        future_years: Years to predict
        historical_df: Historical data
    
    Returns:
        Dictionary of prediction DataFrames for each target
    """
    print(f"🚀 GENERATING MULTI-TARGET PREDICTIONS")
    print("=" * 60)
    
    all_predictions = {}
    
    for target, target_models in models_dict.items():
        print(f"\n🎯 Predicting {target}...")
        target_predictions = {}
        
        for model_type, model_info in target_models.items():
            try:
                pred_df = predict_future(model_info, future_years, historical_df)
                target_predictions[model_type] = pred_df
                print(f"✅ {model_type} predictions completed for {target}")
            except Exception as e:
                print(f"❌ Error predicting {target} with {model_type}: {e}")
                continue
        
        all_predictions[target] = target_predictions
    
    print(f"\n✅ ALL PREDICTIONS COMPLETED!")
    return all_predictions


# Example usage and testing
if __name__ == "__main__":
    # This would typically be run with actual trained models
    print("🧪 Testing prediction module...")
    
    # Create sample historical data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    historical_data = {
        'date': dates,
        'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'day': [d.day for d in dates]
    }
    
    historical_df = pd.DataFrame(historical_data)
    
    # Create mock model info
    mock_model_info = {
        'model_type': 'prophet',
        'target': 'temperature'
    }
    
    print("✅ Prediction module test setup completed!")