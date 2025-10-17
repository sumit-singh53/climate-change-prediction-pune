"""
Advanced Machine Learning Models for Climate and AQI Prediction
Implements ensemble methods, deep learning, and time series models for high accuracy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import joblib
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from config import MODEL_CONFIG, DATABASE_CONFIG, PUNE_LOCATIONS
import logging

class AdvancedMLModels:
    """Advanced ML models for climate and AQI prediction with ensemble methods"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG['sqlite_path']
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def load_data(self, days_back: int = 365) -> pd.DataFrame:
        """Load and prepare data for training"""
        conn = sqlite3.connect(self.db_path)
        
        # Load weather data
        weather_query = f'''
            SELECT * FROM weather_historical 
            WHERE timestamp > datetime('now', '-{days_back} days')
            ORDER BY timestamp
        '''
        weather_df = pd.read_sql_query(weather_query, conn)
        
        # Load air quality data
        aqi_query = f'''
            SELECT * FROM air_quality_historical 
            WHERE timestamp > datetime('now', '-{days_back} days')
            ORDER BY timestamp
        '''
        aqi_df = pd.read_sql_query(aqi_query, conn)
        
        # Load IoT sensor data
        iot_query = f'''
            SELECT timestamp, location_id, sensor_type, value, quality_score
            FROM iot_sensor_data 
            WHERE timestamp > datetime('now', '-{days_back} days')
            ORDER BY timestamp
        '''
        iot_df = pd.read_sql_query(iot_query, conn)
        
        conn.close()
        
        # Merge datasets
        combined_df = self.merge_datasets(weather_df, aqi_df, iot_df)
        return combined_df
    
    def merge_datasets(self, weather_df: pd.DataFrame, aqi_df: pd.DataFrame, iot_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather, AQI, and IoT data"""
        # Convert timestamps
        for df in [weather_df, aqi_df, iot_df]:
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Merge weather and AQI data
        if not weather_df.empty and not aqi_df.empty:
            combined_df = pd.merge(weather_df, aqi_df, on=['timestamp', 'location_id'], how='outer')
        elif not weather_df.empty:
            combined_df = weather_df.copy()
        elif not aqi_df.empty:
            combined_df = aqi_df.copy()
        else:
            return pd.DataFrame()
        
        # Pivot IoT data and merge
        if not iot_df.empty:
            iot_pivot = iot_df.pivot_table(
                index=['timestamp', 'location_id'], 
                columns='sensor_type', 
                values='value', 
                aggfunc='mean'
            ).reset_index()
            
            # Rename IoT columns to avoid conflicts
            iot_columns = {col: f'iot_{col}' for col in iot_pivot.columns if col not in ['timestamp', 'location_id']}
            iot_pivot.rename(columns=iot_columns, inplace=True)
            
            combined_df = pd.merge(combined_df, iot_pivot, on=['timestamp', 'location_id'], how='left')
        
        return combined_df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better prediction accuracy"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                       3: 1, 4: 1, 5: 1,   # Spring
                                       6: 2, 7: 2, 8: 2,   # Summer
                                       9: 3, 10: 3, 11: 3}) # Autumn
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weather-based features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            # Heat index calculation
            df['heat_index'] = self.calculate_heat_index(df['temperature'], df['humidity'])
            
            # Dew point calculation
            df['dew_point'] = self.calculate_dew_point(df['temperature'], df['humidity'])
        
        if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
            # Wind components
            df['wind_u'] = df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
            df['wind_v'] = df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
        
        # Air quality features
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)  # Avoid division by zero
        
        # Location-based features
        location_encoder = LabelEncoder()
        if 'location_id' in df.columns:
            df['location_encoded'] = location_encoder.fit_transform(df['location_id'])
            
            # Add location metadata
            for location_id, location_config in PUNE_LOCATIONS.items():
                mask = df['location_id'] == location_id
                df.loc[mask, 'latitude'] = location_config.lat
                df.loc[mask, 'longitude'] = location_config.lon
                df.loc[mask, 'zone_encoded'] = hash(location_config.zone) % 100
        
        # Lag features for time series
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['hour', 'day_of_week', 'month', 'season', 'location_encoded']:
                df[f'{col}_lag1'] = df.groupby('location_id')[col].shift(1)
                df[f'{col}_lag24'] = df.groupby('location_id')[col].shift(24)  # 24 hours ago
                df[f'{col}_rolling_mean_24h'] = df.groupby('location_id')[col].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{col}_rolling_std_24h'] = df.groupby('location_id')[col].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)
        
        return df
    
    def calculate_heat_index(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature and humidity"""
        temp_f = temp_c * 9/5 + 32  # Convert to Fahrenheit
        
        # Simplified heat index formula
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
        
        # More complex formula for higher temperatures
        mask = hi > 80
        if mask.any():
            hi_complex = (-42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
                         - 0.22475541 * temp_f * humidity - 0.00683783 * temp_f**2
                         - 0.05481717 * humidity**2 + 0.00122874 * temp_f**2 * humidity
                         + 0.00085282 * temp_f * humidity**2 - 0.00000199 * temp_f**2 * humidity**2)
            hi[mask] = hi_complex[mask]
        
        return (hi - 32) * 5/9  # Convert back to Celsius
    
    def calculate_dew_point(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate dew point from temperature and humidity"""
        a = 17.27
        b = 237.7
        
        alpha = ((a * temp_c) / (b + temp_c)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return dew_point
    
    def prepare_training_data(self, df: pd.DataFrame, target_variable: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training"""
        # Remove rows with missing target variable
        df_clean = df.dropna(subset=[target_variable]).copy()
        
        # Select features
        exclude_columns = ['timestamp', 'location_id', 'data_source', 'quality_score', 'dominant_pollutant']
        feature_columns = [col for col in df_clean.columns if col not in exclude_columns + [target_variable]]
        
        # Handle missing values
        df_clean[feature_columns] = df_clean[feature_columns].fillna(df_clean[feature_columns].median())
        
        X = df_clean[feature_columns].values
        y = df_clean[target_variable].values
        
        return X, y, feature_columns
    
    def train_ensemble_model(self, X: np.ndarray, y: np.ndarray, target_variable: str) -> Dict[str, Any]:
        """Train ensemble model with multiple algorithms"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train models and collect predictions
        model_predictions = {}
        model_scores = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name} for {target_variable}...")
            
            if name in ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_predictions[name] = y_pred
            model_scores[name] = {'mae': mae, 'mse': mse, 'r2': r2}
            
            self.logger.info(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # Create ensemble prediction (weighted average based on R2 scores)
        weights = {}
        total_r2 = sum(scores['r2'] for scores in model_scores.values() if scores['r2'] > 0)
        
        if total_r2 > 0:
            for name, scores in model_scores.items():
                weights[name] = max(scores['r2'], 0) / total_r2
        else:
            # Equal weights if all R2 scores are negative
            weights = {name: 1/len(models) for name in models.keys()}
        
        # Calculate ensemble prediction
        ensemble_pred = np.zeros_like(y_test)
        for name, pred in model_predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Ensemble metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        self.logger.info(f"Ensemble - MAE: {ensemble_mae:.4f}, MSE: {ensemble_mse:.4f}, R2: {ensemble_r2:.4f}")
        
        return {
            'models': models,
            'scaler': scaler,
            'weights': weights,
            'model_scores': model_scores,
            'ensemble_score': {'mae': ensemble_mae, 'mse': ensemble_mse, 'r2': ensemble_r2},
            'test_predictions': ensemble_pred,
            'test_actual': y_test
        }
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, target_variable: str, sequence_length: int = 24) -> Dict[str, Any]:
        """Train LSTM model for time series prediction"""
        # Prepare sequences
        X_seq, y_seq = self.create_sequences(X, y, sequence_length)
        
        if len(X_seq) == 0:
            self.logger.warning(f"Not enough data for LSTM training for {target_variable}")
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # Train model
        self.logger.info(f"Training LSTM for {target_variable}...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_scaled, y_test_scaled),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"LSTM - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
        
        return {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'sequence_length': sequence_length,
            'score': {'mae': mae, 'mse': mse, 'r2': r2},
            'history': history.history,
            'test_predictions': y_pred,
            'test_actual': y_test
        }
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_all_models(self, days_back: int = 365):
        """Train models for all target variables"""
        self.logger.info("Loading and preparing data...")
        df = self.load_data(days_back)
        
        if df.empty:
            self.logger.error("No data available for training")
            return
        
        # Feature engineering
        df_features = self.feature_engineering(df)
        
        # Train models for each target variable
        for target_variable in MODEL_CONFIG['target_variables']:
            if target_variable not in df_features.columns:
                self.logger.warning(f"Target variable {target_variable} not found in data")
                continue
            
            self.logger.info(f"Training models for {target_variable}...")
            
            # Prepare data
            X, y, feature_columns = self.prepare_training_data(df_features, target_variable)
            
            if len(X) < 100:  # Minimum data requirement
                self.logger.warning(f"Insufficient data for {target_variable}: {len(X)} samples")
                continue
            
            # Train ensemble model
            ensemble_results = self.train_ensemble_model(X, y, target_variable)
            
            # Train LSTM model
            lstm_results = self.train_lstm_model(X, y, target_variable)
            
            # Store results
            self.models[target_variable] = {
                'ensemble': ensemble_results,
                'lstm': lstm_results,
                'feature_columns': feature_columns,
                'training_date': datetime.now().isoformat()
            }
            
            # Store feature importance
            if 'random_forest' in ensemble_results.get('models', {}):
                rf_model = ensemble_results['models']['random_forest']
                self.feature_importance[target_variable] = dict(zip(
                    feature_columns, 
                    rf_model.feature_importances_
                ))
        
        # Save models
        self.save_models()
        self.logger.info("Model training completed!")
    
    def predict(self, location_id: str, target_variable: str, horizon_days: int = 1) -> Dict[str, Any]:
        """Make predictions for a specific location and target variable"""
        if target_variable not in self.models:
            raise ValueError(f"No trained model found for {target_variable}")
        
        # Get recent data for prediction
        recent_data = self.get_recent_data_for_prediction(location_id, days_back=30)
        
        if recent_data.empty:
            raise ValueError(f"No recent data available for location {location_id}")
        
        # Feature engineering
        recent_features = self.feature_engineering(recent_data)
        
        # Prepare features
        model_info = self.models[target_variable]
        feature_columns = model_info['feature_columns']
        
        # Get the most recent complete record
        latest_data = recent_features.dropna(subset=feature_columns).tail(1)
        
        if latest_data.empty:
            raise ValueError("No complete recent data available for prediction")
        
        X_latest = latest_data[feature_columns].values
        
        # Ensemble prediction
        ensemble_results = model_info.get('ensemble', {})
        ensemble_pred = None
        
        if ensemble_results:
            scaler = ensemble_results['scaler']
            models = ensemble_results['models']
            weights = ensemble_results['weights']
            
            X_scaled = scaler.transform(X_latest)
            
            ensemble_pred = 0
            for name, model in models.items():
                if name in ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']:
                    pred = model.predict(X_latest)[0]
                else:
                    pred = model.predict(X_scaled)[0]
                
                ensemble_pred += weights[name] * pred
        
        # LSTM prediction (if available)
        lstm_pred = None
        lstm_results = model_info.get('lstm', {})
        
        if lstm_results and 'model' in lstm_results:
            sequence_length = lstm_results['sequence_length']
            
            if len(recent_features) >= sequence_length:
                # Prepare sequence
                sequence_data = recent_features[feature_columns].tail(sequence_length).values
                scaler_X = lstm_results['scaler_X']
                scaler_y = lstm_results['scaler_y']
                
                X_seq = scaler_X.transform(sequence_data).reshape(1, sequence_length, -1)
                lstm_pred_scaled = lstm_results['model'].predict(X_seq)
                lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)[0][0]
        
        # Combine predictions
        final_prediction = None
        confidence = 0.0
        
        if ensemble_pred is not None and lstm_pred is not None:
            # Weight ensemble and LSTM predictions
            final_prediction = 0.7 * ensemble_pred + 0.3 * lstm_pred
            confidence = 0.8
        elif ensemble_pred is not None:
            final_prediction = ensemble_pred
            confidence = 0.7
        elif lstm_pred is not None:
            final_prediction = lstm_pred
            confidence = 0.6
        
        return {
            'location_id': location_id,
            'target_variable': target_variable,
            'prediction': final_prediction,
            'ensemble_prediction': ensemble_pred,
            'lstm_prediction': lstm_pred,
            'confidence': confidence,
            'prediction_time': datetime.now().isoformat(),
            'horizon_days': horizon_days
        }
    
    def get_recent_data_for_prediction(self, location_id: str, days_back: int = 30) -> pd.DataFrame:
        """Get recent data for making predictions"""
        conn = sqlite3.connect(self.db_path)
        
        # Combined query for all data types
        query = f'''
            SELECT 
                w.timestamp, w.location_id,
                w.temperature, w.humidity, w.pressure, w.wind_speed, w.wind_direction,
                w.precipitation, w.solar_radiation, w.uv_index, w.visibility, w.cloud_cover,
                a.pm25, a.pm10, a.no2, a.so2, a.co, a.o3, a.aqi
            FROM weather_historical w
            LEFT JOIN air_quality_historical a ON w.timestamp = a.timestamp AND w.location_id = a.location_id
            WHERE w.location_id = ? AND w.timestamp > datetime('now', '-{days_back} days')
            ORDER BY w.timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(location_id,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def save_models(self):
        """Save trained models to disk"""
        import os
        os.makedirs('outputs/models', exist_ok=True)
        
        for target_variable, model_info in self.models.items():
            # Save ensemble models
            if 'ensemble' in model_info:
                ensemble_path = f'outputs/models/{target_variable}_ensemble.joblib'
                joblib.dump(model_info['ensemble'], ensemble_path)
            
            # Save LSTM models
            if 'lstm' in model_info and 'model' in model_info['lstm']:
                lstm_path = f'outputs/models/{target_variable}_lstm.h5'
                model_info['lstm']['model'].save(lstm_path)
                
                # Save LSTM scalers and metadata
                lstm_meta_path = f'outputs/models/{target_variable}_lstm_meta.joblib'
                lstm_meta = {k: v for k, v in model_info['lstm'].items() if k != 'model'}
                joblib.dump(lstm_meta, lstm_meta_path)
        
        # Save feature importance
        with open('outputs/models/feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        self.logger.info("Models saved successfully!")
    
    def load_models(self):
        """Load trained models from disk"""
        import os
        
        for target_variable in MODEL_CONFIG['target_variables']:
            model_info = {'feature_columns': []}
            
            # Load ensemble model
            ensemble_path = f'outputs/models/{target_variable}_ensemble.joblib'
            if os.path.exists(ensemble_path):
                model_info['ensemble'] = joblib.load(ensemble_path)
            
            # Load LSTM model
            lstm_path = f'outputs/models/{target_variable}_lstm.h5'
            lstm_meta_path = f'outputs/models/{target_variable}_lstm_meta.joblib'
            
            if os.path.exists(lstm_path) and os.path.exists(lstm_meta_path):
                model_info['lstm'] = joblib.load(lstm_meta_path)
                model_info['lstm']['model'] = tf.keras.models.load_model(lstm_path)
            
            if model_info['ensemble'] or model_info.get('lstm'):
                self.models[target_variable] = model_info
        
        # Load feature importance
        importance_path = 'outputs/models/feature_importance.json'
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                self.feature_importance = json.load(f)
        
        self.logger.info(f"Loaded models for {len(self.models)} target variables")


if __name__ == "__main__":
    # Example usage
    ml_models = AdvancedMLModels()
    
    # Train models
    ml_models.train_all_models(days_back=180)
    
    # Make predictions
    try:
        prediction = ml_models.predict('pune_central', 'temperature', horizon_days=1)
        print(f"Temperature prediction: {prediction}")
        
        aqi_prediction = ml_models.predict('pune_central', 'pm25', horizon_days=1)
        print(f"PM2.5 prediction: {aqi_prediction}")
    except Exception as e:
        print(f"Prediction error: {e}")