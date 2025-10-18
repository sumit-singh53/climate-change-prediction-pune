"""
Enhanced Machine Learning Models for Climate and AQI Prediction
Implements state-of-the-art ensemble methods with advanced feature engineering
"""

import json
import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    VotingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV,
    TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Input, Attention, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import optuna

warnings.filterwarnings("ignore")

from config import DATABASE_CONFIG, MODEL_CONFIG, PUNE_LOCATIONS


class EnhancedMLModels:
    """Enhanced ML models with advanced feature engineering and hyperparameter optimization"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG["sqlite_path"]
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_performance = {}
        self.feature_importance = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def load_and_prepare_data(self, days_back: int = 730) -> pd.DataFrame:
        """Load and prepare data with advanced feature engineering"""
        conn = sqlite3.connect(self.db_path)
        
        # Load weather data
        weather_query = f"""
            SELECT * FROM weather_historical 
            WHERE timestamp > datetime('now', '-{days_back} days')
            ORDER BY timestamp
        """
        weather_df = pd.read_sql_query(weather_query, conn)
        
        # Load air quality data
        aqi_query = f"""
            SELECT * FROM air_quality_historical 
            WHERE timestamp > datetime('now', '-{days_back} days')
            ORDER BY timestamp
        """
        aqi_df = pd.read_sql_query(aqi_query, conn)
        
        conn.close()
        
        # Merge and engineer features
        combined_df = self.merge_and_engineer_features(weather_df, aqi_df)
        return combined_df
    
    def merge_and_engineer_features(self, weather_df: pd.DataFrame, aqi_df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for better model performance"""
        
        # Convert timestamps
        for df in [weather_df, aqi_df]:
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Merge on timestamp and location
        if not weather_df.empty and not aqi_df.empty:
            combined_df = pd.merge(
                weather_df, aqi_df, 
                on=['timestamp', 'location_id'], 
                how='outer', 
                suffixes=('_weather', '_aqi')
            )
        elif not weather_df.empty:
            combined_df = weather_df.copy()
        elif not aqi_df.empty:
            combined_df = aqi_df.copy()
        else:
            return pd.DataFrame()
        
        # Advanced Feature Engineering
        combined_df = self.create_advanced_features(combined_df)
        
        return combined_df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better prediction accuracy"""
        
        if df.empty:
            return df
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['location_id', 'timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                       3: 1, 4: 1, 5: 1,   # Spring
                                       6: 2, 7: 2, 8: 2,   # Summer
                                       9: 3, 10: 3, 11: 3}) # Autumn
        
        # Cyclical encoding for time features
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
            
            # Comfort index
            df['comfort_index'] = (df['temperature'] - df['dew_point']) / df['humidity'] * 100
        
        # Wind chill and apparent temperature
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['wind_chill'] = self.calculate_wind_chill(df['temperature'], df['wind_speed'])
            df['apparent_temp'] = df['temperature'] + 0.33 * (df.get('pressure', 1013) / 10) - 0.7 * df['wind_speed']
        
        # Air quality interactions
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)  # Avoid division by zero
            df['pm_total'] = df['pm25'] + df['pm10']
        
        # Lag features (previous values)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in ['temperature', 'humidity', 'pm25', 'pm10', 'aqi']:
            if col in df.columns:
                for lag in [1, 3, 6, 12, 24]:  # 1h, 3h, 6h, 12h, 24h lags
                    df[f'{col}_lag_{lag}'] = df.groupby('location_id')[col].shift(lag)
        
        # Rolling statistics
        for col in ['temperature', 'humidity', 'pm25', 'pm10', 'aqi']:
            if col in df.columns:
                for window in [6, 12, 24, 48]:  # 6h, 12h, 24h, 48h windows
                    df[f'{col}_rolling_mean_{window}'] = df.groupby('location_id')[col].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
                    df[f'{col}_rolling_std_{window}'] = df.groupby('location_id')[col].rolling(window, min_periods=1).std().reset_index(0, drop=True)
                    df[f'{col}_rolling_max_{window}'] = df.groupby('location_id')[col].rolling(window, min_periods=1).max().reset_index(0, drop=True)
                    df[f'{col}_rolling_min_{window}'] = df.groupby('location_id')[col].rolling(window, min_periods=1).min().reset_index(0, drop=True)
        
        # Rate of change features
        for col in ['temperature', 'humidity', 'pm25', 'pm10', 'aqi']:
            if col in df.columns:
                df[f'{col}_change_1h'] = df.groupby('location_id')[col].diff(1)
                df[f'{col}_change_3h'] = df.groupby('location_id')[col].diff(3)
                df[f'{col}_change_6h'] = df.groupby('location_id')[col].diff(6)
        
        # Location-based features
        location_encoder = {loc_id: idx for idx, loc_id in enumerate(PUNE_LOCATIONS.keys())}
        df['location_encoded'] = df['location_id'].map(location_encoder)
        
        # Add location metadata
        for loc_id, loc_config in PUNE_LOCATIONS.items():
            mask = df['location_id'] == loc_id
            df.loc[mask, 'latitude'] = loc_config.lat
            df.loc[mask, 'longitude'] = loc_config.lon
            # Add elevation as 0 for now (can be updated with real data)
            df.loc[mask, 'elevation'] = 0
        
        # Distance-based features (from city center - Pune Central)
        pune_central = PUNE_LOCATIONS['pune_central']
        df['distance_from_center'] = np.sqrt(
            (df['latitude'] - pune_central.lat) ** 2 + 
            (df['longitude'] - pune_central.lon) ** 2
        )
        
        return df
    
    def calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature and humidity"""
        # Simplified heat index calculation
        hi = 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (humidity * 0.094))
        return hi
    
    def calculate_dew_point(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate dew point from temperature and humidity"""
        # Magnus formula approximation
        a, b = 17.27, 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        return dew_point
    
    def calculate_wind_chill(self, temp: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill from temperature and wind speed"""
        # Wind chill formula (for temperatures in Celsius)
        wind_chill = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
        return wind_chill
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict:
        """Use Optuna for hyperparameter optimization"""
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = xgb.XGBRegressor(**params, random_state=42)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
                model = RandomForestRegressor(**params, random_state=42)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def train_enhanced_models(self, target_variable: str) -> Dict[str, Any]:
        """Train enhanced models with hyperparameter optimization"""
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        if df.empty or target_variable not in df.columns:
            raise ValueError(f"No data available for target variable: {target_variable}")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'location_id', target_variable
        ] and not col.endswith('_target')]
        
        X = df[feature_cols].copy()
        y = df[target_variable].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Split data (time-aware split)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        models = {}
        performances = {}
        
        # 1. Optimized XGBoost
        print(f"Optimizing XGBoost for {target_variable}...")
        xgb_params = self.optimize_hyperparameters(
            pd.DataFrame(X_train), y_train, 'xgboost'
        )
        xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42)
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        # 2. Optimized LightGBM
        print(f"Optimizing LightGBM for {target_variable}...")
        lgb_params = self.optimize_hyperparameters(
            pd.DataFrame(X_train), y_train, 'lightgbm'
        )
        lgb_model = lgb.LGBMRegressor(**lgb_params, random_state=42, verbose=-1)
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        # 3. Optimized Random Forest
        print(f"Optimizing Random Forest for {target_variable}...")
        rf_params = self.optimize_hyperparameters(
            pd.DataFrame(X_train), y_train, 'random_forest'
        )
        rf_model = RandomForestRegressor(**rf_params, random_state=42)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # 4. Enhanced LSTM
        lstm_model = self.build_enhanced_lstm(X_train.shape[1])
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=10, factor=0.5)
        
        lstm_model.fit(
            X_train_lstm, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        models['lstm'] = lstm_model
        
        # 5. Ensemble Model
        ensemble_model = VotingRegressor([
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model)
        ])
        ensemble_model.fit(X_train, y_train)
        models['ensemble'] = ensemble_model
        
        # Evaluate all models
        for name, model in models.items():
            if name == 'lstm':
                y_pred = model.predict(X_test_lstm).flatten()
            else:
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            performances[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'accuracy': max(0, r2 * 100)  # Convert R² to percentage
            }
            
            print(f"{name.upper()} - MAE: {mae:.3f}, RMSE: {np.sqrt(mse):.3f}, R²: {r2:.3f}, Accuracy: {max(0, r2 * 100):.1f}%")
        
        # Store models and metadata
        self.models[target_variable] = models
        self.scalers[target_variable] = scaler
        self.feature_selectors[target_variable] = selector
        self.model_performance[target_variable] = performances
        
        # Feature importance for tree-based models
        feature_importance = {}
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(selected_features, model.feature_importances_))
                feature_importance[name] = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        self.feature_importance[target_variable] = feature_importance
        
        return {
            'models': models,
            'performances': performances,
            'feature_importance': feature_importance,
            'selected_features': selected_features
        }
    
    def build_enhanced_lstm(self, input_dim: int) -> tf.keras.Model:
        """Build enhanced LSTM with attention mechanism"""
        
        inputs = Input(shape=(1, input_dim))
        
        # LSTM layers with dropout
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=16)(lstm2, lstm2)
        attention = LayerNormalization()(attention)
        
        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(pooled)
        dense1 = Dropout(0.3)(dense1)
        dense1 = BatchNormalization()(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def predict_enhanced(self, target_variable: str, input_data: pd.DataFrame, 
                        horizon_days: int = 7) -> Dict[str, Any]:
        """Make enhanced predictions with confidence intervals"""
        
        if target_variable not in self.models:
            raise ValueError(f"No trained model found for {target_variable}")
        
        models = self.models[target_variable]
        scaler = self.scalers[target_variable]
        selector = self.feature_selectors[target_variable]
        
        # Prepare input data
        input_processed = self.create_advanced_features(input_data)
        
        # Select and scale features
        feature_cols = [col for col in input_processed.columns if col not in [
            'timestamp', 'location_id', target_variable
        ]]
        
        X = input_processed[feature_cols].fillna(input_processed[feature_cols].median())
        X_selected = selector.transform(X)
        X_scaled = scaler.transform(X_selected)
        
        # Get predictions from all models
        predictions = {}
        
        for name, model in models.items():
            if name == 'lstm':
                X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                pred = model.predict(X_lstm, verbose=0).flatten()
            else:
                pred = model.predict(X_scaled)
            
            predictions[name] = pred
        
        # Ensemble prediction (weighted average based on performance)
        performances = self.model_performance[target_variable]
        weights = {name: perf['r2'] for name, perf in performances.items() if perf['r2'] > 0}
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            ensemble_pred = np.zeros_like(predictions['xgboost'])
            for name, pred in predictions.items():
                if name in weights:
                    ensemble_pred += pred * (weights[name] / total_weight)
        else:
            ensemble_pred = predictions['ensemble']
        
        # Calculate confidence intervals (using ensemble std)
        pred_std = np.std([pred for pred in predictions.values()], axis=0)
        confidence_lower = ensemble_pred - 1.96 * pred_std
        confidence_upper = ensemble_pred + 1.96 * pred_std
        
        return {
            'predictions': ensemble_pred,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'individual_predictions': predictions,
            'model_performances': performances,
            'prediction_std': pred_std
        }
    
    def save_models(self, target_variable: str, filepath: str):
        """Save trained models"""
        model_data = {
            'models': self.models.get(target_variable, {}),
            'scaler': self.scalers.get(target_variable),
            'selector': self.feature_selectors.get(target_variable),
            'performance': self.model_performance.get(target_variable, {}),
            'feature_importance': self.feature_importance.get(target_variable, {})
        }
        
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, target_variable: str, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)
        
        self.models[target_variable] = model_data['models']
        self.scalers[target_variable] = model_data['scaler']
        self.feature_selectors[target_variable] = model_data['selector']
        self.model_performance[target_variable] = model_data['performance']
        self.feature_importance[target_variable] = model_data['feature_importance']
        
        print(f"Models loaded from {filepath}")