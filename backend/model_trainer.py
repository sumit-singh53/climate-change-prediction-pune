"""
Model Training Module
Trains multiple ML models (Linear Regression, Random Forest, Prophet, LSTM)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Advanced model trainer for climate prediction"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
        print(f"🎯 Preparing features for {target} prediction...")
        
        # Select core climate features (avoid overfitting with too many features)
        core_features = ['year', 'month', 'day', 'day_of_year']
        
        # Add other climate variables (excluding target)
        climate_vars = ['temperature', 'rainfall', 'humidity', 'aqi', 'wind_speed', 'pressure', 'co2']
        for var in climate_vars:
            if var in df.columns and var != target:
                core_features.append(var)
        
        # Add some time-based features if they exist
        time_features = ['is_monsoon']
        for feat in time_features:
            if feat in df.columns:
                core_features.append(feat)
        
        # Add a few key rolling/lag features if they exist (not all to avoid overfitting)
        if target in ['temperature', 'rainfall']:
            key_rolling = [
                f'{target}_rolling_mean_7',
                f'{target}_rolling_mean_30',
                f'{target}_lag_1',
                f'{target}_lag_7'
            ]
            for feat in key_rolling:
                if feat in df.columns:
                    core_features.append(feat)
        
        # Handle categorical variables
        df_features = df.copy()
        if 'season' in df.columns:
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df_features = pd.concat([df_features, season_dummies], axis=1)
            core_features.extend(season_dummies.columns)
        
        # Select only available features
        available_features = [col for col in core_features if col in df_features.columns]
        
        X = df_features[available_features]
        y = df[target]
        
        # Remove any rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Features prepared: {len(available_features)} features, {len(y)} samples")
        print(f"   Selected features: {available_features[:10]}{'...' if len(available_features) > 10 else ''}")
        return X, y
    
    def train_linear_regression(self, X: pd.DataFrame, y: pd.Series, 
                               target: str) -> Dict[str, Any]:
        """Train Linear Regression model"""
        print(f"📈 Training Linear Regression for {target}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Feature importance (coefficients)
        feature_importance = dict(zip(X.columns, abs(model.coef_)))
        
        model_info = {
            'model': model,
            'model_type': 'linear_regression',
            'target': target,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"✅ Linear Regression trained - Test R²: {test_r2:.3f}, MAE: {test_mae:.3f}")
        return model_info
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                           target: str, optimize: bool = True) -> Dict[str, Any]:
        """Train Random Forest model with optional hyperparameter optimization"""
        print(f"🌲 Training Random Forest for {target}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        if optimize:
            print("🔧 Optimizing hyperparameters...")
            # Grid search for best parameters
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
        else:
            # Use default parameters
            model = RandomForestRegressor(
                n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        
        model_info = {
            'model': model,
            'model_type': 'random_forest',
            'target': target,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"✅ Random Forest trained - Test R²: {test_r2:.3f}, MAE: {test_mae:.3f}")
        return model_info
    
    def train_prophet(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Train Prophet model for time series forecasting"""
        print(f"📊 Training Prophet for {target}...")
        
        # Prepare data for Prophet
        prophet_df = df[['date', target]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        # Split data
        split_date = prophet_df['ds'].quantile(0.8)
        train_df = prophet_df[prophet_df['ds'] <= split_date]
        test_df = prophet_df[prophet_df['ds'] > split_date]
        
        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        
        # Add custom seasonalities for Indian climate
        model.add_seasonality(name='monsoon', period=365.25, fourier_order=10)
        
        model.fit(train_df)
        
        # Make predictions on test set
        future_test = model.make_future_dataframe(periods=len(test_df), freq='D')
        forecast = model.predict(future_test)
        
        # Get test predictions
        test_forecast = forecast.tail(len(test_df))
        y_pred_test = test_forecast['yhat'].values
        y_test = test_df['y'].values
        
        # Metrics
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        model_info = {
            'model': model,
            'model_type': 'prophet',
            'target': target,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'forecast': forecast,
            'test_df': test_df,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"✅ Prophet trained - Test R²: {test_r2:.3f}, MAE: {test_mae:.3f}")
        return model_info
    
    def train_lstm(self, df: pd.DataFrame, target: str, 
                   sequence_length: int = 30, epochs: int = 50) -> Dict[str, Any]:
        """Train LSTM model for time series prediction"""
        print(f"🧠 Training LSTM for {target}...")
        
        # Prepare data
        df_sorted = df.sort_values('date').copy()
        target_values = df_sorted[target].values.reshape(-1, 1)
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        target_scaled = scaler.fit_transform(target_values)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(target_scaled)):
            X.append(target_scaled[i-sequence_length:i, 0])
            y.append(target_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Predictions
        y_pred_test = model.predict(X_test, verbose=0)
        
        # Inverse transform
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = scaler.inverse_transform(y_pred_test).flatten()
        
        # Metrics
        test_r2 = r2_score(y_test_actual, y_pred_actual)
        test_mae = mean_absolute_error(y_test_actual, y_pred_actual)
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        
        model_info = {
            'model': model,
            'model_type': 'lstm',
            'target': target,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'history': history.history,
            'y_test': y_test_actual,
            'y_pred_test': y_pred_actual
        }
        
        print(f"✅ LSTM trained - Test R²: {test_r2:.3f}, MAE: {test_mae:.3f}")
        return model_info
    
    def save_model(self, model_info: Dict[str, Any], filepath: str):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if model_info['model_type'] == 'lstm':
            # Save Keras model
            model_info['model'].save(f"{filepath}_lstm.h5")
            # Save scaler and other info
            joblib.dump({
                'scaler': model_info['scaler'],
                'sequence_length': model_info['sequence_length'],
                'target': model_info['target'],
                'metrics': {
                    'test_r2': model_info['test_r2'],
                    'test_mae': model_info['test_mae'],
                    'test_rmse': model_info['test_rmse']
                }
            }, f"{filepath}_lstm_info.pkl")
        else:
            # Save sklearn/prophet models
            joblib.dump(model_info, filepath)
        
        print(f"✅ Model saved to {filepath}")


def train_model(df: pd.DataFrame, target: str, model_type: str, 
                optimize: bool = False, save_path: str = None) -> Dict[str, Any]:
    """
    Main function to train a specific model type
    
    Args:
        df: Input DataFrame with features and target
        target: Target variable name ('temperature' or 'rainfall')
        model_type: Type of model ('linear', 'random_forest', 'prophet', 'lstm')
        optimize: Whether to optimize hyperparameters
        save_path: Path to save the trained model
    
    Returns:
        Dictionary containing trained model and performance metrics
    """
    print(f"🤖 TRAINING {model_type.upper()} MODEL FOR {target.upper()}")
    print("=" * 60)
    
    trainer = ModelTrainer()
    
    if model_type == 'linear':
        X, y = trainer.prepare_features(df, target)
        model_info = trainer.train_linear_regression(X, y, target)
    
    elif model_type == 'random_forest':
        X, y = trainer.prepare_features(df, target)
        model_info = trainer.train_random_forest(X, y, target, optimize)
    
    elif model_type == 'prophet':
        model_info = trainer.train_prophet(df, target)
    
    elif model_type == 'lstm':
        model_info = trainer.train_lstm(df, target)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save model if path provided
    if save_path:
        trainer.save_model(model_info, save_path)
    
    print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
    print(f"   🎯 Target: {target}")
    print(f"   🤖 Model: {model_type}")
    print(f"   📈 Test R²: {model_info.get('test_r2', 'N/A'):.3f}")
    print(f"   📉 Test MAE: {model_info.get('test_mae', 'N/A'):.3f}")
    print(f"   📊 Test RMSE: {model_info.get('test_rmse', 'N/A'):.3f}")
    
    return model_info


def train_all_models(df: pd.DataFrame, targets: List[str] = None, 
                     save_dir: str = "outputs/models") -> Dict[str, Dict[str, Any]]:
    """
    Train all model types for all targets
    
    Args:
        df: Input DataFrame
        targets: List of target variables
        save_dir: Directory to save models
    
    Returns:
        Dictionary of all trained models
    """
    if targets is None:
        targets = ['temperature', 'rainfall']
    
    print("🚀 TRAINING ALL MODELS FOR ALL TARGETS")
    print("=" * 60)
    
    all_models = {}
    model_types = ['linear', 'random_forest', 'prophet', 'lstm']
    
    for target in targets:
        all_models[target] = {}
        
        for model_type in model_types:
            try:
                save_path = f"{save_dir}/{target}_{model_type}_model.pkl"
                model_info = train_model(df, target, model_type, save_path=save_path)
                all_models[target][model_type] = model_info
                
            except Exception as e:
                print(f"❌ Error training {model_type} for {target}: {e}")
                continue
    
    print(f"\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
    return all_models


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    sample_data = {
        'date': dates,
        'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates)),
        'aqi': 70 + np.random.normal(0, 15, len(dates)),
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'day': [d.day for d in dates]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Test individual model training
    print("Testing Linear Regression...")
    model_info = train_model(df_sample, 'temperature', 'linear')
    
    print("\nTesting Random Forest...")
    model_info = train_model(df_sample, 'temperature', 'random_forest')
    
    print("\n✅ Model training tests completed!")