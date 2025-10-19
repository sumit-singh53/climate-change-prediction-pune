#!/usr/bin/env python3
"""
Improved Model Trainer for Authentic Climate Data
Optimized for larger datasets with better feature selection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class ImprovedClimateModelTrainer:
    """
    Improved model trainer optimized for authentic climate data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
    
    def prepare_features_improved(self, df, target, max_features=15):
        """
        Improved feature preparation with better selection
        """
        print(f"🎯 Preparing optimized features for {target} prediction...")
        
        # Core climate features (most important for climate prediction)
        core_features = []
        
        # Time features (essential for climate patterns)
        time_features = ['year', 'month', 'day_of_year']
        for feat in time_features:
            if feat in df.columns:
                core_features.append(feat)
        
        # Add cyclical encoding for better time representation
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            core_features.extend(['month_sin', 'month_cos'])
        
        if 'day_of_year' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            core_features.extend(['day_sin', 'day_cos'])
        
        # Climate variables (excluding target)
        climate_vars = ['temperature', 'rainfall', 'humidity', 'aqi', 'wind_speed', 'pressure', 'co2']
        for var in climate_vars:
            if var in df.columns and var != target:
                core_features.append(var)
        
        # Seasonal indicators
        if 'season' in df.columns:
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
            core_features.extend(season_dummies.columns)
        
        if 'is_monsoon' in df.columns:
            core_features.append('is_monsoon')
        
        # Select only available features
        available_features = [feat for feat in core_features if feat in df.columns]
        
        # Prepare feature matrix
        X = df[available_features].copy()
        y = df[target].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Feature selection if we have too many features
        if len(available_features) > max_features:
            print(f"🔧 Selecting top {max_features} features from {len(available_features)} available")
            
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_features = [feat for feat, selected in zip(available_features, selected_mask) if selected]
            
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            self.feature_selectors[target] = selector
            
            print(f"✅ Selected features: {selected_features}")
        
        print(f"✅ Final features: {len(X.columns)} features, {len(y)} samples")
        return X, y
    
    def train_improved_model(self, df, target, model_type='random_forest'):
        """
        Train improved model with better performance
        """
        print(f"🤖 Training improved {model_type} model for {target}...")
        
        # Prepare features
        X, y = self.prepare_features_improved(df, target)
        
        # Split data (larger test set for better evaluation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, shuffle=False
        )
        
        # Scale features for better performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target] = scaler
        
        # Train model based on type
        if model_type == 'random_forest':
            # Optimized Random Forest parameters
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)  # RF doesn't need scaling
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
        elif model_type == 'ridge':
            # Ridge regression with regularization
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
        else:  # linear
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Cross-validation for more robust evaluation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Feature importance (for tree-based models)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, abs(model.coef_)))
        
        model_info = {
            'model': model,
            'model_type': model_type,
            'target': target,
            'features': list(X.columns),
            'scaler': scaler,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'feature_importance': feature_importance,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"✅ Model trained successfully!")
        print(f"   📊 Test R²: {test_r2:.3f}")
        print(f"   📊 Test MAE: {test_mae:.3f}")
        print(f"   📊 Test RMSE: {test_rmse:.3f}")
        print(f"   📊 CV R² (mean±std): {cv_mean:.3f}±{cv_std:.3f}")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.2:
            print(f"   ⚠️ Possible overfitting detected (train R²: {train_r2:.3f})")
        else:
            print(f"   ✅ Good generalization (train R²: {train_r2:.3f})")
        
        return model_info
    
    def predict_with_model(self, model_info, X_new):
        """
        Make predictions with trained model
        """
        model = model_info['model']
        scaler = model_info.get('scaler')
        model_type = model_info['model_type']
        
        # Ensure X_new has the same features
        required_features = model_info['features']
        X_pred = X_new[required_features].copy()
        
        # Handle missing values
        X_pred = X_pred.fillna(X_pred.mean())
        
        # Scale if needed
        if model_type in ['ridge', 'linear'] and scaler is not None:
            X_pred_scaled = scaler.transform(X_pred)
            predictions = model.predict(X_pred_scaled)
        else:
            predictions = model.predict(X_pred)
        
        return predictions

def demonstrate_improved_training():
    """
    Demonstrate improved model training with authentic data
    """
    print("🤖 IMPROVED MODEL TRAINING DEMONSTRATION")
    print("=" * 60)
    
    # Load authentic dataset
    data_path = "data/pune_authentic_climate_2000_2024.csv"
    
    if not os.path.exists(data_path):
        print("❌ Authentic dataset not found. Please run create_authentic_dataset.py first")
        return
    
    print("📊 Loading authentic Pune climate dataset...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df):,} records from {df['date'].min().year} to {df['date'].max().year}")
    
    # Initialize improved trainer
    trainer = ImprovedClimateModelTrainer()
    
    # Train models for different targets
    targets = ['temperature', 'rainfall']
    models = {}
    
    for target in targets:
        print(f"\n🎯 Training models for {target}...")
        
        # Try different model types
        model_types = ['random_forest', 'ridge', 'linear']
        target_models = {}
        
        for model_type in model_types:
            try:
                model_info = trainer.train_improved_model(df, target, model_type)
                target_models[model_type] = model_info
            except Exception as e:
                print(f"❌ Error training {model_type}: {e}")
        
        models[target] = target_models
        
        # Find best model for this target
        if target_models:
            best_model_name = max(target_models.keys(), 
                                key=lambda x: target_models[x]['test_r2'])
            best_r2 = target_models[best_model_name]['test_r2']
            
            print(f"🏆 Best model for {target}: {best_model_name} (R² = {best_r2:.3f})")
    
    # Summary
    print(f"\n📊 TRAINING SUMMARY")
    print("=" * 40)
    
    for target, target_models in models.items():
        print(f"\n{target.title()} Models:")
        for model_name, model_info in target_models.items():
            r2 = model_info['test_r2']
            mae = model_info['test_mae']
            cv_mean = model_info['cv_r2_mean']
            print(f"  {model_name}: R² = {r2:.3f}, MAE = {mae:.3f}, CV = {cv_mean:.3f}")
    
    return models

if __name__ == "__main__":
    import os
    demonstrate_improved_training()