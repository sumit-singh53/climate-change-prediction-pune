"""
Data Preprocessing Module
Handles missing values, normalization, feature scaling, and time-series formatting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Advanced data preprocessing for climate data"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.feature_stats = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """Handle missing values with multiple strategies"""
        print("🔧 Handling missing values...")
        
        df_clean = df.copy()
        
        # Check for missing values
        missing_info = df_clean.isnull().sum()
        missing_percent = (missing_info / len(df_clean)) * 100
        
        if missing_info.sum() == 0:
            print("✅ No missing values found")
            return df_clean
        
        print(f"⚠️ Found missing values:")
        for col, count in missing_info[missing_info > 0].items():
            print(f"   - {col}: {count} ({missing_percent[col]:.1f}%)")
        
        if strategy == 'smart':
            # Smart imputation based on data type and patterns
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if col in ['temperature', 'humidity', 'pressure']:
                        # Use forward fill then backward fill for weather data
                        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                    elif col in ['rainfall']:
                        # Rainfall can be 0, so fill with 0
                        df_clean[col] = df_clean[col].fillna(0)
                    elif col in ['aqi', 'co2']:
                        # Use interpolation for pollution data
                        df_clean[col] = df_clean[col].interpolate(method='linear')
                    else:
                        # Use median for other numeric columns
                        if df_clean[col].dtype in ['int64', 'float64']:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        else:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        elif strategy == 'knn':
            # KNN imputation for numeric columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            self.imputers['knn'] = imputer
        
        elif strategy == 'median':
            # Simple median imputation
            imputer = SimpleImputer(strategy='median')
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            self.imputers['median'] = imputer
        
        print(f"✅ Missing values handled using {strategy} strategy")
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List]:
        """Detect outliers in the data"""
        print("🔍 Detecting outliers...")
        
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_indices = df[z_scores > 3].index.tolist()
            
            if outlier_indices:
                outliers[col] = outlier_indices
                print(f"   ⚠️ {col}: {len(outlier_indices)} outliers detected")
        
        if not outliers:
            print("✅ No significant outliers detected")
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, outliers: Dict[str, List], 
                       method: str = 'cap') -> pd.DataFrame:
        """Handle outliers using various methods"""
        if not outliers:
            return df
        
        print(f"🔧 Handling outliers using {method} method...")
        df_clean = df.copy()
        
        for col, indices in outliers.items():
            if method == 'cap':
                # Cap outliers to 5th and 95th percentiles
                lower_cap = df_clean[col].quantile(0.05)
                upper_cap = df_clean[col].quantile(0.95)
                df_clean[col] = df_clean[col].clip(lower=lower_cap, upper=upper_cap)
            
            elif method == 'remove':
                # Remove outlier rows (be careful with time series)
                df_clean = df_clean.drop(indices)
            
            elif method == 'transform':
                # Log transform for skewed data
                if df_clean[col].min() > 0:
                    df_clean[col] = np.log1p(df_clean[col])
        
        print("✅ Outliers handled")
        return df_clean
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Create comprehensive time-based features"""
        print("📅 Creating time-based features...")
        
        df_time = df.copy()
        
        # Ensure date column is datetime
        df_time[date_col] = pd.to_datetime(df_time[date_col])
        
        # Basic time features
        df_time['year'] = df_time[date_col].dt.year
        df_time['month'] = df_time[date_col].dt.month
        df_time['day'] = df_time[date_col].dt.day
        df_time['day_of_week'] = df_time[date_col].dt.dayofweek
        df_time['day_of_year'] = df_time[date_col].dt.dayofyear
        df_time['week_of_year'] = df_time[date_col].dt.isocalendar().week
        df_time['quarter'] = df_time[date_col].dt.quarter
        
        # Cyclical encoding for periodic features
        df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
        df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
        df_time['day_sin'] = np.sin(2 * np.pi * df_time['day_of_year'] / 365)
        df_time['day_cos'] = np.cos(2 * np.pi * df_time['day_of_year'] / 365)
        df_time['week_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
        df_time['week_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
        
        # Season indicators
        df_time['is_winter'] = df_time['month'].isin([12, 1, 2]).astype(int)
        df_time['is_summer'] = df_time['month'].isin([3, 4, 5]).astype(int)
        df_time['is_monsoon'] = df_time['month'].isin([6, 7, 8, 9]).astype(int)
        df_time['is_post_monsoon'] = df_time['month'].isin([10, 11]).astype(int)
        
        # Long-term trends
        df_time['years_since_start'] = df_time['year'] - df_time['year'].min()
        df_time['trend'] = np.arange(len(df_time))
        
        print("✅ Time features created")
        return df_time
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str], 
                           lags: List[int] = [1, 7, 30, 365]) -> pd.DataFrame:
        """Create lag features for time series"""
        print("⏰ Creating lag features...")
        
        df_lag = df.copy()
        
        for col in target_cols:
            if col in df_lag.columns:
                for lag in lags:
                    df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag)
        
        print(f"✅ Created lag features for {len(target_cols)} variables")
        return df_lag
    
    def create_rolling_features(self, df: pd.DataFrame, target_cols: List[str],
                               windows: List[int] = [7, 30, 90, 365]) -> pd.DataFrame:
        """Create rolling statistics features"""
        print("📊 Creating rolling features...")
        
        df_roll = df.copy()
        
        for col in target_cols:
            if col in df_roll.columns:
                for window in windows:
                    df_roll[f'{col}_rolling_mean_{window}'] = df_roll[col].rolling(window, min_periods=1).mean()
                    df_roll[f'{col}_rolling_std_{window}'] = df_roll[col].rolling(window, min_periods=1).std()
                    df_roll[f'{col}_rolling_min_{window}'] = df_roll[col].rolling(window, min_periods=1).min()
                    df_roll[f'{col}_rolling_max_{window}'] = df_roll[col].rolling(window, min_periods=1).max()
        
        print(f"✅ Created rolling features for {len(target_cols)} variables")
        return df_roll
    
    def scale_features(self, df: pd.DataFrame, method: str = 'robust', 
                      exclude_cols: List[str] = None) -> pd.DataFrame:
        """Scale numerical features"""
        print(f"⚖️ Scaling features using {method} scaler...")
        
        if exclude_cols is None:
            exclude_cols = ['date', 'year', 'month', 'day', 'season']
        
        df_scaled = df.copy()
        
        # Select numeric columns to scale
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
        
        # Store scaler for inverse transform
        self.scalers[method] = scaler
        self.feature_stats['scaled_columns'] = cols_to_scale
        
        print(f"✅ Scaled {len(cols_to_scale)} features")
        return df_scaled
    
    def prepare_time_series(self, df: pd.DataFrame, target_col: str, 
                           sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for time series models (LSTM)"""
        print(f"🔄 Preparing time series data for {target_col}...")
        
        # Sort by date
        df_sorted = df.sort_values('date').copy()
        
        # Get target values
        target_values = df_sorted[target_col].values
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(target_values)):
            X.append(target_values[i-sequence_length:i])
            y.append(target_values[i])
        
        X, y = np.array(X), np.array(y)
        
        print(f"✅ Created {len(X)} sequences of length {sequence_length}")
        return X, y


def clean_and_preprocess(df: pd.DataFrame, target_variables: List[str] = None,
                        scaling_method: str = 'robust', 
                        create_features: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Main preprocessing function that handles the complete data cleaning pipeline
    
    Args:
        df: Input DataFrame
        target_variables: List of target variables for feature engineering
        scaling_method: Method for feature scaling ('standard', 'minmax', 'robust')
        create_features: Whether to create advanced features
    
    Returns:
        Dictionary containing processed DataFrames for different purposes
    """
    print("🔧 STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    if target_variables is None:
        target_variables = ['temperature', 'rainfall']
    
    preprocessor = DataPreprocessor()
    
    # Step 1: Handle missing values
    df_clean = preprocessor.handle_missing_values(df, strategy='smart')
    
    # Step 2: Detect and handle outliers
    outliers = preprocessor.detect_outliers(df_clean, method='iqr')
    df_clean = preprocessor.handle_outliers(df_clean, outliers, method='cap')
    
    # Step 3: Create time features
    df_time = preprocessor.create_time_features(df_clean)
    
    # Step 4: Create advanced features if requested
    if create_features:
        df_features = preprocessor.create_lag_features(df_time, target_variables)
        df_features = preprocessor.create_rolling_features(df_features, target_variables)
    else:
        df_features = df_time
    
    # Step 5: Scale features
    df_scaled = preprocessor.scale_features(df_features, method=scaling_method)
    
    # Step 6: Remove rows with NaN values created by lag/rolling features
    df_final = df_scaled.dropna().reset_index(drop=True)
    
    print(f"\n📊 PREPROCESSING SUMMARY:")
    print(f"   📥 Input records: {len(df):,}")
    print(f"   📤 Output records: {len(df_final):,}")
    print(f"   📊 Features created: {len(df_final.columns)}")
    print(f"   🎯 Target variables: {target_variables}")
    
    # Return different versions for different use cases
    results = {
        'raw': df,
        'cleaned': df_clean,
        'with_time_features': df_time,
        'with_all_features': df_features,
        'scaled': df_scaled,
        'final': df_final,
        'preprocessor': preprocessor
    }
    
    print("✅ Preprocessing pipeline completed!")
    return results


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    sample_data = {
        'date': dates,
        'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates)),
        'aqi': 70 + np.random.normal(0, 15, len(dates))
    }
    
    # Add some missing values for testing
    sample_data['temperature'][100:110] = np.nan
    sample_data['rainfall'][200:205] = np.nan
    
    df_sample = pd.DataFrame(sample_data)
    
    # Test preprocessing
    results = clean_and_preprocess(df_sample, target_variables=['temperature', 'rainfall'])
    
    print(f"\n✅ Preprocessing test completed!")
    print(f"Final dataset shape: {results['final'].shape}")
    print(f"Columns: {list(results['final'].columns)[:10]}...")  # Show first 10 columns