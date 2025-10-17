# Configuration file for enhanced climate and AQI prediction system

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class LocationConfig:
    """Configuration for different locations in Pune"""
    name: str
    lat: float
    lon: float
    district: str
    zone: str

# Pune locations for comprehensive coverage
PUNE_LOCATIONS = {
    'pune_central': LocationConfig('Pune Central', 18.5204, 73.8567, 'Pune', 'Central'),
    'pimpri_chinchwad': LocationConfig('Pimpri-Chinchwad', 18.6298, 73.7997, 'Pimpri-Chinchwad', 'North'),
    'hadapsar': LocationConfig('Hadapsar', 18.5089, 73.9260, 'Pune', 'East'),
    'kothrud': LocationConfig('Kothrud', 18.5074, 73.8077, 'Pune', 'West'),
    'wakad': LocationConfig('Wakad', 18.5975, 73.7898, 'Pune', 'Northwest'),
    'baner': LocationConfig('Baner', 18.5590, 73.7793, 'Pune', 'Northwest'),
    'katraj': LocationConfig('Katraj', 18.4486, 73.8594, 'Pune', 'South'),
    'wagholi': LocationConfig('Wagholi', 18.5793, 73.9800, 'Pune', 'East'),
}

# API Configuration
API_CONFIG = {
    'open_meteo': {
        'weather_url': 'https://api.open-meteo.com/v1/forecast',
        'air_quality_url': 'https://air-quality-api.open-meteo.com/v1/air-quality',
        'historical_url': 'https://archive-api.open-meteo.com/v1/archive'
    },
    'rate_limit': 10000,  # requests per day
    'timeout': 30
}

# Real-time Data Configuration
REALTIME_CONFIG = {
    'update_interval_minutes': 30,  # How often to fetch new data
    'data_retention_days': 365,
    'max_api_calls_per_hour': 1000,
    'enable_caching': True,
    'cache_duration_minutes': 15
}

# Model Configuration
MODEL_CONFIG = {
    'ensemble_models': ['random_forest', 'xgboost', 'lstm', 'transformer'],
    'prediction_horizons': [1, 3, 7, 30],  # days
    'features': {
        'weather': ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'precipitation'],
        'air_quality': ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3'],
        'temporal': ['hour', 'day_of_week', 'month', 'season'],
        'location': ['lat', 'lon', 'elevation', 'land_use']
    },
    'target_variables': ['temperature', 'humidity', 'pm25', 'pm10', 'aqi'],
    'model_update_frequency': 'daily'
}

# Database Configuration
DATABASE_CONFIG = {
    'sqlite_path': 'data/climate_aqi_database.db',
    'tables': {
        'weather_data': 'weather_historical',
        'air_quality_data': 'air_quality_historical',
        'realtime_data': 'realtime_api_data',
        'predictions': 'model_predictions',
        'model_performance': 'model_metrics'
    }
}

# Paths
PATHS = {
    'data': {
        'raw': 'data/raw',
        'processed': 'data/processed',
        'external': 'data/external',
        'realtime': 'data/realtime'
    },
    'models': 'outputs/models',
    'logs': 'outputs/logs',
    'figures': 'outputs/figures'
}