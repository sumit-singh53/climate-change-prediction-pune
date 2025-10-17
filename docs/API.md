# Data Sources & System Architecture

This document describes the data sources, system architecture, and internal APIs of the Enhanced Climate & AQI Prediction System.

## 🌐 External Data Sources

### Open-Meteo APIs
The system automatically fetches data from Open-Meteo, a reliable weather and air quality API service.

#### Weather Data API
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Parameters**: Current weather conditions for all 8 Pune locations
- **Data Points**: Temperature, humidity, pressure, wind, precipitation, solar radiation, UV index
- **Update Frequency**: Every 30 minutes

#### Air Quality API
- **Endpoint**: `https://air-quality-api.open-meteo.com/v1/air-quality`
- **Parameters**: Current air quality conditions
- **Data Points**: PM2.5, PM10, NO2, SO2, CO, O3, AQI
- **Update Frequency**: Every 30 minutes

---

## 🗄️ Database Schema

### Real-time Weather Data
```sql
CREATE TABLE realtime_weather (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    location_id TEXT,
    temperature REAL,
    humidity REAL,
    pressure REAL,
    wind_speed REAL,
    wind_direction REAL,
    precipitation REAL,
    solar_radiation REAL,
    uv_index REAL,
    visibility REAL,
    cloud_cover REAL,
    feels_like REAL,
    dew_point REAL,
    data_source TEXT,
    quality_score REAL
);
```

### Real-time Air Quality Data
```sql
CREATE TABLE realtime_air_quality (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    location_id TEXT,
    pm25 REAL,
    pm10 REAL,
    no2 REAL,
    so2 REAL,
    co REAL,
    o3 REAL,
    aqi REAL,
    aqi_category TEXT,
    dominant_pollutant TEXT,
    health_recommendation TEXT,
    data_source TEXT,
    quality_score REAL
);
```

### Location Metadata
```sql
CREATE TABLE location_metadata (
    location_id TEXT PRIMARY KEY,
    name TEXT,
    latitude REAL,
    longitude REAL,
    district TEXT,
    zone TEXT,
    elevation REAL,
    population_density REAL,
    last_updated DATETIME
);
```

---

## 🏗️ System Architecture

### Data Flow
```
External APIs → Real-time Collector → Database → ML Models → Dashboard
     ↓              ↓                    ↓          ↓          ↓
Open-Meteo    Smart Caching        SQLite    Predictions   Streamlit
```

### Components

#### 1. Real-time Data Collector (`realtime_data_collector.py`)
- Fetches data from external APIs
- Implements smart caching to avoid rate limits
- Validates and processes incoming data
- Stores data in SQLite database

#### 2. ML Models (`advanced_ml_models.py`)
- Ensemble models: Random Forest, XGBoost, LightGBM
- Deep learning: LSTM networks
- Feature engineering and preprocessing
- Multi-horizon predictions (1-30 days)

#### 3. Dashboard (`realtime_dashboard.py`)
- Streamlit-based web interface
- Real-time data visualization
- Interactive maps and charts
- Prediction displays

#### 4. System Orchestrator (`main_orchestrator.py`)
- Coordinates all system components
- Manages background services
- Handles system startup and shutdown

---

## 📊 Data Processing Pipeline

### 1. Data Collection
```python
# Automatic data collection every 30 minutes
weather_data = await collector.fetch_weather_data(location_id)
air_quality_data = await collector.fetch_air_quality_data(location_id)
```

### 2. Data Processing
```python
# Process and enhance raw API data
processed_data = {
    'timestamp': current_time,
    'location_id': location_id,
    'temperature': raw_data['temperature_2m'],
    'aqi_category': get_aqi_category(aqi_value),
    'health_recommendation': get_health_recommendation(aqi_value),
    'quality_score': calculate_quality_score(raw_data)
}
```

### 3. ML Predictions
```python
# Generate predictions using ensemble models
prediction = ml_models.predict(
    location_id='pune_central',
    target_variable='temperature',
    horizon_days=7
)
```

---

## 🔧 Configuration

### API Configuration
```python
API_CONFIG = {
    'open_meteo': {
        'weather_url': 'https://api.open-meteo.com/v1/forecast',
        'air_quality_url': 'https://air-quality-api.open-meteo.com/v1/air-quality'
    },
    'rate_limit': 10000,  # requests per day
    'timeout': 30
}
```

### Real-time Configuration
```python
REALTIME_CONFIG = {
    'update_interval_minutes': 30,
    'data_retention_days': 365,
    'max_api_calls_per_hour': 1000,
    'enable_caching': True,
    'cache_duration_minutes': 15
}
```

---

## 📈 Data Quality & Monitoring

### Quality Scoring
Each data point receives a quality score based on:
- **Completeness**: Percentage of fields with valid data
- **Validity**: Values within expected ranges
- **Timeliness**: Data freshness
- **Consistency**: Comparison with historical patterns

### Health Recommendations
AQI-based health recommendations:
- **0-50**: Good - Ideal for outdoor activities
- **51-100**: Moderate - Acceptable for most people
- **101-150**: Unhealthy for Sensitive Groups
- **151-200**: Unhealthy - Limit outdoor activities
- **201-300**: Very Unhealthy - Avoid outdoor activities
- **300+**: Hazardous - Health alert

---

## 🚀 Usage Examples

### Getting Latest Data
```python
from src.realtime_data_collector import RealtimeDataCollector

collector = RealtimeDataCollector()
latest_data = collector.get_latest_data('pune_central')
```

### Making Predictions
```python
from src.advanced_ml_models import AdvancedMLModels

ml_models = AdvancedMLModels()
prediction = ml_models.predict(
    location_id='pune_central',
    target_variable='pm25',
    horizon_days=3
)
```

### Running Dashboard
```bash
# Start the complete system
python run_system.py

# Dashboard only
python run_system.py --mode dashboard
```

---

## 📞 Support

For technical questions about the system architecture or data sources:
- Check the main [README.md](../README.md) for setup instructions
- Review the [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- Create an issue on GitHub for specific problems