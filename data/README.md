# Data Directory

This directory contains all data files used by the Enhanced Climate & AQI Prediction System.

## Directory Structure

```
data/
├── raw/           # Raw data from APIs and sensors
├── processed/     # Cleaned and processed data
├── external/      # External datasets
└── api/          # API response cache
```

## Data Sources

### 1. Weather Data
- **Source**: Open-Meteo API
- **Coverage**: 8 locations across Pune
- **Parameters**: Temperature, humidity, pressure, wind, precipitation, solar radiation
- **Update Frequency**: Hourly

### 2. Air Quality Data
- **Source**: Open-Meteo Air Quality API
- **Coverage**: Same 8 locations
- **Parameters**: PM2.5, PM10, NO2, SO2, CO, O3, AQI
- **Update Frequency**: Hourly

### 3. Historical Climate Data
- **Source**: External climate datasets
- **Coverage**: Long-term historical data for Pune region
- **Parameters**: Climate trends and patterns
- **Update Frequency**: Static historical data

## Data Format

All data is stored in standardized formats:
- **CSV files** for tabular data
- **JSON files** for API responses
- **SQLite database** for operational data

## Data Quality

The system automatically:
- Validates incoming data
- Scores data quality (0.0-1.0)
- Handles missing values
- Detects outliers
- Maintains data lineage

## Privacy & Security

- No personal data is collected
- All environmental data is anonymized
- Data retention follows configured policies
- Regular cleanup of old data files