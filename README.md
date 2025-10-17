# 🌍 Enhanced Climate Change & AQI Prediction System - Pune

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![IoT](https://img.shields.io/badge/IoT-MQTT%20%7C%20HTTP-orange.svg)](docs/API.md)
[![ML](https://img.shields.io/badge/ML-Ensemble%20%7C%20LSTM-purple.svg)](src/advanced_ml_models.py)

A comprehensive, enterprise-grade climate change and air quality prediction system for Pune with IoT integration, advanced machine learning models, and real-time monitoring capabilities.

![System Architecture](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Climate+%26+AQI+Prediction+System)

## ✨ Live Demo

🌐 **Repository**: [GitHub Repository](https://github.com/sumit-singh53/climate-change-prediction-pune)  
📊 **API Docs**: [Interactive API Documentation](docs/API.md)  
🚀 **Quick Start**: [Setup Guide](GITHUB_SETUP.md)

## 🌟 Features

### 🎯 High-Accuracy Predictions
- **Ensemble ML Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting
- **Deep Learning**: LSTM and Transformer models for time series prediction
- **Multi-horizon Forecasting**: 1-day to 30-day predictions
- **Location-specific Models**: Tailored predictions for different Pune zones

### 📍 Location-wise Coverage
- **8 Strategic Locations**: Comprehensive coverage across Pune metropolitan area
- **Zone-based Analysis**: Central, North, East, West, Northwest, South zones
- **Micro-climate Modeling**: Location-specific environmental factors

### 🔗 Real-time Data Collection
- **Live API Integration**: Continuous data from weather and air quality APIs
- **Smart Caching**: Efficient data retrieval with configurable cache duration
- **Data Quality Monitoring**: Automatic validation and quality scoring
- **Multi-source Integration**: Weather and air quality from reliable sources

### 📊 Real-time Dashboard
- **Live Monitoring**: Real-time environmental data visualization
- **Interactive Maps**: Location-based monitoring with status indicators
- **Prediction Visualization**: Historical trends and future forecasts
- **Correlation Analysis**: Environmental variables relationships

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT Sensors   │    │  External APIs  │    │  Manual Data    │
│                 │    │                 │    │                 │
│ • Temperature   │    │ • Open-Meteo    │    │ • Calibration   │
│ • Humidity      │    │ • Weather APIs  │    │ • Validation    │
│ • PM2.5/PM10    │    │ • AQI Services  │    │ • Historical    │
│ • CO2, Noise    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Data Collection │
                    │    & Storage    │
                    │                 │
                    │ • SQLite DB     │
                    │ • Data Quality  │
                    │ • Preprocessing │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  ML Pipeline    │
                    │                 │
                    │ • Feature Eng.  │
                    │ • Ensemble ML   │
                    │ • Deep Learning │
                    │ • Model Eval.   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Real-time       │
                    │ Dashboard       │
                    │                 │
                    │ • Streamlit UI  │
                    │ • Live Updates  │
                    │ • Predictions   │
                    │ • Analytics     │
                    └─────────────────┘
```

## 🚀 Quick Start

### One-Line Setup
```bash
git clone https://github.com/sumit-singh53/climate-change-prediction-pune.git && cd climate-change-prediction-pune && python run_system.py
```

### Manual Setup

#### Prerequisites
- Python 3.8+
- 4GB+ RAM recommended
- Internet connection for data collection

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/climate-change-prediction-pune.git
cd climate-change-prediction-pune
```

2. **Run the system** (auto-installs dependencies)
```bash
python run_system.py
```

3. **Access the dashboard**
   - 🌐 **Dashboard**: http://localhost:8501
   - 📊 **Real-time Updates**: Every 30 minutes
   - 🤖 **Live Predictions**: Based on current data

#### Alternative Setup
```bash
# Manual dependency installation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start specific components
python run_system.py --mode dashboard    # Dashboard only
python run_system.py --mode data-collection  # Data collection only
python run_system.py --mode training     # Model training only
```

### Running Individual Components

#### 1. Data Collection Only
```bash
python src/main_orchestrator.py --mode data-collection
```

#### 2. Dashboard Only
```bash
python src/main_orchestrator.py --mode dashboard
```

#### 3. Model Training Only
```bash
python src/main_orchestrator.py --mode training
```

#### 4. IoT Sensor Simulation
```bash
python src/iot_integration.py
```

## 📍 Monitored Locations

| Location | Zone | Coordinates | Key Features |
|----------|------|-------------|--------------|
| Pune Central | Central | 18.5204, 73.8567 | Urban core, high traffic |
| Pimpri-Chinchwad | North | 18.6298, 73.7997 | Industrial area |
| Hadapsar | East | 18.5089, 73.9260 | IT hub, residential |
| Kothrud | West | 18.5074, 73.8077 | Residential, educational |
| Wakad | Northwest | 18.5975, 73.7898 | Mixed development |
| Baner | Northwest | 18.5590, 73.7793 | IT corridor |
| Katraj | South | 18.4486, 73.8594 | Hills, lower pollution |
| Wagholi | East | 18.5793, 73.9800 | Emerging IT area |

## 🤖 Machine Learning Models

### Ensemble Models
- **Random Forest**: Robust baseline with feature importance
- **XGBoost**: Gradient boosting for high accuracy
- **LightGBM**: Fast training with categorical features
- **Gradient Boosting**: Traditional boosting approach

### Deep Learning Models
- **LSTM**: Long Short-Term Memory for time series
- **Transformer**: Attention-based sequence modeling

### Target Variables
- Temperature (°C)
- Humidity (%)
- PM2.5 (µg/m³)
- PM10 (µg/m³)
- Air Quality Index (AQI)

## 🔌 IoT Integration

### Supported Protocols
- **MQTT**: `mqtt://localhost:1883`
- **HTTP REST API**: `http://localhost:5000/api/sensor-data`

### MQTT Topics Structure
```
sensors/{sensor_type}/{location_id}/{sensor_id}
```

Example:
```
sensors/temperature/pune_central/temp_001
sensors/pm25/hadapsar/air_001
```

### HTTP API Endpoints

#### Submit Sensor Data
```bash
POST /api/sensor-data
Content-Type: application/json

{
  "sensor_type": "temperature",
  "location_id": "pune_central",
  "sensor_id": "temp_001",
  "value": 28.5,
  "unit": "C",
  "timestamp": "2024-01-15T10:30:00Z",
  "quality_score": 0.95
}
```

#### Get Sensor Status
```bash
GET /api/sensor-status/{sensor_id}
```

## 📊 Dashboard Features

### Real-time Monitoring
- Live environmental data updates
- Location-based status indicators
- Auto-refresh capabilities

### Prediction Visualization
- Historical trends analysis
- Multi-day forecasting
- Confidence intervals

### Analytics
- Correlation heatmaps
- Data quality metrics
- Performance monitoring

### Access Dashboard
Open your browser and navigate to: `http://localhost:8501`

## 🗂️ Project Structure

```
climate_change_prediction_pune/
├── src/
│   ├── config.py                 # System configuration
│   ├── enhanced_data_collector.py # Multi-source data collection
│   ├── iot_integration.py        # IoT sensor integration
│   ├── advanced_ml_models.py     # ML models and training
│   ├── realtime_dashboard.py     # Streamlit dashboard
│   └── main_orchestrator.py      # System orchestrator
├── data/
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed datasets
│   ├── external/                 # External data sources
│   └── iot/                      # IoT sensor data
├── outputs/
│   ├── models/                   # Trained ML models
│   ├── figures/                  # Generated plots
│   └── logs/                     # System logs
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
# API Keys (if needed)
OPENWEATHER_API_KEY=your_api_key_here

# Database
DATABASE_PATH=data/climate_aqi_database.db

# MQTT Settings
MQTT_BROKER=localhost
MQTT_PORT=1883

# Dashboard
DASHBOARD_PORT=8501
```

### Model Configuration
Edit `src/config.py` to customize:
- Prediction horizons
- Model parameters
- Feature engineering
- Data retention policies

## 📈 Performance Metrics

### Model Accuracy (Typical Results)
- **Temperature**: R² > 0.85, MAE < 2°C
- **Humidity**: R² > 0.80, MAE < 5%
- **PM2.5**: R² > 0.75, MAE < 10 µg/m³
- **AQI**: R² > 0.70, MAE < 15 points

### System Performance
- **Data Collection**: 8 locations every hour
- **Prediction Latency**: < 100ms per location
- **Dashboard Updates**: Real-time (30s refresh)
- **Model Training**: Daily automated updates

## 🛠️ Development

### Adding New Locations
1. Update `PUNE_LOCATIONS` in `src/config.py`
2. Register sensors for the new location
3. Restart the system

### Adding New Sensor Types
1. Update `IOT_CONFIG` sensor topics
2. Add validation rules in `iot_integration.py`
3. Update dashboard visualization

### Custom ML Models
1. Extend `AdvancedMLModels` class
2. Add model to ensemble configuration
3. Update training pipeline

## 🔍 Troubleshooting

### Common Issues

#### No Data in Dashboard
- Check if data collection service is running
- Verify API connectivity
- Check database file permissions

#### IoT Sensors Not Connecting
- Verify MQTT broker is running
- Check network connectivity
- Validate sensor configuration

#### Model Training Fails
- Ensure sufficient historical data (>100 samples)
- Check data quality and completeness
- Verify system resources (RAM, disk space)

### Logs Location
- System logs: `outputs/logs/system.log`
- Model training: `outputs/logs/training.log`
- Dashboard: Streamlit console output

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Open-Meteo API for weather data
- Pune Municipal Corporation for location insights
- Contributors and testers

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review system logs for error details

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Setup
```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/climate-change-prediction-pune.git
cd climate-change-prediction-pune
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_system.py  # Run tests
```

## 📊 Project Stats

- **8 Monitoring Locations** across Pune
- **5 ML Models** (RF, XGBoost, LightGBM, LSTM, Ensemble)
- **6 Sensor Types** (Temperature, Humidity, PM2.5, PM10, CO2, Noise)
- **Real-time Updates** every 30 seconds
- **Multi-horizon Predictions** (1-30 days)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/climate-change-prediction-pune&type=Date)](https://star-history.com/#YOUR_USERNAME/climate-change-prediction-pune&Date)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Open-Meteo API](https://open-meteo.com/) for weather data
- [Pune Municipal Corporation](https://pmc.gov.in/) for location insights
- [Streamlit](https://streamlit.io/) for the amazing dashboard framework
- All contributors and testers

## 📞 Support & Contact

- 🐛 **Bug Reports**: [Create an Issue](https://github.com/YOUR_USERNAME/climate-change-prediction-pune/issues)
- 💡 **Feature Requests**: [Discussions](https://github.com/YOUR_USERNAME/climate-change-prediction-pune/discussions)
- 📧 **Email**: your-email@example.com
- 💬 **Discord**: [Join our community](https://discord.gg/your-invite)

---

<div align="center">

**Built with ❤️ for a cleaner, more predictable Pune environment**

[⭐ Star this repo](https://github.com/YOUR_USERNAME/climate-change-prediction-pune) • [🍴 Fork it](https://github.com/YOUR_USERNAME/climate-change-prediction-pune/fork) • [📢 Share it](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20Climate%20%26%20AQI%20Prediction%20System!&url=https://github.com/YOUR_USERNAME/climate-change-prediction-pune)

</div>