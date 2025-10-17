# 📁 Project Structure

This document provides an overview of the Enhanced Climate & AQI Prediction System project structure.

## 🏗️ Directory Layout

```
climate_change_prediction_pune/
├── 📊 src/                          # Core system modules
│   ├── config.py                    # System configuration
│   ├── realtime_data_collector.py   # Real-time API data collection
│   ├── enhanced_data_collector.py   # Historical data collection
│   ├── advanced_ml_models.py        # ML models & training
│   ├── realtime_dashboard.py        # Streamlit dashboard
│   └── main_orchestrator.py         # System orchestration
├── 📚 docs/                         # Documentation
│   ├── API.md                       # Data sources & architecture
│   └── DEPLOYMENT.md                # Deployment guide
├── 🔧 .github/                      # GitHub configuration
│   ├── workflows/ci-cd.yml          # GitHub Actions CI/CD
│   └── workflows/test-config.yml    # Quick validation tests
├── 📂 data/                         # Data storage (auto-created)
│   ├── raw/                         # Raw API data
│   ├── processed/                   # Processed datasets
│   ├── external/                    # External data sources
│   └── realtime/                    # Real-time data cache
├── 📈 outputs/                      # Generated outputs (auto-created)
│   ├── models/                      # Trained ML models
│   ├── figures/                     # Generated plots
│   └── logs/                        # System logs
├── 📓 notebooks/                    # Jupyter notebooks (legacy)
│   └── README.md                    # Notebook documentation
├── 🐳 Docker files                  # Containerization
│   ├── Dockerfile                   # Container definition
│   └── docker-compose.yml           # Multi-service setup
├── 📋 Configuration & Setup
│   ├── requirements.txt             # Python dependencies
│   ├── requirements-dev.txt         # Development dependencies
│   ├── setup.py                     # Package setup
│   ├── pytest.ini                  # Test configuration
│   └── .gitignore                   # Git ignore rules
├── 🚀 Entry Points
│   ├── run_system.py                # Main system launcher
│   └── test_system.py               # System validation tests
└── 📖 Documentation
    ├── README.md                    # Main project documentation
    ├── LICENSE                      # MIT License
    ├── CONTRIBUTING.md              # Contribution guidelines
    └── CHANGELOG.md                 # Version history
```

## 🔧 Core Components

### 1. Data Collection (`src/`)
- **`realtime_data_collector.py`**: Fetches live data from weather APIs every 30 minutes
- **`enhanced_data_collector.py`**: Collects historical data for model training
- **`config.py`**: Central configuration for all system components

### 2. Machine Learning (`src/`)
- **`advanced_ml_models.py`**: Ensemble models (RF, XGBoost, LightGBM, LSTM)
- Supports multi-horizon predictions (1-30 days)
- Automatic model training and evaluation

### 3. User Interface (`src/`)
- **`realtime_dashboard.py`**: Streamlit-based web dashboard
- Real-time data visualization and predictions
- Interactive maps and charts

### 4. System Management (`src/`)
- **`main_orchestrator.py`**: Coordinates all system components
- **`run_system.py`**: Easy-to-use system launcher
- **`test_system.py`**: Comprehensive system validation

## 📊 Data Flow

```
External APIs → Real-time Collector → Database → ML Models → Dashboard
     ↓              ↓                    ↓          ↓          ↓
Open-Meteo    Smart Caching        SQLite    Predictions   Streamlit
```

## 🎯 Key Features

### ✅ **Automated Data Collection**
- Fetches data from reliable weather APIs
- Smart caching to avoid rate limits
- Data quality validation and scoring

### ✅ **Advanced Machine Learning**
- Ensemble models for high accuracy
- Deep learning with LSTM networks
- Multi-location and multi-horizon predictions

### ✅ **Real-time Dashboard**
- Live data visualization
- Interactive maps of all 8 Pune locations
- Health recommendations based on AQI

### ✅ **Production Ready**
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Comprehensive testing and validation

## 🚀 Quick Start

### Run the Complete System
```bash
python run_system.py
```

### Run Individual Components
```bash
# Dashboard only
python run_system.py --mode dashboard

# Data collection only
python run_system.py --mode data-collection

# Model training only
python run_system.py --mode training
```

### Docker Deployment
```bash
docker-compose up -d
```

## 📈 Monitoring Locations

The system monitors 8 strategic locations across Pune:

| Location | Zone | Coordinates | Features |
|----------|------|-------------|----------|
| Pune Central | Central | 18.5204, 73.8567 | Urban core, high traffic |
| Pimpri-Chinchwad | North | 18.6298, 73.7997 | Industrial area |
| Hadapsar | East | 18.5089, 73.9260 | IT hub, residential |
| Kothrud | West | 18.5074, 73.8077 | Residential, educational |
| Wakad | Northwest | 18.5975, 73.7898 | Mixed development |
| Baner | Northwest | 18.5590, 73.7793 | IT corridor |
| Katraj | South | 18.4486, 73.8594 | Hills, lower pollution |
| Wagholi | East | 18.5793, 73.9800 | Emerging IT area |

## 🔍 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB disk space
- Internet connection

### Recommended
- Python 3.9+
- 8GB RAM
- 5GB disk space
- Stable internet connection

## 📞 Support

- **Documentation**: Check [README.md](README.md) for detailed setup
- **Issues**: Create GitHub issues for bugs or feature requests
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines