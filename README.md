# 🌍 Pune Climate Change Prediction Dashboard

A comprehensive, AI-powered climate analysis and prediction system for Pune, India. This interactive dashboard provides historical climate data analysis, real-time monitoring, and machine learning-based future predictions.

![Climate Dashboard](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🚀 Live Demo

**[Access the Live Dashboard](https://your-username-pune-climate-dashboard-streamlit-app-xyz.streamlit.app)**

## ✨ Features

### 📊 **Comprehensive Climate Analysis**
- **151,952+ historical records** from 2000-2025
- **8 Pune locations** with detailed coverage
- **Temperature, rainfall, and air quality trends**
- **Seasonal pattern analysis**

### 🤖 **Machine Learning Predictions**
- **Future climate forecasts** (2025-2050)
- **Multiple ML algorithms** (Random Forest, Linear Regression)
- **Interactive prediction visualizations**
- **Climate change impact assessment**

### 📍 **Location-Specific Analysis**
- **Individual location analysis** for 8 Pune areas
- **Location comparison tools**
- **Seasonal climate profiles**
- **Area-specific climate trends**

### 🎛️ **Interactive Controls**
- **Date range filtering**
- **Location selection and comparison**
- **Dynamic chart updates**
- **Real-time data visualization**

## 🏙️ Covered Locations

- **Pune Central** (Central Zone)
- **Pimpri-Chinchwad** (North Zone)
- **Hadapsar** (East Zone)
- **Kothrud** (West Zone)
- **Wakad** (Northwest Zone)
- **Baner** (Northwest Zone)
- **Katraj** (South Zone)
- **Wagholi** (East Zone)

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/pune-climate-dashboard.git
cd pune-climate-dashboard

# Install dependencies
pip install -r requirements.txt

# Generate climate data
python reset_database.py

# Run the dashboard
python run_climate_dashboard.py
```

Access at: `http://localhost:8501`

### Option 2: Deploy to Streamlit Cloud

1. **Fork this repository**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Select this repository**
5. **Set main file**: `streamlit_app.py`
6. **Deploy!**

## 📊 Data Overview

### Historical Climate Data (2000-2025)
- **Temperature**: Daily averages with seasonal variations
- **Rainfall**: Monthly precipitation patterns
- **Humidity**: Relative humidity measurements
- **Air Quality**: AQI, PM2.5, PM10, and other pollutants
- **Weather Patterns**: Wind speed, pressure, solar radiation

### Data Sources
- **Realistic climate modeling** based on Pune's geography
- **Seasonal pattern simulation** (Winter, Summer, Monsoon, Post-Monsoon)
- **Climate change trends** incorporated
- **Location-specific variations** for different Pune areas

## 🔮 Predictions & Analysis

### Climate Change Indicators
- **Temperature trends**: Rising temperatures over time
- **Rainfall patterns**: Changing precipitation cycles
- **Air quality**: Urban pollution trends
- **Seasonal shifts**: Evolving weather patterns

### Future Projections (2025-2050)
- **Temperature forecasts**: Expected warming trends
- **Rainfall predictions**: Precipitation changes
- **Air quality outlook**: Pollution trajectory
- **Climate impact assessment**: Risk evaluation

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualizations**: Plotly, Matplotlib
- **Database**: SQLite
- **Deployment**: Streamlit Cloud, Railway, Render

## 📁 Project Structure

```
pune-climate-dashboard/
├── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── climate_dashboard.py  # Dashboard implementation
│   ├── comprehensive_data_generator.py  # Data generation
│   └── config.py            # Configuration settings
├── data/                    # Database and data files
├── requirements.txt         # Python dependencies
├── requirements_deploy.txt  # Lightweight deployment deps
├── .streamlit/             # Streamlit configuration
└── README.md               # This file
```

## 🎯 Usage Examples

### Analyze Specific Location
1. Select "Specific Location" mode
2. Choose any Pune area (e.g., "Pune Central")
3. View detailed climate analysis and trends

### Compare Multiple Areas
1. Select "Compare Locations" mode
2. Choose 2-8 locations to compare
3. Analyze differences in climate patterns

### Future Predictions
1. Navigate to the predictions section
2. View ML-generated forecasts for 2025-2050
3. Analyze climate change impacts

## 📈 Key Insights

### Climate Trends (2000-2024)
- **Temperature**: Gradual warming trend (+0.5°C per decade)
- **Rainfall**: Slight decrease in annual precipitation
- **Air Quality**: Increasing AQI due to urbanization
- **Seasonal Changes**: Shifting monsoon patterns

### Future Outlook (2025-2050)
- **Continued warming**: 1-2°C increase expected
- **Rainfall variability**: More extreme weather events
- **Air quality challenges**: Urban pollution concerns
- **Adaptation needs**: Climate resilience planning

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pune Municipal Corporation** for climate data insights
- **Indian Meteorological Department** for weather patterns
- **Streamlit Community** for the amazing framework
- **Open Source Contributors** for various libraries used

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/pune-climate-dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/pune-climate-dashboard/discussions)
- **Email**: your-email@example.com

---

**🌍 Built with ❤️ for climate science and environmental sustainability**

*Help us understand and adapt to climate change in Pune!*