# 🌡️ Pune Climate Prediction System - Usage Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Setup
```bash
# Clone and setup
git clone <repository-url>
cd climate_change_prediction_pune
python setup.py
```

### Step 2: Test Backend
```bash
# Verify everything works
python test_backend.py
```

### Step 3: Launch App
```bash
# Start the web application
python run_app.py
```

**🌐 Open your browser to: http://localhost:8501**

---

## 📊 Using the Web Application

### 1. Data Overview Tab
- **Load Data**: Click "Load Climate Data" to fetch historical data
- **Configure Range**: Use sidebar to set date range (2000-2024)
- **Review Quality**: Check data statistics and missing values
- **Data Preview**: Examine the first 10 records

### 2. Model Training Tab
- **Preprocess**: Click "Preprocess Data" to clean and prepare data
- **Select Models**: Choose from Linear, Random Forest, Prophet, LSTM
- **Train Models**: Click "Train All Models" (may take 2-5 minutes)
- **Review Performance**: Check R², RMSE, and MAE metrics

### 3. Predictions Tab
- **Set Time Range**: Configure prediction years (2026-2050)
- **Generate Predictions**: Click "Generate Predictions"
- **View Results**: Review average, min, max predictions
- **Compare Models**: See different model predictions

### 4. Visualizations Tab
- **Create Charts**: Click "Generate Visualizations"
- **Time Series**: Historical trends and patterns
- **Seasonal Analysis**: Monthly and seasonal breakdowns
- **Correlations**: Variable relationship heatmaps
- **Predictions**: Future vs historical comparisons

### 5. Reports Tab
- **Configure Report**: Select what to include
- **Generate PDF**: Click "Generate Report"
- **Download**: Use download button for PDF report
- **Preview**: Review report summary statistics

---

## 🎛️ Configuration Options

### Sidebar Controls

#### Data Configuration
- **Start Year**: Historical data start (2000-2023)
- **End Year**: Historical data end (2021-2024)
- **Include Current**: Add real-time data

#### Model Configuration
- **Target Variables**: Choose what to predict
  - Temperature (°C)
  - Rainfall (mm)
  - Humidity (%)
  - AQI (Air Quality Index)
- **Model Types**: Select ML algorithms
  - Linear Regression (fast, interpretable)
  - Random Forest (accurate, robust)
  - Prophet (time series specialist)
  - LSTM (deep learning, complex patterns)

#### Prediction Configuration
- **Start Year**: When predictions begin (2025-2030)
- **End Year**: When predictions end (2030-2050)

---

## 📈 Understanding the Results

### Model Performance Metrics

#### R² Score (Coefficient of Determination)
- **Range**: 0 to 1
- **Interpretation**: 
  - 0.9+ = Excellent
  - 0.8-0.9 = Good
  - 0.6-0.8 = Fair
  - <0.6 = Poor

#### RMSE (Root Mean Square Error)
- **Units**: Same as target variable
- **Interpretation**: Lower is better
- **Temperature**: <2°C is good
- **Rainfall**: <5mm is good

#### MAE (Mean Absolute Error)
- **Units**: Same as target variable
- **Interpretation**: Average prediction error
- **Temperature**: <1.5°C is excellent
- **Rainfall**: <3mm is excellent

### Prediction Confidence

#### High Confidence
- **Prophet**: Provides confidence intervals
- **Ensemble**: Multiple models agree
- **Historical**: Strong seasonal patterns

#### Medium Confidence
- **Single Model**: One algorithm only
- **Limited Data**: <5 years of history
- **High Variability**: Irregular patterns

#### Low Confidence
- **Extreme Future**: >20 years ahead
- **Climate Change**: Unprecedented conditions
- **Model Disagreement**: Different predictions

---

## 🔧 Advanced Usage

### Custom Data Upload
1. Prepare CSV with columns: date, temperature, rainfall, humidity, aqi
2. Use "Upload Data" feature (coming soon)
3. Retrain models with new data

### Model Optimization
1. Enable "Optimize Hyperparameters" in Model Training
2. Increases training time but improves accuracy
3. Recommended for production use

### Batch Predictions
1. Set wide prediction range (2026-2050)
2. Generate predictions for all models
3. Export results via Reports tab

---

## 📊 Interpreting Visualizations

### Time Series Plots
- **Blue Line**: Historical data
- **Red Dashed**: Future predictions
- **Shaded Area**: Confidence intervals
- **Vertical Line**: Prediction start point

### Seasonal Analysis
- **Bar Charts**: Average by season
- **Box Plots**: Distribution and outliers
- **Line Charts**: Monthly patterns
- **Trend Lines**: Year-over-year changes

### Correlation Heatmaps
- **Red**: Strong positive correlation
- **Blue**: Strong negative correlation
- **White**: No correlation
- **Numbers**: Correlation coefficients (-1 to 1)

### Climate Trends
- **Temperature**: Generally increasing (climate change)
- **Rainfall**: Variable, monsoon-dependent
- **AQI**: Policy-dependent improvements
- **CO₂**: Steadily increasing globally

---

## 📄 Report Contents

### Executive Summary
- Key findings and trends
- Climate change indicators
- Risk assessment overview

### Data Analysis
- Statistical summaries
- Data quality metrics
- Historical patterns

### Model Performance
- Accuracy metrics for each model
- Best performing algorithms
- Prediction reliability

### Future Projections
- Temperature forecasts
- Rainfall predictions
- Seasonal changes
- Extreme events

### Risk Assessment
- Climate Risk Index (Low/Medium/High)
- Identified risk factors
- Vulnerability analysis

### Recommendations
- Adaptation strategies
- Mitigation measures
- Policy suggestions
- Infrastructure needs

---

## 🚨 Troubleshooting

### Common Issues

#### "No data loaded" Error
- **Solution**: Click "Load Climate Data" first
- **Check**: Internet connection for API calls
- **Verify**: Date range is reasonable (2000-2024)

#### Model Training Fails
- **Solution**: Ensure data is preprocessed first
- **Check**: Sufficient data (>100 records)
- **Try**: Reduce model complexity or date range

#### Predictions Not Generated
- **Solution**: Train models first
- **Check**: Prediction years are in future
- **Verify**: Historical data available

#### Visualizations Empty
- **Solution**: Load data and generate predictions
- **Check**: Target variables are selected
- **Try**: Refresh the page

#### Report Generation Fails
- **Solution**: Ensure all previous steps completed
- **Check**: Disk space for PDF creation
- **Try**: Reduce report complexity

### Performance Issues

#### Slow Loading
- **Reduce**: Date range (try 2020-2024)
- **Limit**: Number of models (try 2-3)
- **Check**: System resources (RAM, CPU)

#### Memory Errors
- **Reduce**: Feature engineering complexity
- **Limit**: Prediction time range
- **Close**: Other applications

---

## 💡 Tips for Best Results

### Data Quality
- Use recent data (2020+) for better accuracy
- Include current data for up-to-date patterns
- Check for missing values and outliers

### Model Selection
- **Quick Results**: Linear Regression
- **Best Accuracy**: Random Forest
- **Time Series**: Prophet
- **Complex Patterns**: LSTM

### Prediction Horizons
- **Short-term** (1-5 years): High accuracy
- **Medium-term** (5-15 years): Good accuracy
- **Long-term** (15+ years): Lower accuracy

### Interpretation
- Consider multiple models for robustness
- Account for climate change trends
- Use confidence intervals when available
- Validate with domain knowledge

---

## 🌍 Climate Context for Pune

### Seasonal Patterns
- **Winter** (Dec-Feb): Cool, dry (15-25°C)
- **Summer** (Mar-May): Hot, dry (25-40°C)
- **Monsoon** (Jun-Sep): Warm, wet (20-30°C)
- **Post-Monsoon** (Oct-Nov): Pleasant (20-30°C)

### Climate Change Trends
- **Temperature**: +0.5°C per decade
- **Rainfall**: More variable, intense events
- **Air Quality**: Improving with policies
- **Extreme Events**: More frequent heat waves

### Local Factors
- **Urban Heat Island**: City center warmer
- **Monsoon Dependency**: 80% rainfall in 4 months
- **Topography**: Western Ghats influence
- **Development**: Rapid urbanization effects

---

## 📞 Support

### Getting Help
- **Check**: This usage guide first
- **Run**: `python test_backend.py` for diagnostics
- **Review**: Console output for error messages
- **Search**: Error messages online

### Reporting Issues
- Include error messages
- Describe steps to reproduce
- Mention system specifications
- Attach log files if available

### Feature Requests
- Describe the desired functionality
- Explain the use case
- Suggest implementation approach

---

**🎯 Ready to explore Pune's climate future? Start with Step 1 above!**