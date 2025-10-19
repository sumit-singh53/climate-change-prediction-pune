"""
Comprehensive Streamlit Climate Dashboard for Pune
Features: Home, Data Explorer, Model Training, Visualizations, AI Insights, Risk Assessment, Reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import asyncio
import os
import sys
from datetime import datetime, timedelta
from io import BytesIO
import base64

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
try:
    from data_collector import fetch_city_data
    from data_preprocessor import clean_and_preprocess
    from model_trainer import train_model
    from predictor import predict_future
    from evaluator import evaluate_model
    from report_generator import generate_report
except ImportError as e:
    st.error(f"Backend import error: {e}")

# Page configuration
st.set_page_config(
    page_title="🌡️ Pune Climate Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2rem;
        color: #2e8b57;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #F44336, #D32F2F);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'data_loaded': False,
        'models_trained': False,
        'predictions_generated': False,
        'climate_data': pd.DataFrame(),
        'uploaded_data': None,
        'trained_models': {},
        'predictions': {},
        'risk_score': 0,
        'insights': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Utility functions
@st.cache_data
def load_climate_data(start_year: int, end_year: int, include_current: bool = True, use_authentic: bool = True):
    """Load and cache climate data"""
    try:
        if use_authentic:
            # Try to load authentic dataset first
            authentic_path = "data/pune_authentic_climate_2000_2024.csv"
            if os.path.exists(authentic_path):
                st.info("📊 Loading authentic Pune climate dataset...")
                data = pd.read_csv(authentic_path)
                data['date'] = pd.to_datetime(data['date'])
                
                # Filter by requested years
                if 'year' in data.columns:
                    data = data[(data['year'] >= start_year) & (data['year'] <= end_year)]
                
                st.success(f"✅ Loaded {len(data):,} authentic climate records!")
                return data
        
        # Fallback to original method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(
            fetch_city_data("Pune", start_year, end_year, include_current)
        )
        loop.close()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_live_weather():
    """Simulate live weather data (in production, use real API)"""
    return {
        'temperature': 28.5 + np.random.normal(0, 2),
        'humidity': 65 + np.random.normal(0, 10),
        'aqi': 85 + np.random.normal(0, 15),
        'wind_speed': 12 + np.random.normal(0, 3),
        'pressure': 1013 + np.random.normal(0, 5)
    }

def calculate_risk_score(temp_rise, aqi_avg, rainfall_change):
    """Calculate climate risk score"""
    risk = 0
    
    # Temperature risk (0-40 points)
    if temp_rise > 3:
        risk += 40
    elif temp_rise > 2:
        risk += 30
    elif temp_rise > 1:
        risk += 20
    else:
        risk += 10
    
    # AQI risk (0-30 points)
    if aqi_avg > 150:
        risk += 30
    elif aqi_avg > 100:
        risk += 20
    elif aqi_avg > 50:
        risk += 10
    else:
        risk += 5
    
    # Rainfall risk (0-30 points)
    if abs(rainfall_change) > 20:
        risk += 30
    elif abs(rainfall_change) > 10:
        risk += 20
    else:
        risk += 10
    
    return min(risk, 100)

def generate_ai_insights(data, predictions=None):
    """Generate AI-powered insights"""
    insights = []
    
    if not data.empty:
        # Temperature insights
        temp_trend = data.groupby('year')['temperature'].mean().diff().mean() if 'year' in data.columns else 0
        if temp_trend > 0.05:
            insights.append(f"🌡️ Temperature is rising at {temp_trend:.2f}°C per year")
        
        # Rainfall insights
        if 'rainfall' in data.columns:
            rainfall_trend = data.groupby('year')['rainfall'].sum().pct_change().mean() * 100 if 'year' in data.columns else 0
            if abs(rainfall_trend) > 1:
                direction = "increasing" if rainfall_trend > 0 else "decreasing"
                insights.append(f"🌧️ Annual rainfall is {direction} by {abs(rainfall_trend):.1f}% per year")
        
        # AQI insights
        if 'aqi' in data.columns:
            avg_aqi = data['aqi'].mean()
            if avg_aqi > 100:
                insights.append(f"💨 Air quality is concerning with average AQI of {avg_aqi:.0f}")
            elif avg_aqi > 50:
                insights.append(f"💨 Air quality is moderate with average AQI of {avg_aqi:.0f}")
    
    # Prediction insights
    if predictions:
        for var, pred_data in predictions.items():
            if not pred_data.empty and f'{var}_predicted' in pred_data.columns:
                future_avg = pred_data[f'{var}_predicted'].mean()
                if var == 'temperature':
                    current_avg = data['temperature'].mean() if 'temperature' in data.columns else 25
                    change = future_avg - current_avg
                    if change > 1:
                        insights.append(f"🔥 Temperature expected to rise by {change:.1f}°C by 2050")
                elif var == 'rainfall':
                    current_avg = data['rainfall'].mean() if 'rainfall' in data.columns else 100
                    change_pct = ((future_avg - current_avg) / current_avg) * 100
                    if abs(change_pct) > 5:
                        direction = "increase" if change_pct > 0 else "decrease"
                        insights.append(f"🌧️ Rainfall may {direction} by {abs(change_pct):.1f}% by 2050")
    
    return insights

# Main dashboard function
def main():
    """Main dashboard application"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌡️ Pune Climate Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 2rem;">Comprehensive Climate Analysis & Prediction System</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("🎛️ Dashboard Controls")
        
        # Data configuration
        st.header("📊 Data Settings")
        start_year = st.slider("Start Year", 2000, 2023, 2020)
        end_year = st.slider("End Year", 2021, 2024, 2024)
        
        # Model configuration
        st.header("🤖 Model Settings")
        model_type = st.selectbox("Model Type", ["Linear Regression", "Random Forest", "Prophet"])
        target_param = st.selectbox("Parameter", ["Temperature", "Rainfall"])
        forecast_range = st.slider("Forecast Range", 2026, 2050, (2026, 2035))
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Home", "📊 Data Explorer", "🤖 Model Training", 
        "📈 Visualizations", "🧠 AI Insights", "⚠️ Risk Index", "📄 Reports"
    ])
    
    # Tab 1: Home Page / Dashboard
    with tab1:
        st.markdown('<h2 class="sub-header">🏠 Pune Climate Overview</h2>', unsafe_allow_html=True)
        
        # City information
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📍 City Information</h3>
                <p><strong>Name:</strong> Pune, Maharashtra</p>
                <p><strong>Coordinates:</strong> 18.5204°N, 73.8567°E</p>
                <p><strong>Population:</strong> 3.1M+</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            live_weather = get_live_weather()
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌤️ Live Weather</h3>
                <p><strong>Temperature:</strong> {live_weather['temperature']:.1f}°C</p>
                <p><strong>Humidity:</strong> {live_weather['humidity']:.0f}%</p>
                <p><strong>Wind:</strong> {live_weather['wind_speed']:.1f} km/h</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💨 Air Quality</h3>
                <p><strong>AQI:</strong> {live_weather['aqi']:.0f}</p>
                <p><strong>Status:</strong> {'Good' if live_weather['aqi'] < 50 else 'Moderate' if live_weather['aqi'] < 100 else 'Poor'}</p>
                <p><strong>Pressure:</strong> {live_weather['pressure']:.0f} hPa</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Load data for historical averages
        if st.button("📊 Load Climate Data", type="primary"):
            with st.spinner("Loading climate data..."):
                data = load_climate_data(start_year, end_year, True)
                if not data.empty:
                    st.session_state.climate_data = data
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
        
        # Historical averages summary
        if st.session_state.data_loaded and not st.session_state.climate_data.empty:
            data = st.session_state.climate_data
            
            st.markdown('<h3 style="color: #2e8b57; margin-top: 2rem;">📊 Historical Climate Summary</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_temp = data['temperature'].mean()
                st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C", 
                         delta=f"{avg_temp - 25:.1f}°C from normal")
            
            with col2:
                total_rainfall = data['rainfall'].sum()
                st.metric("🌧️ Total Rainfall", f"{total_rainfall:.0f}mm",
                         delta=f"{(total_rainfall/len(data)*365) - 700:.0f}mm from normal")
            
            with col3:
                avg_humidity = data['humidity'].mean()
                st.metric("💧 Avg Humidity", f"{avg_humidity:.0f}%",
                         delta=f"{avg_humidity - 65:.0f}% from normal")
            
            with col4:
                avg_aqi = data['aqi'].mean()
                st.metric("💨 Avg AQI", f"{avg_aqi:.0f}",
                         delta=f"{avg_aqi - 75:.0f} from baseline")
    
    # Tab 2: Data Explorer Section
    with tab2:
        st.markdown('<h2 class="sub-header">📊 Data Explorer</h2>', unsafe_allow_html=True)
        
        # File upload section
        st.markdown("### 📁 Upload Custom Dataset")
        uploaded_file = st.file_uploader("Upload custom climate dataset (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = uploaded_data
                st.success(f"✅ Uploaded dataset with {len(uploaded_data)} records")
                
                # Show uploaded data preview
                st.markdown("#### 👀 Uploaded Data Preview")
                st.dataframe(uploaded_data.head(), use_container_width=True)
                
                # Option to use uploaded data
                if st.button("🔄 Use Uploaded Data for Analysis"):
                    st.session_state.climate_data = uploaded_data
                    st.session_state.data_loaded = True
                    st.success("✅ Now using uploaded data for analysis")
                    
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {e}")
        
        # Data exploration
        if st.session_state.data_loaded and not st.session_state.climate_data.empty:
            data = st.session_state.climate_data
            
            # Tabular data display
            st.markdown("### 📋 Climate Records")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'year' in data.columns:
                    year_filter = st.selectbox("Filter by Year", 
                                             options=['All'] + sorted(data['year'].unique().tolist()))
                else:
                    year_filter = 'All'
            
            with col2:
                if 'season' in data.columns:
                    season_filter = st.selectbox("Filter by Season",
                                               options=['All'] + data['season'].unique().tolist())
                else:
                    season_filter = 'All'
            
            with col3:
                show_rows = st.selectbox("Show Rows", [10, 25, 50, 100])
            
            # Apply filters
            filtered_data = data.copy()
            if year_filter != 'All' and 'year' in data.columns:
                filtered_data = filtered_data[filtered_data['year'] == year_filter]
            if season_filter != 'All' and 'season' in data.columns:
                filtered_data = filtered_data[filtered_data['season'] == season_filter]
            
            st.dataframe(filtered_data.head(show_rows), use_container_width=True)
            
            # Correlation heatmap
            st.markdown("### 🔥 Correlation Analysis")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            climate_vars = [col for col in numeric_cols if col not in ['year', 'month', 'day', 'day_of_year']]
            
            if len(climate_vars) > 1:
                correlation_matrix = data[climate_vars].corr()
                
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    title="Climate Variables Correlation Matrix"
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Correlation insights
                st.markdown("#### 🔍 Key Correlations")
                high_corr = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            var1, var2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                            high_corr.append(f"**{var1}** ↔ **{var2}**: {corr_val:.2f}")
                
                if high_corr:
                    for corr in high_corr[:5]:  # Show top 5
                        st.write(f"• {corr}")
                else:
                    st.info("No strong correlations (>0.5) found between variables")
    
    # Tab 3: Model Training Section
    with tab3:
        st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first in the Home or Data Explorer tab")
        else:
            data = st.session_state.climate_data
            
            # Model configuration
            col1, col2 = st.columns(2)
            
            with col1:
                model_mapping = {
                    "Linear Regression": "linear",
                    "Random Forest": "random_forest", 
                    "Prophet": "prophet"
                }
                selected_model = model_mapping[model_type]
                
                param_mapping = {
                    "Temperature": "temperature",
                    "Rainfall": "rainfall"
                }
                selected_param = param_mapping[target_param]
            
            with col2:
                st.info(f"**Selected Configuration:**\n- Model: {model_type}\n- Parameter: {target_param}\n- Forecast: {forecast_range[0]}-{forecast_range[1]}")
            
            # Training section
            if st.button("🚀 Train & Predict", type="primary"):
                if selected_param not in data.columns:
                    st.error(f"❌ Parameter '{selected_param}' not found in data")
                else:
                    with st.spinner(f"Training {model_type} model for {target_param}..."):
                        try:
                            # Preprocess data
                            processed_results = clean_and_preprocess(
                                data, 
                                target_variables=[selected_param],
                                scaling_method='robust'
                            )
                            
                            if not processed_results:
                                st.error("❌ Data preprocessing failed")
                            else:
                                processed_data = processed_results['final']
                                
                                # Train model
                                model_info = train_model(
                                    processed_data,
                                    selected_param,
                                    selected_model,
                                    optimize=False
                                )
                                
                                # Store model
                                st.session_state.trained_models[selected_param] = {selected_model: model_info}
                                st.session_state.models_trained = True
                                
                                # Generate predictions
                                future_years = list(range(forecast_range[0], forecast_range[1] + 1))
                                predictions = predict_future(model_info, future_years, processed_data)
                                
                                # Store predictions
                                st.session_state.predictions[selected_param] = {selected_model: predictions}
                                st.session_state.predictions_generated = True
                                
                                st.success("✅ Model training and prediction completed!")
                                
                        except Exception as e:
                            st.error(f"❌ Training failed: {e}")
            
            # Display results
            if st.session_state.models_trained and selected_param in st.session_state.trained_models:
                model_info = st.session_state.trained_models[selected_param][selected_model]
                
                st.markdown("### 📊 Model Performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{model_info.get('test_r2', 0):.3f}")
                with col2:
                    st.metric("RMSE", f"{model_info.get('test_rmse', 0):.3f}")
                with col3:
                    st.metric("MAE", f"{model_info.get('test_mae', 0):.3f}")
                
                # Show predictions if available
                if (st.session_state.predictions_generated and 
                    selected_param in st.session_state.predictions):
                    
                    pred_data = st.session_state.predictions[selected_param][selected_model]
                    
                    if not pred_data.empty:
                        st.markdown("### 🔮 Predictions Preview")
                        st.dataframe(pred_data.head(10), use_container_width=True)
                        
                        # Quick prediction summary
                        pred_col = f'{selected_param}_predicted'
                        if pred_col in pred_data.columns:
                            avg_pred = pred_data[pred_col].mean()
                            min_pred = pred_data[pred_col].min()
                            max_pred = pred_data[pred_col].max()
                            
                            st.markdown("#### 📈 Prediction Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average", f"{avg_pred:.2f}")
                            with col2:
                                st.metric("Minimum", f"{min_pred:.2f}")
                            with col3:
                                st.metric("Maximum", f"{max_pred:.2f}")
    
    # Tab 4: Visualizations Panel
    with tab4:
        st.markdown('<h2 class="sub-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
        else:
            data = st.session_state.climate_data
            
            # Chart selection
            chart_type = st.selectbox("Select Visualization", [
                "Line Chart - Historical vs Predicted",
                "Bar Chart - Yearly Rainfall", 
                "Heatmap - Monthly Temperature & Humidity",
                "Scatter Plot - CO₂ vs Temperature",
                "Forecast Curve - Future Predictions",
                "Pie Chart - Seasonal Rainfall Distribution"
            ])
            
            if chart_type == "Line Chart - Historical vs Predicted":
                if st.session_state.predictions_generated:
                    for param, pred_models in st.session_state.predictions.items():
                        for model, pred_data in pred_models.items():
                            if not pred_data.empty:
                                fig = go.Figure()
                                
                                # Historical data
                                if param in data.columns:
                                    fig.add_trace(go.Scatter(
                                        x=data['date'],
                                        y=data[param],
                                        mode='lines',
                                        name=f'Historical {param.title()}',
                                        line=dict(color='blue', width=2)
                                    ))
                                
                                # Predicted data
                                pred_col = f'{param}_predicted'
                                if pred_col in pred_data.columns:
                                    fig.add_trace(go.Scatter(
                                        x=pred_data['date'],
                                        y=pred_data[pred_col],
                                        mode='lines',
                                        name=f'Predicted {param.title()}',
                                        line=dict(color='red', width=2, dash='dash')
                                    ))
                                
                                fig.update_layout(
                                    title=f"{param.title()} - Historical vs Predicted",
                                    xaxis_title="Date",
                                    yaxis_title=param.title(),
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Generate predictions first to see comparison charts")
            
            elif chart_type == "Bar Chart - Yearly Rainfall":
                if 'rainfall' in data.columns and 'year' in data.columns:
                    yearly_rainfall = data.groupby('year')['rainfall'].sum().reset_index()
                    
                    fig = px.bar(
                        yearly_rainfall,
                        x='year',
                        y='rainfall',
                        title="Annual Rainfall Totals",
                        color='rainfall',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Rainfall or year data not available")
            
            elif chart_type == "Heatmap - Monthly Temperature & Humidity":
                if 'temperature' in data.columns and 'humidity' in data.columns:
                    # Create monthly averages
                    monthly_data = data.groupby(['year', 'month']).agg({
                        'temperature': 'mean',
                        'humidity': 'mean'
                    }).reset_index()
                    
                    # Create pivot table for heatmap
                    temp_pivot = monthly_data.pivot(index='year', columns='month', values='temperature')
                    
                    fig = px.imshow(
                        temp_pivot,
                        title="Monthly Average Temperature Heatmap",
                        labels=dict(x="Month", y="Year", color="Temperature (°C)"),
                        color_continuous_scale="RdYlBu_r"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Temperature or humidity data not available")
            
            elif chart_type == "Scatter Plot - CO₂ vs Temperature":
                if 'co2' in data.columns and 'temperature' in data.columns:
                    fig = px.scatter(
                        data,
                        x='co2',
                        y='temperature',
                        title="CO₂ vs Temperature Correlation",
                        trendline="ols",
                        color='year' if 'year' in data.columns else None
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("CO₂ or temperature data not available")
            
            elif chart_type == "Pie Chart - Seasonal Rainfall Distribution":
                if 'rainfall' in data.columns and 'season' in data.columns:
                    seasonal_rainfall = data.groupby('season')['rainfall'].sum().reset_index()
                    
                    fig = px.pie(
                        seasonal_rainfall,
                        values='rainfall',
                        names='season',
                        title="Seasonal Rainfall Distribution"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Rainfall or season data not available")
            
            else:
                st.info("Select a visualization type to display charts")
    
    # Tab 5: AI Insights Section
    with tab5:
        st.markdown('<h2 class="sub-header">🧠 AI-Generated Insights</h2>', unsafe_allow_html=True)
        
        if st.button("🔮 Generate AI Insights", type="primary"):
            if st.session_state.data_loaded:
                with st.spinner("Analyzing climate patterns..."):
                    insights = generate_ai_insights(
                        st.session_state.climate_data,
                        st.session_state.predictions if st.session_state.predictions_generated else None
                    )
                    st.session_state.insights = insights
                    st.success("✅ AI insights generated!")
        
        # Display insights
        if st.session_state.insights:
            st.markdown("### 🎯 Key Climate Insights")
            
            for i, insight in enumerate(st.session_state.insights, 1):
                st.markdown(f"""
                <div class="insight-box">
                    <h4>Insight #{i}</h4>
                    <p>{insight}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional analysis
        if st.session_state.data_loaded and not st.session_state.climate_data.empty:
            data = st.session_state.climate_data
            
            st.markdown("### 📊 Climate Trend Analysis")
            
            # Temperature trend
            if 'temperature' in data.columns and 'year' in data.columns:
                yearly_temp = data.groupby('year')['temperature'].mean()
                temp_trend = np.polyfit(yearly_temp.index, yearly_temp.values, 1)[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🌡️ Temperature Trend", f"{temp_trend:.3f}°C/year",
                             delta="Warming" if temp_trend > 0 else "Cooling")
                
                # Rainfall trend
                if 'rainfall' in data.columns:
                    yearly_rain = data.groupby('year')['rainfall'].sum()
                    rain_trend = np.polyfit(yearly_rain.index, yearly_rain.values, 1)[0]
                    
                    with col2:
                        st.metric("🌧️ Rainfall Trend", f"{rain_trend:.1f}mm/year",
                                 delta="Increasing" if rain_trend > 0 else "Decreasing")
    
    # Tab 6: Climate Risk Index
    with tab6:
        st.markdown('<h2 class="sub-header">⚠️ Climate Risk Assessment</h2>', unsafe_allow_html=True)
        
        if st.button("📊 Calculate Risk Score", type="primary"):
            if st.session_state.data_loaded:
                data = st.session_state.climate_data
                
                # Calculate risk factors
                temp_rise = 0
                aqi_avg = data['aqi'].mean() if 'aqi' in data.columns else 75
                rainfall_change = 0
                
                if 'temperature' in data.columns and 'year' in data.columns:
                    yearly_temp = data.groupby('year')['temperature'].mean()
                    if len(yearly_temp) > 1:
                        temp_rise = (yearly_temp.iloc[-1] - yearly_temp.iloc[0]) / len(yearly_temp) * 10
                
                if 'rainfall' in data.columns and 'year' in data.columns:
                    yearly_rain = data.groupby('year')['rainfall'].sum()
                    if len(yearly_rain) > 1:
                        rainfall_change = ((yearly_rain.iloc[-1] - yearly_rain.iloc[0]) / yearly_rain.iloc[0]) * 100
                
                # Calculate overall risk score
                risk_score = calculate_risk_score(temp_rise, aqi_avg, abs(rainfall_change))
                st.session_state.risk_score = risk_score
                
                st.success("✅ Risk assessment completed!")
        
        # Display risk assessment
        if st.session_state.risk_score > 0:
            risk_score = st.session_state.risk_score
            
            # Risk level determination
            if risk_score < 30:
                risk_level = "Low"
                risk_color = "risk-low"
                risk_emoji = "🟢"
            elif risk_score < 70:
                risk_level = "Moderate" 
                risk_color = "risk-moderate"
                risk_emoji = "🟠"
            else:
                risk_level = "High"
                risk_color = "risk-high"
                risk_emoji = "🔴"
            
            # Display risk indicator
            st.markdown(f"""
            <div class="{risk_color}">
                <h2>{risk_emoji} Climate Risk Level: {risk_level}</h2>
                <h3>Risk Score: {risk_score}/100</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk breakdown
            st.markdown("### 📊 Risk Factor Breakdown")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🌡️ Temperature Risk")
                if st.session_state.data_loaded:
                    data = st.session_state.climate_data
                    if 'temperature' in data.columns:
                        avg_temp = data['temperature'].mean()
                        st.metric("Average Temperature", f"{avg_temp:.1f}°C")
                        if avg_temp > 30:
                            st.warning("High temperature levels detected")
                        else:
                            st.success("Temperature levels normal")
            
            with col2:
                st.markdown("#### 💨 Air Quality Risk")
                if st.session_state.data_loaded:
                    data = st.session_state.climate_data
                    if 'aqi' in data.columns:
                        avg_aqi = data['aqi'].mean()
                        st.metric("Average AQI", f"{avg_aqi:.0f}")
                        if avg_aqi > 100:
                            st.error("Poor air quality")
                        elif avg_aqi > 50:
                            st.warning("Moderate air quality")
                        else:
                            st.success("Good air quality")
            
            with col3:
                st.markdown("#### 🌧️ Rainfall Risk")
                if st.session_state.data_loaded:
                    data = st.session_state.climate_data
                    if 'rainfall' in data.columns:
                        total_rainfall = data['rainfall'].sum()
                        st.metric("Total Rainfall", f"{total_rainfall:.0f}mm")
                        if total_rainfall < 500:
                            st.warning("Low rainfall levels")
                        else:
                            st.success("Adequate rainfall")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            recommendations = []
            if risk_level == "High":
                recommendations = [
                    "🌳 Implement urban forestry programs to reduce heat island effect",
                    "💧 Develop water conservation and rainwater harvesting systems", 
                    "🚗 Promote electric vehicles and reduce emissions",
                    "🏠 Encourage green building practices and energy efficiency",
                    "📢 Establish early warning systems for extreme weather"
                ]
            elif risk_level == "Moderate":
                recommendations = [
                    "🌱 Increase green cover in urban areas",
                    "♻️ Implement waste management and recycling programs",
                    "🚌 Improve public transportation systems",
                    "💡 Promote renewable energy adoption"
                ]
            else:
                recommendations = [
                    "📊 Continue monitoring climate indicators",
                    "🌿 Maintain current environmental protection measures",
                    "📚 Educate community about climate awareness"
                ]
            
            for rec in recommendations:
                st.write(f"• {rec}")
    
    # Tab 7: Download Report
    with tab7:
        st.markdown('<h2 class="sub-header">📄 Climate Report Generation</h2>', unsafe_allow_html=True)
        
        # Report configuration
        st.markdown("### ⚙️ Report Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            include_predictions = st.checkbox("Include Predictions", True)
            include_visualizations = st.checkbox("Include Visualizations", True)
        
        with col2:
            include_risk_assessment = st.checkbox("Include Risk Assessment", True)
            include_recommendations = st.checkbox("Include Recommendations", True)
        
        # Generate report
        if st.button("📄 Generate Climate Report", type="primary"):
            if not st.session_state.data_loaded:
                st.error("❌ Please load data first")
            else:
                with st.spinner("Generating comprehensive climate report..."):
                    try:
                        data = st.session_state.climate_data
                        predictions = st.session_state.predictions if include_predictions else None
                        
                        # Prepare report data
                        report_data = {
                            'city': 'Pune',
                            'data_period': f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
                            'total_records': len(data),
                            'avg_temperature': data['temperature'].mean() if 'temperature' in data.columns else 0,
                            'total_rainfall': data['rainfall'].sum() if 'rainfall' in data.columns else 0,
                            'avg_aqi': data['aqi'].mean() if 'aqi' in data.columns else 0,
                            'risk_score': st.session_state.risk_score,
                            'insights': st.session_state.insights
                        }
                        
                        # Create report content
                        report_content = f"""
# 🌡️ Pune Climate Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary

This comprehensive climate analysis report for Pune, Maharashtra provides insights into historical climate patterns, future predictions, and risk assessments based on {report_data['total_records']:,} data records spanning from {report_data['data_period']}.

## 🏙️ City Overview

- **Location:** Pune, Maharashtra, India
- **Coordinates:** 18.5204°N, 73.8567°E
- **Analysis Period:** {report_data['data_period']}
- **Data Points:** {report_data['total_records']:,} records

## 📈 Key Climate Metrics

- **Average Temperature:** {report_data['avg_temperature']:.1f}°C
- **Total Rainfall:** {report_data['total_rainfall']:.0f}mm
- **Average AQI:** {report_data['avg_aqi']:.0f}
- **Climate Risk Score:** {report_data['risk_score']}/100

## 🧠 AI-Generated Insights

"""
                        
                        for i, insight in enumerate(report_data['insights'], 1):
                            report_content += f"{i}. {insight}\n"
                        
                        if include_predictions and st.session_state.predictions_generated:
                            report_content += "\n## 🔮 Future Predictions\n\n"
                            for param, pred_models in st.session_state.predictions.items():
                                for model, pred_data in pred_models.items():
                                    if not pred_data.empty:
                                        pred_col = f'{param}_predicted'
                                        if pred_col in pred_data.columns:
                                            avg_pred = pred_data[pred_col].mean()
                                            report_content += f"- **{param.title()}:** Average predicted value of {avg_pred:.2f}\n"
                        
                        if include_risk_assessment and st.session_state.risk_score > 0:
                            risk_level = "Low" if st.session_state.risk_score < 30 else "Moderate" if st.session_state.risk_score < 70 else "High"
                            report_content += f"\n## ⚠️ Risk Assessment\n\n"
                            report_content += f"**Climate Risk Level:** {risk_level}\n"
                            report_content += f"**Risk Score:** {st.session_state.risk_score}/100\n"
                        
                        # Display report preview
                        st.markdown("### 👀 Report Preview")
                        st.markdown(report_content)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Report (Markdown)",
                            data=report_content,
                            file_name=f"pune_climate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                        
                        st.success("✅ Report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {e}")
        
        # Report statistics
        if st.session_state.data_loaded:
            st.markdown("### 📊 Report Statistics")
            
            data = st.session_state.climate_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Records", f"{len(data):,}")
            with col2:
                st.metric("Variables", len(data.columns))
            with col3:
                years_covered = data['date'].max().year - data['date'].min().year + 1 if 'date' in data.columns else 0
                st.metric("Years Covered", years_covered)

if __name__ == "__main__":
    main()