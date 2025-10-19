"""
Enhanced Pune Climate Dashboard
Comprehensive climate analysis with AI recommendations, chatbot, and real-time data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import os
import sys
from datetime import datetime, timedelta
import yaml
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
try:
    from data_collector import fetch_city_data, load_csv_data
    from data_preprocessor import clean_and_preprocess
    from model_trainer import train_model
    from predictor import predict_future
    from evaluator import evaluate_model
    from report_generator import generate_report
    from visualization import ClimateVisualizationEngine, create_all_visualizations
    from api_client import get_current_weather
    from ai_recommendations import generate_ai_recommendations
    from climate_chatbot import create_climate_chatbot, chat_with_bot
except ImportError as e:
    st.error(f"Backend import error: {e}")

# Page configuration
st.set_page_config(
    page_title="🌡️ Enhanced Pune Climate Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from YAML file"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            'app': {'name': 'Pune Climate Dashboard', 'version': '1.0.0'},
            'features': {
                'real_time_data': True,
                'ai_recommendations': True,
                'chatbot': True,
                'advanced_analytics': True
            }
        }

config = load_config()

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .risk-indicator {
        font-size: 2rem;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-low { background: linear-gradient(135deg, #4CAF50, #45a049); }
    .risk-moderate { background: linear-gradient(135deg, #FF9800, #F57C00); }
    .risk-high { background: linear-gradient(135deg, #F44336, #D32F2F); }
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
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
        'insights': [],
        'recommendations': {},
        'chatbot': None,
        'chat_history': [],
        'real_time_data': {},
        'current_tab': 'Home'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Utility functions
@st.cache_data
def load_climate_data(start_year: int, end_year: int, include_current: bool = True, csv_path: str = None):
    """Load and cache climate data"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(
            fetch_city_data("Pune", start_year, end_year, include_current, csv_path)
        )
        loop.close()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_real_time_weather():
    """Get real-time weather data"""
    try:
        if config['features']['real_time_data']:
            return get_current_weather("Pune")
        else:
            # Fallback to simulated data
            return {
                'temperature': 28.5 + np.random.normal(0, 2),
                'humidity': 65 + np.random.normal(0, 10),
                'aqi': 85 + np.random.normal(0, 15),
                'wind_speed': 12 + np.random.normal(0, 3),
                'pressure': 1013 + np.random.normal(0, 5),
                'timestamp': datetime.now()
            }
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return {}

def calculate_enhanced_risk_score(data, predictions=None):
    """Calculate enhanced climate risk score"""
    risk_score = 0
    risk_factors = []
    
    if not data.empty:
        # Temperature risk (0-35 points)
        if 'temperature' in data.columns:
            temp_mean = data['temperature'].mean()
            if temp_mean > 35:
                risk_score += 35
                risk_factors.append("Extreme temperature levels")
            elif temp_mean > 32:
                risk_score += 25
                risk_factors.append("High temperature levels")
            elif temp_mean > 28:
                risk_score += 15
                risk_factors.append("Elevated temperature levels")
        
        # AQI risk (0-30 points)
        if 'aqi' in data.columns:
            aqi_mean = data['aqi'].mean()
            if aqi_mean > 150:
                risk_score += 30
                risk_factors.append("Severe air pollution")
            elif aqi_mean > 100:
                risk_score += 20
                risk_factors.append("Poor air quality")
            elif aqi_mean > 75:
                risk_score += 10
                risk_factors.append("Moderate air quality concerns")
        
        # Rainfall variability risk (0-25 points)
        if 'rainfall' in data.columns and 'year' in data.columns:
            yearly_rainfall = data.groupby('year')['rainfall'].sum()
            if len(yearly_rainfall) > 1:
                rainfall_cv = yearly_rainfall.std() / yearly_rainfall.mean()
                if rainfall_cv > 0.4:
                    risk_score += 25
                    risk_factors.append("High rainfall variability")
                elif rainfall_cv > 0.3:
                    risk_score += 15
                    risk_factors.append("Moderate rainfall variability")
        
        # Climate change trend risk (0-10 points)
        if 'temperature' in data.columns and 'year' in data.columns:
            yearly_temp = data.groupby('year')['temperature'].mean()
            if len(yearly_temp) > 1:
                temp_trend = np.polyfit(yearly_temp.index, yearly_temp.values, 1)[0]
                if temp_trend > 0.2:
                    risk_score += 10
                    risk_factors.append("Rapid warming trend")
                elif temp_trend > 0.1:
                    risk_score += 5
                    risk_factors.append("Warming trend detected")
    
    # Future predictions risk
    if predictions:
        for var, pred_data in predictions.items():
            if not pred_data.empty and f'{var}_predicted' in pred_data.columns:
                if var == 'temperature':
                    future_avg = pred_data[f'{var}_predicted'].mean()
                    current_avg = data[var].mean() if var in data.columns else 25
                    temp_increase = future_avg - current_avg
                    
                    if temp_increase > 3:
                        risk_score += 10
                        risk_factors.append("Severe future warming predicted")
                    elif temp_increase > 2:
                        risk_score += 5
                        risk_factors.append("Significant future warming predicted")
    
    return min(risk_score, 100), risk_factors

def main():
    """Main enhanced dashboard application"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌡️ Enhanced Pune Climate Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Advanced Climate Intelligence Platform v{config["app"]["version"]}</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("🎛️ Dashboard Controls")
        
        # Real-time data toggle
        if config['features']['real_time_data']:
            if st.button("🔄 Refresh Real-time Data"):
                st.cache_data.clear()
                st.rerun()
        
        # Data configuration
        st.header("📊 Data Settings")
        start_year = st.slider("Start Year", 2000, 2023, 2020)
        end_year = st.slider("End Year", 2021, 2024, 2024)
        
        # File upload
        uploaded_file = st.file_uploader("Upload Climate Data (CSV)", type=['csv'])
        if uploaded_file:
            st.session_state.uploaded_data = uploaded_file
        
        # Model configuration
        st.header("🤖 Model Settings")
        model_type = st.selectbox("Model Type", ["Linear Regression", "Random Forest", "Prophet"])
        target_param = st.selectbox("Parameter", ["Temperature", "Rainfall"])
        forecast_range = st.slider("Forecast Range", 2026, 2050, (2026, 2035))
        
        # Feature toggles
        st.header("🔧 Features")
        enable_ai_recommendations = st.checkbox("AI Recommendations", config['features']['ai_recommendations'])
        enable_chatbot = st.checkbox("Climate Chatbot", config['features']['chatbot'])
        enable_advanced_analytics = st.checkbox("Advanced Analytics", config['features']['advanced_analytics'])
    
    # Main content tabs
    tabs = st.tabs([
        "🏠 Home", "📊 Data Explorer", "🤖 Model Training", 
        "📈 Visualizations", "🧠 AI Insights", "⚠️ Risk Assessment", 
        "💬 Climate Chat", "📄 Reports"
    ])
    
    # Tab 1: Enhanced Home Page
    with tabs[0]:
        st.markdown("## 🏠 Pune Climate Intelligence Center")
        
        # Real-time weather display
        if config['features']['real_time_data']:
            st.markdown("### 🌤️ Live Weather Conditions")
            
            real_time_data = get_real_time_weather()
            if real_time_data:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🌡️ Temperature</h3>
                        <h2>{real_time_data.get('temperature', 0):.1f}°C</h2>
                        <p>Real-time</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💧 Humidity</h3>
                        <h2>{real_time_data.get('humidity', 0):.0f}%</h2>
                        <p>Current level</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💨 AQI</h3>
                        <h2>{real_time_data.get('aqi', 0):.0f}</h2>
                        <p>Air Quality</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🌪️ Wind</h3>
                        <h2>{real_time_data.get('wind_speed', 0):.1f}</h2>
                        <p>km/h</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.session_state.real_time_data = real_time_data
        
        # Data loading section
        st.markdown("### 📊 Climate Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Load Historical Data", type="primary"):
                with st.spinner("Loading comprehensive climate data..."):
                    csv_path = None
                    if st.session_state.uploaded_data:
                        # Save uploaded file temporarily
                        csv_path = f"temp_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        with open(csv_path, 'wb') as f:
                            f.write(st.session_state.uploaded_data.getvalue())
                    
                    data = load_climate_data(start_year, end_year, True, csv_path)
                    
                    # Clean up temp file
                    if csv_path and os.path.exists(csv_path):
                        os.remove(csv_path)
                    
                    if not data.empty:
                        st.session_state.climate_data = data
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded {len(data):,} climate records!")
                    else:
                        st.error("❌ Failed to load data")
        
        with col2:
            if st.button("🔮 Quick Analysis", type="secondary"):
                if st.session_state.data_loaded:
                    st.info("🔄 Running quick climate analysis...")
                    # Trigger quick analysis in other tabs
                    st.session_state.current_tab = 'Quick Analysis'
                else:
                    st.warning("⚠️ Please load data first")
        
        # Historical overview
        if st.session_state.data_loaded and not st.session_state.climate_data.empty:
            data = st.session_state.climate_data
            
            st.markdown("### 📈 Historical Climate Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_temp = data['temperature'].mean() if 'temperature' in data.columns else 0
                temp_trend = "📈" if avg_temp > 27 else "📉" if avg_temp < 25 else "➡️"
                st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C", delta=f"{temp_trend} Trend")
            
            with col2:
                total_rainfall = data['rainfall'].sum() if 'rainfall' in data.columns else 0
                st.metric("🌧️ Total Rainfall", f"{total_rainfall:.0f}mm", 
                         delta=f"{(total_rainfall/len(data)*365) - 700:.0f}mm vs normal")
            
            with col3:
                avg_humidity = data['humidity'].mean() if 'humidity' in data.columns else 0
                st.metric("💧 Avg Humidity", f"{avg_humidity:.0f}%",
                         delta=f"{avg_humidity - 65:.0f}% vs normal")
            
            with col4:
                avg_aqi = data['aqi'].mean() if 'aqi' in data.columns else 0
                aqi_status = "Good" if avg_aqi < 50 else "Moderate" if avg_aqi < 100 else "Poor"
                st.metric("💨 Air Quality", f"{avg_aqi:.0f} AQI", delta=aqi_status)
            
            # Quick visualization
            if 'temperature' in data.columns and 'date' in data.columns:
                st.markdown("### 📊 Temperature Trend")
                
                fig = px.line(data.tail(365), x='date', y='temperature', 
                             title="Temperature Trend (Last 365 Days)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Enhanced Data Explorer
    with tabs[1]:
        st.markdown("## 📊 Advanced Data Explorer")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first in the Home tab")
        else:
            data = st.session_state.climate_data
            
            # Data overview
            st.markdown("### 📋 Dataset Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(data):,}")
            with col2:
                st.metric("Variables", len(data.columns))
            with col3:
                years_span = data['date'].max().year - data['date'].min().year + 1 if 'date' in data.columns else 0
                st.metric("Years Covered", years_span)
            
            # Interactive data table
            st.markdown("### 🔍 Interactive Data Table")
            
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
            
            # Advanced analytics
            if enable_advanced_analytics:
                st.markdown("### 🔬 Advanced Analytics")
                
                # Correlation analysis
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
                
                # Statistical summary
                st.markdown("### 📊 Statistical Summary")
                st.dataframe(data[climate_vars].describe(), use_container_width=True)
    
    # Tab 3: Enhanced Model Training
    with tabs[2]:
        st.markdown("## 🤖 Advanced Model Training")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
        else:
            data = st.session_state.climate_data
            
            # Model configuration
            st.markdown("### ⚙️ Model Configuration")
            
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
                
                st.info(f"**Configuration:**\n- Model: {model_type}\n- Parameter: {target_param}\n- Forecast: {forecast_range[0]}-{forecast_range[1]}")
            
            with col2:
                # Advanced options
                optimize_hyperparams = st.checkbox("🔧 Optimize Hyperparameters")
                cross_validation = st.checkbox("📊 Cross Validation")
                ensemble_models = st.checkbox("🎯 Ensemble Multiple Models")
            
            # Training section
            if st.button("🚀 Train Advanced Models", type="primary"):
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
                                
                                # Train model(s)
                                if ensemble_models:
                                    # Train multiple models
                                    models_to_train = ['linear', 'random_forest', 'prophet']
                                    trained_models = {}
                                    
                                    progress_bar = st.progress(0)
                                    for i, model in enumerate(models_to_train):
                                        st.info(f"Training {model} model...")
                                        model_info = train_model(
                                            processed_data,
                                            selected_param,
                                            model,
                                            optimize=optimize_hyperparams
                                        )
                                        trained_models[model] = model_info
                                        progress_bar.progress((i + 1) / len(models_to_train))
                                    
                                    st.session_state.trained_models[selected_param] = trained_models
                                else:
                                    # Train single model
                                    model_info = train_model(
                                        processed_data,
                                        selected_param,
                                        selected_model,
                                        optimize=optimize_hyperparams
                                    )
                                    
                                    st.session_state.trained_models[selected_param] = {selected_model: model_info}
                                
                                st.session_state.models_trained = True
                                
                                # Generate predictions
                                future_years = list(range(forecast_range[0], forecast_range[1] + 1))
                                predictions = {}
                                
                                for model_name, model_info in st.session_state.trained_models[selected_param].items():
                                    pred_data = predict_future(model_info, future_years, processed_data)
                                    predictions[model_name] = pred_data
                                
                                st.session_state.predictions[selected_param] = predictions
                                st.session_state.predictions_generated = True
                                
                                st.success("✅ Advanced model training completed!")
                                
                        except Exception as e:
                            st.error(f"❌ Training failed: {e}")
            
            # Display results
            if st.session_state.models_trained and selected_param in st.session_state.trained_models:
                st.markdown("### 📊 Model Performance Comparison")
                
                models = st.session_state.trained_models[selected_param]
                
                # Create performance comparison table
                performance_data = []
                for model_name, model_info in models.items():
                    performance_data.append({
                        'Model': model_name.title(),
                        'R² Score': f"{model_info.get('test_r2', 0):.3f}",
                        'RMSE': f"{model_info.get('test_rmse', 0):.3f}",
                        'MAE': f"{model_info.get('test_mae', 0):.3f}",
                        'Training Time': f"{model_info.get('training_time', 0):.1f}s"
                    })
                
                if performance_data:
                    performance_df = pd.DataFrame(performance_data)
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # Best model recommendation
                    best_model = max(models.items(), key=lambda x: x[1].get('test_r2', 0))
                    st.success(f"🏆 Best performing model: **{best_model[0].title()}** (R² = {best_model[1].get('test_r2', 0):.3f})")
    
    # Tab 4: Enhanced Visualizations
    with tabs[3]:
        st.markdown("## 📈 Advanced Climate Visualizations")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
        else:
            data = st.session_state.climate_data
            
            # Visualization controls
            st.markdown("### 🎨 Visualization Controls")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                chart_type = st.selectbox("Chart Type", [
                    "Climate Overview",
                    "Temperature Analysis",
                    "Rainfall Patterns", 
                    "Air Quality Trends",
                    "Correlation Analysis",
                    "Seasonal Patterns",
                    "Future Predictions"
                ])
            
            with col2:
                time_period = st.selectbox("Time Period", [
                    "All Data",
                    "Last 5 Years",
                    "Last 2 Years",
                    "Last Year"
                ])
            
            with col3:
                chart_style = st.selectbox("Style", [
                    "Interactive",
                    "Static",
                    "Animated"
                ])
            
            # Filter data based on time period
            if time_period != "All Data":
                years_back = {"Last Year": 1, "Last 2 Years": 2, "Last 5 Years": 5}[time_period]
                cutoff_date = datetime.now() - timedelta(days=years_back*365)
                if 'date' in data.columns:
                    data = data[data['date'] >= cutoff_date]
            
            # Generate visualizations
            if chart_type == "Climate Overview":
                st.markdown("### 🌍 Climate Variables Overview")
                
                # Create comprehensive overview
                visualizer = ClimateVisualizationEngine()
                overview_chart = visualizer.create_climate_summary_chart(data)
                st.plotly_chart(overview_chart, use_container_width=True)
            
            elif chart_type == "Temperature Analysis":
                if 'temperature' in data.columns:
                    st.markdown("### 🌡️ Temperature Analysis")
                    
                    # Temperature trends
                    fig = px.line(data, x='date', y='temperature', 
                                 title="Temperature Trends Over Time")
                    
                    # Add moving average
                    if len(data) > 30:
                        data['temp_ma'] = data['temperature'].rolling(window=30).mean()
                        fig.add_scatter(x=data['date'], y=data['temp_ma'], 
                                      mode='lines', name='30-day Moving Average')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Temperature distribution
                    fig_hist = px.histogram(data, x='temperature', nbins=50,
                                          title="Temperature Distribution")
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            elif chart_type == "Future Predictions":
                if st.session_state.predictions_generated:
                    st.markdown("### 🔮 Future Climate Predictions")
                    
                    for param, pred_models in st.session_state.predictions.items():
                        st.markdown(f"#### {param.title()} Predictions")
                        
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
                        
                        # Predictions from different models
                        colors = ['red', 'green', 'orange', 'purple']
                        for i, (model_name, pred_data) in enumerate(pred_models.items()):
                            if not pred_data.empty:
                                pred_col = f'{param}_predicted'
                                if pred_col in pred_data.columns:
                                    fig.add_trace(go.Scatter(
                                        x=pred_data['date'],
                                        y=pred_data[pred_col],
                                        mode='lines',
                                        name=f'{model_name.title()} Prediction',
                                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                                    ))
                        
                        fig.update_layout(
                            title=f"{param.title()} - Historical vs Predicted",
                            xaxis_title="Date",
                            yaxis_title=param.title(),
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Generate predictions first in the Model Training tab")
    
    # Tab 5: AI Insights & Recommendations
    with tabs[4]:
        st.markdown("## 🧠 AI-Powered Climate Insights")
        
        if enable_ai_recommendations and st.session_state.data_loaded:
            data = st.session_state.climate_data
            
            if st.button("🔮 Generate AI Insights & Recommendations", type="primary"):
                with st.spinner("Analyzing climate patterns and generating recommendations..."):
                    # Calculate risk score
                    risk_score, risk_factors = calculate_enhanced_risk_score(
                        data, st.session_state.predictions if st.session_state.predictions_generated else None
                    )
                    st.session_state.risk_score = risk_score
                    
                    # Generate AI recommendations
                    recommendations = generate_ai_recommendations(
                        data, 
                        st.session_state.predictions if st.session_state.predictions_generated else None,
                        risk_score
                    )
                    st.session_state.recommendations = recommendations
                    
                    st.success("✅ AI analysis completed!")
            
            # Display recommendations
            if st.session_state.recommendations:
                recommendations = st.session_state.recommendations
                
                st.markdown("### 🚨 Priority Actions")
                for i, action in enumerate(recommendations['priority_actions'][:5], 1):
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{i}. {action}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Intervention impacts
                if recommendations['intervention_impacts']:
                    st.markdown("### 💡 Intervention Impact Analysis")
                    
                    for intervention, impact in recommendations['intervention_impacts'].items():
                        with st.expander(f"🎯 {intervention.replace('_', ' ').title()}"):
                            st.write(impact['description'])
                            
                            # Create impact visualization
                            if 'temperature_reduction_celsius' in impact:
                                st.metric("Temperature Reduction", f"{impact['temperature_reduction_celsius']}°C")
                            if 'co2_absorption_kg_per_year' in impact:
                                st.metric("CO₂ Absorption", f"{impact['co2_absorption_kg_per_year']:,.0f} kg/year")
                
                # Implementation timeline
                st.markdown("### 📅 Implementation Timeline")
                timeline = recommendations['implementation_timeline']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Immediate (0-6 months):**")
                    for action in timeline.get('immediate_0_6_months', [])[:3]:
                        st.write(f"• {action}")
                
                with col2:
                    st.markdown("**Short-term (6-18 months):**")
                    for action in timeline.get('short_term_6_18_months', [])[:3]:
                        st.write(f"• {action}")
        else:
            st.info("Enable AI Recommendations in the sidebar to access this feature")
    
    # Tab 6: Enhanced Risk Assessment
    with tabs[5]:
        st.markdown("## ⚠️ Comprehensive Climate Risk Assessment")
        
        if st.session_state.data_loaded:
            data = st.session_state.climate_data
            
            if st.button("📊 Calculate Enhanced Risk Score", type="primary"):
                risk_score, risk_factors = calculate_enhanced_risk_score(
                    data, st.session_state.predictions if st.session_state.predictions_generated else None
                )
                st.session_state.risk_score = risk_score
                st.session_state.risk_factors = risk_factors
            
            # Display risk assessment
            if st.session_state.risk_score > 0:
                risk_score = st.session_state.risk_score
                
                # Risk level determination
                if risk_score < 30:
                    risk_level = "Low"
                    risk_class = "risk-low"
                    risk_emoji = "🟢"
                    risk_color = "#4CAF50"
                elif risk_score < 70:
                    risk_level = "Moderate" 
                    risk_class = "risk-moderate"
                    risk_emoji = "🟠"
                    risk_color = "#FF9800"
                else:
                    risk_level = "High"
                    risk_class = "risk-high"
                    risk_emoji = "🔴"
                    risk_color = "#F44336"
                
                # Risk indicator
                st.markdown(f"""
                <div class="risk-indicator {risk_class}">
                    <h2>{risk_emoji} Climate Risk Level: {risk_level}</h2>
                    <h3>Risk Score: {risk_score}/100</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk breakdown visualization
                st.markdown("### 📊 Risk Factor Analysis")
                
                # Create risk factor chart
                if hasattr(st.session_state, 'risk_factors'):
                    risk_factors = st.session_state.risk_factors
                    
                    if risk_factors:
                        fig = go.Figure(data=[
                            go.Bar(x=risk_factors, y=[1]*len(risk_factors), 
                                  marker_color=risk_color)
                        ])
                        fig.update_layout(
                            title="Identified Risk Factors",
                            xaxis_title="Risk Factors",
                            yaxis_title="Presence",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("✅ No significant risk factors identified")
                
                # Risk mitigation strategies
                st.markdown("### 🛡️ Risk Mitigation Strategies")
                
                mitigation_strategies = {
                    "High": [
                        "🚨 Immediate action required - Implement emergency climate protocols",
                        "🌳 Urgent urban forestry program - Plant 10,000+ trees",
                        "💧 Emergency water conservation measures",
                        "🏥 Establish climate health monitoring systems",
                        "📢 Public awareness campaigns about climate risks"
                    ],
                    "Moderate": [
                        "⚠️ Proactive measures needed - Develop climate adaptation plan",
                        "🌱 Increase green infrastructure by 25%",
                        "🚌 Improve public transportation systems",
                        "🏢 Implement green building standards",
                        "📊 Enhanced climate monitoring network"
                    ],
                    "Low": [
                        "✅ Maintain current measures - Continue monitoring",
                        "🌿 Preserve existing green spaces",
                        "📈 Regular climate trend analysis",
                        "🎓 Community education programs",
                        "🔄 Periodic risk reassessment"
                    ]
                }
                
                for strategy in mitigation_strategies[risk_level]:
                    st.write(f"• {strategy}")
        else:
            st.warning("⚠️ Please load climate data first")
    
    # Tab 7: Climate Chatbot
    with tabs[6]:
        st.markdown("## 💬 Climate Intelligence Chatbot")
        
        if enable_chatbot:
            # Initialize chatbot
            if st.session_state.chatbot is None:
                st.session_state.chatbot = create_climate_chatbot(
                    st.session_state.climate_data if st.session_state.data_loaded else None
                )
            
            # Chat interface
            st.markdown("### 🤖 Ask me anything about Pune's climate!")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 💬 Conversation History")
                for i, exchange in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 exchanges
                    st.markdown(f"""
                    <div class="chat-container">
                        <strong>👤 You:</strong> {exchange['user']}<br>
                        <strong>🤖 Climate Bot:</strong> {exchange['bot']}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chat input
            user_message = st.text_input("Type your climate question here:", 
                                       placeholder="e.g., What's the temperature trend in Pune?")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("💬 Send Message"):
                    if user_message:
                        bot_response = chat_with_bot(st.session_state.chatbot, user_message)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'user': user_message,
                            'bot': bot_response,
                            'timestamp': datetime.now()
                        })
                        
                        st.rerun()
            
            with col2:
                if st.button("🔄 Clear Chat"):
                    st.session_state.chat_history = []
                    if st.session_state.chatbot:
                        st.session_state.chatbot.clear_history()
                    st.rerun()
            
            with col3:
                if st.button("📊 Get Climate Summary"):
                    if st.session_state.chatbot:
                        summary = st.session_state.chatbot.get_climate_summary()
                        st.info(summary)
            
            # Sample questions
            st.markdown("### 💡 Sample Questions")
            sample_questions = [
                "What's the average temperature in Pune?",
                "How is climate change affecting Pune?",
                "What are the future temperature predictions?",
                "How can we improve air quality in Pune?",
                "What climate adaptation strategies do you recommend?"
            ]
            
            for question in sample_questions:
                if st.button(f"❓ {question}", key=f"sample_{question}"):
                    bot_response = chat_with_bot(st.session_state.chatbot, question)
                    st.session_state.chat_history.append({
                        'user': question,
                        'bot': bot_response,
                        'timestamp': datetime.now()
                    })
                    st.rerun()
        else:
            st.info("Enable Climate Chatbot in the sidebar to access this feature")
    
    # Tab 8: Enhanced Reports
    with tabs[7]:
        st.markdown("## 📄 Comprehensive Climate Reports")
        
        if st.session_state.data_loaded:
            data = st.session_state.climate_data
            
            # Report configuration
            st.markdown("### ⚙️ Report Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                include_predictions = st.checkbox("Include Predictions", True)
                include_visualizations = st.checkbox("Include Visualizations", True)
                include_ai_insights = st.checkbox("Include AI Insights", True)
            
            with col2:
                include_risk_assessment = st.checkbox("Include Risk Assessment", True)
                include_recommendations = st.checkbox("Include Recommendations", True)
                report_format = st.selectbox("Report Format", ["PDF", "HTML"])
            
            # Generate comprehensive report
            if st.button("📄 Generate Enhanced Climate Report", type="primary"):
                with st.spinner("Generating comprehensive climate intelligence report..."):
                    try:
                        # Prepare all data for report
                        predictions = st.session_state.predictions if include_predictions else None
                        recommendations = st.session_state.recommendations if include_ai_insights else None
                        risk_score = st.session_state.risk_score if include_risk_assessment else None
                        
                        # Generate report
                        report_path = generate_report(
                            data=data,
                            predictions=predictions,
                            model_results=None,
                            visuals=None,
                            output_path=f"enhanced_climate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        )
                        
                        if os.path.exists(report_path):
                            file_size = os.path.getsize(report_path)
                            st.success(f"✅ Enhanced report generated successfully!")
                            
                            # Download button
                            with open(report_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Enhanced Report",
                                    data=file.read(),
                                    file_name=os.path.basename(report_path),
                                    mime="application/pdf"
                                )
                            
                            st.info(f"📊 Report size: {file_size:,} bytes")
                        else:
                            st.error("❌ Report generation failed")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating report: {e}")
            
            # Report preview
            st.markdown("### 👀 Report Preview")
            
            if st.session_state.data_loaded:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Data Records", f"{len(data):,}")
                    st.metric("📅 Analysis Period", f"{data['date'].max().year - data['date'].min().year + 1} years")
                
                with col2:
                    models_count = sum(len(models) for models in st.session_state.trained_models.values()) if st.session_state.models_trained else 0
                    st.metric("🤖 Models Trained", models_count)
                    st.metric("🔮 Predictions", "Available" if st.session_state.predictions_generated else "None")
                
                with col3:
                    st.metric("⚠️ Risk Score", f"{st.session_state.risk_score}/100")
                    recommendations_count = len(st.session_state.recommendations.get('priority_actions', [])) if st.session_state.recommendations else 0
                    st.metric("💡 Recommendations", recommendations_count)
        else:
            st.warning("⚠️ Please load climate data first")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌍 Enhanced Pune Climate Dashboard v{config["app"]["version"]} | Powered by Advanced AI & Machine Learning</p>
        <p>📊 Real-time Climate Intelligence • 🤖 AI Recommendations • 💬 Interactive Chatbot</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()