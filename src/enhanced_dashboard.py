"""
Enhanced Real-time Dashboard for Climate and AQI Monitoring and Prediction
Beautiful Streamlit dashboard with advanced visualizations and model accuracy metrics
"""

import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from enhanced_ml_models import EnhancedMLModels
from config import DATABASE_CONFIG, PUNE_LOCATIONS
from realtime_data_collector import RealtimeDataCollector

# Page configuration
st.set_page_config(
    page_title="🌍 Pune Climate & AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .accuracy-badge {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50, #3498DB);
    }
</style>
""", unsafe_allow_html=True)


class EnhancedDashboard:
    """Enhanced dashboard with beautiful visualizations and model accuracy display"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG["sqlite_path"]
        self.ml_models = EnhancedMLModels()
        self.realtime_collector = RealtimeDataCollector()
        
        # Initialize session state
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'model_performance' not in st.session_state:
            st.session_state.model_performance = {}
    
    def load_recent_data(self, hours: int = 24) -> Dict[str, pd.DataFrame]:
        """Load recent data from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Real-time weather data
        weather_query = f"""
            SELECT * FROM weather_historical 
            WHERE timestamp > datetime('now', '-{hours} hours')
            ORDER BY timestamp DESC
        """
        weather_df = pd.read_sql_query(weather_query, conn)
        
        # Real-time air quality data
        aqi_query = f"""
            SELECT * FROM air_quality_historical 
            WHERE timestamp > datetime('now', '-{hours} hours')
            ORDER BY timestamp DESC
        """
        aqi_df = pd.read_sql_query(aqi_query, conn)
        
        conn.close()
        
        # Convert timestamps
        for df in [weather_df, aqi_df]:
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return {
            "weather": weather_df,
            "aqi": aqi_df,
        }
    
    def create_beautiful_metrics(self, data: Dict[str, pd.DataFrame]):
        """Create beautiful metric cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not data["weather"].empty:
                avg_temp = data["weather"]["temperature"].mean()
                temp_change = data["weather"]["temperature"].diff().iloc[-1] if len(data["weather"]) > 1 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌡️ Temperature</h3>
                    <h2>{avg_temp:.1f}°C</h2>
                    <p>{'↗️' if temp_change > 0 else '↘️'} {abs(temp_change):.1f}°C</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h3>🌡️ Temperature</h3><h2>N/A</h2></div>', unsafe_allow_html=True)
        
        with col2:
            if not data["weather"].empty:
                avg_humidity = data["weather"]["humidity"].mean()
                humidity_change = data["weather"]["humidity"].diff().iloc[-1] if len(data["weather"]) > 1 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💧 Humidity</h3>
                    <h2>{avg_humidity:.1f}%</h2>
                    <p>{'↗️' if humidity_change > 0 else '↘️'} {abs(humidity_change):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h3>💧 Humidity</h3><h2>N/A</h2></div>', unsafe_allow_html=True)
        
        with col3:
            if not data["aqi"].empty:
                avg_aqi = data["aqi"]["aqi"].mean()
                aqi_status = self.get_aqi_status(avg_aqi)
                aqi_color = self.get_aqi_color(avg_aqi)
                st.markdown(f"""
                <div class="metric-card" style="background: {aqi_color};">
                    <h3>🏭 Air Quality</h3>
                    <h2>{avg_aqi:.0f} AQI</h2>
                    <p>{aqi_status}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h3>🏭 Air Quality</h3><h2>N/A</h2></div>', unsafe_allow_html=True)
        
        with col4:
            if not data["aqi"].empty:
                avg_pm25 = data["aqi"]["pm25"].mean()
                pm25_change = data["aqi"]["pm25"].diff().iloc[-1] if len(data["aqi"]) > 1 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌫️ PM2.5</h3>
                    <h2>{avg_pm25:.1f} µg/m³</h2>
                    <p>{'↗️' if pm25_change > 0 else '↘️'} {abs(pm25_change):.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h3>🌫️ PM2.5</h3><h2>N/A</h2></div>', unsafe_allow_html=True)
    
    def get_aqi_status(self, aqi: float) -> str:
        """Get AQI status based on value"""
        if aqi <= 50:
            return "Good 😊"
        elif aqi <= 100:
            return "Moderate 😐"
        elif aqi <= 150:
            return "Unhealthy for Sensitive 😷"
        elif aqi <= 200:
            return "Unhealthy 😨"
        elif aqi <= 300:
            return "Very Unhealthy 🚨"
        else:
            return "Hazardous ☠️"
    
    def get_aqi_color(self, aqi: float) -> str:
        """Get color based on AQI value"""
        if aqi <= 50:
            return "linear-gradient(135deg, #4CAF50, #8BC34A)"
        elif aqi <= 100:
            return "linear-gradient(135deg, #FFEB3B, #FFC107)"
        elif aqi <= 150:
            return "linear-gradient(135deg, #FF9800, #FF5722)"
        elif aqi <= 200:
            return "linear-gradient(135deg, #F44336, #E91E63)"
        elif aqi <= 300:
            return "linear-gradient(135deg, #9C27B0, #673AB7)"
        else:
            return "linear-gradient(135deg, #795548, #424242)"
    
    def create_interactive_map(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create beautiful interactive map"""
        fig = go.Figure()
        
        # Add location markers
        for loc_id, loc_config in PUNE_LOCATIONS.items():
            # Get latest data for this location
            weather_data = data["weather"][data["weather"]["location_id"] == loc_id]
            aqi_data = data["aqi"][data["aqi"]["location_id"] == loc_id]
            
            temp = weather_data["temperature"].iloc[-1] if not weather_data.empty else "N/A"
            aqi = aqi_data["aqi"].iloc[-1] if not aqi_data.empty else "N/A"
            
            # Color based on AQI
            if isinstance(aqi, (int, float)):
                color = "green" if aqi <= 50 else "yellow" if aqi <= 100 else "orange" if aqi <= 150 else "red"
                size = min(max(aqi / 5, 10), 30)
            else:
                color = "gray"
                size = 15
            
            fig.add_trace(go.Scattermapbox(
                lat=[loc_config.lat],
                lon=[loc_config.lon],
                mode='markers',
                marker=dict(size=size, color=color),
                text=f"<b>{loc_config.name}</b><br>Temp: {temp}°C<br>AQI: {aqi}",
                hovertemplate="%{text}<extra></extra>",
                name=loc_config.name
            ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=18.5204, lon=73.8567),
                zoom=10
            ),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        return fig
    
    def create_time_series_chart(self, data: Dict[str, pd.DataFrame], variable: str) -> go.Figure:
        """Create beautiful time series chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{variable.title()} Trends', 'Location Comparison'),
            vertical_spacing=0.1
        )
        
        # Overall trend
        if variable in ['temperature', 'humidity', 'pressure', 'wind_speed'] and not data["weather"].empty:
            df = data["weather"]
        elif variable in ['aqi', 'pm25', 'pm10', 'co', 'no2', 'so2'] and not data["aqi"].empty:
            df = data["aqi"]
        else:
            return go.Figure()
        
        if variable in df.columns:
            # Time series for all locations
            for loc_id in df['location_id'].unique():
                loc_data = df[df['location_id'] == loc_id].sort_values('timestamp')
                loc_name = PUNE_LOCATIONS.get(loc_id, type('obj', (object,), {'name': loc_id})).name
                
                fig.add_trace(
                    go.Scatter(
                        x=loc_data['timestamp'],
                        y=loc_data[variable],
                        mode='lines+markers',
                        name=loc_name,
                        line=dict(width=2),
                        marker=dict(size=4)
                    ),
                    row=1, col=1
                )
            
            # Box plot for location comparison
            fig.add_trace(
                go.Box(
                    x=[PUNE_LOCATIONS.get(loc, type('obj', (object,), {'name': loc})).name 
                       for loc in df['location_id']],
                    y=df[variable],
                    name='Distribution',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            title_text=f"{variable.title()} Analysis",
            showlegend=True
        )
        
        return fig
    
    def display_model_accuracy(self):
        """Display model accuracy metrics beautifully"""
        st.markdown("## 🎯 Model Accuracy & Performance")
        
        if st.session_state.model_performance:
            for target_var, performances in st.session_state.model_performance.items():
                st.markdown(f"### {target_var.title()} Prediction Models")
                
                # Create accuracy badges
                accuracy_html = ""
                for model_name, metrics in performances.items():
                    accuracy = metrics.get('accuracy', 0)
                    accuracy_html += f'<span class="accuracy-badge">{model_name.upper()}: {accuracy:.1f}%</span>'
                
                st.markdown(accuracy_html, unsafe_allow_html=True)
                
                # Detailed metrics table
                metrics_df = pd.DataFrame(performances).T
                metrics_df = metrics_df.round(3)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Performance comparison chart
                fig = go.Figure()
                
                models = list(performances.keys())
                accuracies = [performances[model]['accuracy'] for model in models]
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=accuracies,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                    text=[f'{acc:.1f}%' for acc in accuracies],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f'{target_var.title()} Model Accuracy Comparison',
                    xaxis_title='Models',
                    yaxis_title='Accuracy (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔄 Train models to see accuracy metrics")
    
    def create_prediction_charts(self, predictions: Dict, target_variable: str) -> go.Figure:
        """Create beautiful prediction charts with confidence intervals"""
        fig = go.Figure()
        
        # Time axis (next 7 days)
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=len(predictions['predictions']),
            freq='H'
        )
        
        # Main prediction line
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions['predictions'],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=6)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions['confidence_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions['confidence_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.2)',
            name='Confidence Interval',
            hoverinfo='skip'
        ))
        
        # Individual model predictions
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for i, (model_name, pred) in enumerate(predictions['individual_predictions'].items()):
            if model_name != 'ensemble':
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=pred,
                    mode='lines',
                    name=f'{model_name.upper()}',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title=f'{target_variable.title()} Predictions - Next 7 Days',
            xaxis_title='Time',
            yaxis_title=f'{target_variable.title()}',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def train_models_section(self):
        """Model training section"""
        st.markdown("## 🧠 Model Training & Optimization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Advanced ML Pipeline
            - **Feature Engineering**: 50+ advanced features including lag variables, rolling statistics, and weather interactions
            - **Hyperparameter Optimization**: Automated tuning using Optuna
            - **Ensemble Methods**: Combines XGBoost, LightGBM, Random Forest, and LSTM
            - **Time Series Validation**: Proper temporal splitting for realistic evaluation
            """)
        
        with col2:
            target_vars = ['temperature', 'humidity', 'pm25', 'pm10', 'aqi']
            selected_target = st.selectbox("Select Target Variable", target_vars)
            
            if st.button("🚀 Train Enhanced Models", type="primary"):
                with st.spinner(f"Training enhanced models for {selected_target}..."):
                    try:
                        results = self.ml_models.train_enhanced_models(selected_target)
                        st.session_state.model_performance[selected_target] = results['performances']
                        st.session_state.models_trained = True
                        st.success(f"✅ Models trained successfully for {selected_target}!")
                        
                        # Display results
                        st.markdown("### 📊 Training Results")
                        for model_name, metrics in results['performances'].items():
                            st.markdown(f"**{model_name.upper()}**: Accuracy {metrics['accuracy']:.1f}% | RMSE {metrics['rmse']:.3f}")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
    
    def prediction_section(self, data: Dict[str, pd.DataFrame]):
        """Enhanced prediction section"""
        st.markdown("## 🔮 AI-Powered Predictions")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first in the Model Training section")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target_vars = list(st.session_state.model_performance.keys())
            if target_vars:
                selected_target = st.selectbox("Select Variable to Predict", target_vars)
                location_options = list(PUNE_LOCATIONS.keys())
                selected_location = st.selectbox("Select Location", location_options)
                
                if st.button("🎯 Generate Predictions"):
                    with st.spinner("Generating predictions..."):
                        try:
                            # Prepare input data
                            recent_data = data["weather"] if selected_target in ['temperature', 'humidity'] else data["aqi"]
                            location_data = recent_data[recent_data['location_id'] == selected_location].tail(24)
                            
                            if not location_data.empty:
                                predictions = self.ml_models.predict_enhanced(selected_target, location_data)
                                
                                # Store in session state
                                st.session_state.current_predictions = predictions
                                st.session_state.current_target = selected_target
                                st.session_state.current_location = selected_location
                                
                                st.success("✅ Predictions generated!")
                            else:
                                st.error("❌ No recent data available for selected location")
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
        
        with col2:
            if hasattr(st.session_state, 'current_predictions'):
                predictions = st.session_state.current_predictions
                target_var = st.session_state.current_target
                location = st.session_state.current_location
                
                # Display prediction chart
                fig = self.create_prediction_charts(predictions, target_var)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary
                location_name = PUNE_LOCATIONS[location].name
                avg_prediction = np.mean(predictions['predictions'])
                confidence_range = np.mean(predictions['prediction_std'])
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>📍 {location_name} - {target_var.title()} Forecast</h3>
                    <h2>Average: {avg_prediction:.2f} ± {confidence_range:.2f}</h2>
                    <p>Next 24 hours prediction with 95% confidence interval</p>
                </div>
                """, unsafe_allow_html=True)
    
    def run_dashboard(self):
        """Main dashboard function"""
        # Header
        st.markdown('<h1 class="main-header">🌍 Pune Climate & AQI Prediction Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced AI-Powered Environmental Monitoring & Forecasting")
        
        # Sidebar
        st.sidebar.markdown("## 🎛️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "📅 Data Time Range",
            ["Last 6 hours", "Last 24 hours", "Last 3 days", "Last 7 days"],
            index=1,
        )
        
        hours_map = {
            "Last 6 hours": 6,
            "Last 24 hours": 24,
            "Last 3 days": 72,
            "Last 7 days": 168,
        }
        hours = hours_map[time_range]
        
        # Location filter
        location_options = ["All Locations"] + [loc.name for loc in PUNE_LOCATIONS.values()]
        selected_locations = st.sidebar.multiselect(
            "📍 Select Locations",
            location_options,
            default=["All Locations"]
        )
        
        # Load data
        with st.spinner("Loading data..."):
            data = self.load_recent_data(hours)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🗺️ Location Map", "📈 Time Series", "🧠 Model Training", "🔮 Predictions"
        ])
        
        with tab1:
            # Beautiful metrics
            self.create_beautiful_metrics(data)
            
            # Data quality indicators
            st.markdown("## 📊 Data Quality & System Status")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                weather_points = len(data["weather"]) if not data["weather"].empty else 0
                st.metric("🌤️ Weather Data Points", f"{weather_points}")
            
            with col2:
                aqi_points = len(data["aqi"]) if not data["aqi"].empty else 0
                st.metric("🏭 AQI Data Points", f"{aqi_points}")
            
            with col3:
                total_points = weather_points + aqi_points
                st.metric("📊 Total Data Points", f"{total_points}")
            
            # Model accuracy display
            self.display_model_accuracy()
        
        with tab2:
            st.markdown("## 🗺️ Interactive Location Map")
            map_fig = self.create_interactive_map(data)
            st.plotly_chart(map_fig, use_container_width=True)
            
            # Location details
            st.markdown("### 📍 Monitoring Locations")
            for loc_id, loc_config in PUNE_LOCATIONS.items():
                with st.expander(f"{loc_config.name} - {loc_config.zone} Zone"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Coordinates:** {loc_config.latitude:.4f}, {loc_config.longitude:.4f}")
                        st.write(f"**District:** {loc_config.district}")
                    with col2:
                        st.write(f"**Elevation:** {loc_config.elevation}m")
                        st.write(f"**Zone:** {loc_config.zone}")
        
        with tab3:
            st.markdown("## 📈 Time Series Analysis")
            
            # Variable selector
            col1, col2 = st.columns([1, 3])
            with col1:
                variables = ['temperature', 'humidity', 'aqi', 'pm25', 'pm10']
                selected_var = st.selectbox("Select Variable", variables)
            
            with col2:
                if selected_var:
                    ts_fig = self.create_time_series_chart(data, selected_var)
                    st.plotly_chart(ts_fig, use_container_width=True)
        
        with tab4:
            self.train_models_section()
        
        with tab5:
            self.prediction_section(data)
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("*🌍 Enhanced Climate & AQI Prediction Dashboard - Powered by Advanced AI*")


def main():
    """Main function to run the enhanced dashboard"""
    dashboard = EnhancedDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()