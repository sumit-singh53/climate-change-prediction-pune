"""
Advanced Real-time Dashboard for Climate and AQI Monitoring and Prediction
Professional Streamlit dashboard with comprehensive visualizations and ML analytics
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

from optimized_ml_models import OptimizedMLModels
from config import DATABASE_CONFIG, PUNE_LOCATIONS
from realtime_data_collector import RealtimeDataCollector

# Page configuration
st.set_page_config(
    page_title="🌍 Pune Climate & AQI Analytics Dashboard",
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
    
    .location-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .data-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


class AdvancedDashboard:
    """Enhanced dashboard with beautiful visualizations and real-time ML training data"""

    def __init__(self):
        self.db_path = DATABASE_CONFIG["sqlite_path"]
        self.ml_models = OptimizedMLModels()
        self.realtime_collector = RealtimeDataCollector()

    def load_recent_data(self, hours: int = 24) -> Dict[str, pd.DataFrame]:
        """Load recent data from all sources"""
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

    def create_interactive_map(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create beautiful interactive map with real-time data"""
        
        fig = go.Figure()
        
        # Add location markers with real-time data
        for loc_id, loc_config in PUNE_LOCATIONS.items():
            # Get latest data for this location
            if not data["weather"].empty:
                latest_weather = data["weather"][data["weather"]["location_id"] == loc_id]
                if not latest_weather.empty:
                    temp = latest_weather.iloc[0]["temperature"]
                    humidity = latest_weather.iloc[0]["humidity"]
                else:
                    temp = "N/A"
                    humidity = "N/A"
            else:
                temp = "N/A"
                humidity = "N/A"
            
            if not data["aqi"].empty:
                latest_aqi = data["aqi"][data["aqi"]["location_id"] == loc_id]
                if not latest_aqi.empty:
                    aqi = latest_aqi.iloc[0]["aqi"]
                    pm25 = latest_aqi.iloc[0]["pm25"]
                else:
                    aqi = "N/A"
                    pm25 = "N/A"
            else:
                aqi = "N/A"
                pm25 = "N/A"
            
            # Color coding based on AQI
            if aqi != "N/A":
                if aqi <= 50:
                    color = "green"
                    size = 20
                elif aqi <= 100:
                    color = "yellow"
                    size = 25
                elif aqi <= 150:
                    color = "orange"
                    size = 30
                else:
                    color = "red"
                    size = 35
            else:
                color = "gray"
                size = 15
            
            fig.add_trace(go.Scattermapbox(
                lat=[loc_config.lat],
                lon=[loc_config.lon],
                mode='markers',
                marker=dict(size=size, color=color),
                text=f"<b>{loc_config.name}</b><br>🌡️ Temp: {temp}°C<br>💨 AQI: {aqi}<br>🌫️ PM2.5: {pm25}",
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

    def create_real_time_charts(self, data: Dict[str, pd.DataFrame]) -> List[go.Figure]:
        """Create beautiful real-time charts for ML training visualization"""
        
        charts = []
        
        if not data["weather"].empty and not data["aqi"].empty:
            # Combine weather and AQI data
            combined_data = pd.merge(
                data["weather"], data["aqi"], 
                on=["timestamp", "location_id"], 
                how="inner"  # Use inner join to avoid NaN issues
            )
            
            # Clean data - remove rows with NaN values in key columns
            combined_data = combined_data.dropna(subset=["temperature", "aqi", "humidity"])
            
            if not combined_data.empty:
                # 1. Temperature vs AQI Correlation
                fig1 = px.scatter(
                    combined_data, 
                    x="temperature", 
                    y="aqi",
                    color="location_id",
                    size="humidity",
                    title="🌡️ Temperature vs AQI Correlation (Real-time ML Training Data)",
                    labels={"temperature": "Temperature (°C)", "aqi": "Air Quality Index"},
                    template="plotly_dark"
                )
                fig1.update_layout(height=400)
                charts.append(fig1)
            
                # 2. Multi-variable Time Series
                fig2 = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("🌡️ Temperature", "💧 Humidity", "🌫️ PM2.5", "💨 AQI"),
                    vertical_spacing=0.1
                )
                
                for location in combined_data["location_id"].unique():
                    loc_data = combined_data[combined_data["location_id"] == location]
                    
                    # Only add traces if data exists
                    if not loc_data.empty:
                        fig2.add_trace(
                            go.Scatter(x=loc_data["timestamp"], y=loc_data["temperature"], 
                                      name=f"{location} Temp", line=dict(width=2)),
                            row=1, col=1
                        )
                        fig2.add_trace(
                            go.Scatter(x=loc_data["timestamp"], y=loc_data["humidity"], 
                                      name=f"{location} Humidity", line=dict(width=2)),
                            row=1, col=2
                        )
                        fig2.add_trace(
                            go.Scatter(x=loc_data["timestamp"], y=loc_data["pm25"], 
                                      name=f"{location} PM2.5", line=dict(width=2)),
                            row=2, col=1
                        )
                        fig2.add_trace(
                            go.Scatter(x=loc_data["timestamp"], y=loc_data["aqi"], 
                                      name=f"{location} AQI", line=dict(width=2)),
                            row=2, col=2
                        )
                
                fig2.update_layout(
                    height=600, 
                    title_text="📊 Real-time Multi-variable Analysis for ML Training",
                    template="plotly_dark",
                    showlegend=False
                )
                charts.append(fig2)
                
                # 3. Correlation Heatmap
                numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
                # Remove non-numeric or problematic columns
                numeric_cols = [col for col in numeric_cols if col not in ['timestamp']]
                
                if len(numeric_cols) > 1:
                    correlation_data = combined_data[numeric_cols].dropna()
                    if not correlation_data.empty:
                        correlation_matrix = correlation_data.corr()
                        
                        fig3 = px.imshow(
                            correlation_matrix,
                            title="🔥 Feature Correlation Matrix (ML Training Insights)",
                            color_continuous_scale="RdBu_r",
                            template="plotly_dark"
                        )
                        fig3.update_layout(height=500)
                        charts.append(fig3)
        
        # If no combined data, create individual charts
        elif not data["weather"].empty or not data["aqi"].empty:
            if not data["weather"].empty:
                weather_clean = data["weather"].dropna(subset=["temperature", "humidity"])
                if not weather_clean.empty:
                    fig_weather = px.line(
                        weather_clean,
                        x="timestamp",
                        y="temperature",
                        color="location_id",
                        title="🌡️ Temperature Trends (Weather Data Only)",
                        template="plotly_dark"
                    )
                    fig_weather.update_layout(height=400)
                    charts.append(fig_weather)
            
            if not data["aqi"].empty:
                aqi_clean = data["aqi"].dropna(subset=["aqi", "pm25"])
                if not aqi_clean.empty:
                    fig_aqi = px.line(
                        aqi_clean,
                        x="timestamp",
                        y="aqi",
                        color="location_id",
                        title="💨 AQI Trends (Air Quality Data Only)",
                        template="plotly_dark"
                    )
                    fig_aqi.update_layout(height=400)
                    charts.append(fig_aqi)
            
        return charts

    def create_ml_training_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate real-time ML training metrics"""
        
        metrics = {
            "data_points": 0,
            "locations_active": 0,
            "time_span_hours": 0,
            "avg_temperature": 0,
            "temp_variance": 0,
            "avg_aqi": 0,
            "aqi_variance": 0,
            "pollution_trend": "No Data"
        }
        
        try:
            if not data["weather"].empty:
                weather_df = data["weather"].dropna(subset=["temperature"])
                if not weather_df.empty:
                    metrics["data_points"] = len(weather_df)
                    metrics["locations_active"] = weather_df["location_id"].nunique()
                    
                    if len(weather_df) > 1:
                        time_diff = (weather_df["timestamp"].max() - weather_df["timestamp"].min()).total_seconds() / 3600
                        metrics["time_span_hours"] = max(0, time_diff)
                    
                    metrics["avg_temperature"] = weather_df["temperature"].mean()
                    if len(weather_df) > 1:
                        metrics["temp_variance"] = weather_df["temperature"].var()
                
            if not data["aqi"].empty:
                aqi_df = data["aqi"].dropna(subset=["aqi"])
                if not aqi_df.empty:
                    metrics["avg_aqi"] = aqi_df["aqi"].mean()
                    if len(aqi_df) > 1:
                        metrics["aqi_variance"] = aqi_df["aqi"].var()
                        
                        # Check trend only if we have multiple data points
                        if len(aqi_df) >= 2:
                            first_aqi = aqi_df.iloc[0]["aqi"]
                            last_aqi = aqi_df.iloc[-1]["aqi"]
                            metrics["pollution_trend"] = "Improving" if last_aqi < first_aqi else "Worsening"
        
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            
        return metrics

    def run_dashboard(self):
        """Run the enhanced dashboard"""
        
        # Header with gradient
        st.markdown('<h1 class="main-header">🌍 Pune Climate & AQI Analytics System</h1>', unsafe_allow_html=True)
        st.markdown("### 🤖 Real-time ML Training Data & Advanced Analytics")
        
        # Sidebar controls
        st.sidebar.markdown("## 🎛️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=True)
        
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
        
        # Location selector
        selected_locations = st.sidebar.multiselect(
            "📍 Select Locations",
            list(PUNE_LOCATIONS.keys()),
            default=list(PUNE_LOCATIONS.keys())[:4]
        )
        
        # Load data
        with st.spinner("🔄 Loading real-time data for ML training..."):
            data = self.load_recent_data(hours)
        
        # Filter data by selected locations
        if selected_locations:
            for key in data:
                if not data[key].empty and "location_id" in data[key].columns:
                    data[key] = data[key][data[key]["location_id"].isin(selected_locations)]
        
        # Real-time metrics
        st.markdown("## 📊 Real-time ML Training Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = self.create_ml_training_metrics(data)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 Data Points", f"{metrics.get('data_points', 0):,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📍 Active Locations", f"{metrics.get('locations_active', 0)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🌡️ Avg Temperature", f"{metrics.get('avg_temperature', 0):.1f}°C")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💨 Avg AQI", f"{metrics.get('avg_aqi', 0):.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Data Span", f"{metrics.get('time_span_hours', 0):.1f}h")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive Map
        st.markdown("## 🗺️ Real-time Location Status")
        map_fig = self.create_interactive_map(data)
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Real-time Charts for ML Training
        st.markdown("## 📈 Real-time ML Training Visualizations")
        charts = self.create_real_time_charts(data)
        
        for chart in charts:
            st.plotly_chart(chart, use_container_width=True)
        
        # Location Details
        st.markdown("## 📍 Location Details & Training Data")
        
        for loc_id in selected_locations:
            if loc_id in PUNE_LOCATIONS:
                loc_config = PUNE_LOCATIONS[loc_id]
                
                with st.expander(f"🌍 {loc_config.name} - {loc_config.zone} Zone"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="location-card">', unsafe_allow_html=True)
                        st.write(f"**📍 Coordinates:** {loc_config.lat:.4f}, {loc_config.lon:.4f}")
                        st.write(f"**🏙️ District:** {loc_config.district}")
                        st.write(f"**🌐 Zone:** {loc_config.zone}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Latest weather data
                        if not data["weather"].empty:
                            loc_weather = data["weather"][data["weather"]["location_id"] == loc_id]
                            if not loc_weather.empty:
                                latest = loc_weather.iloc[0]
                                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                                st.write("**🌤️ Latest Weather:**")
                                st.write(f"🌡️ Temperature: {latest['temperature']:.1f}°C")
                                st.write(f"💧 Humidity: {latest['humidity']:.1f}%")
                                st.write(f"💨 Wind: {latest['wind_speed']:.1f} m/s")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        # Latest AQI data
                        if not data["aqi"].empty:
                            loc_aqi = data["aqi"][data["aqi"]["location_id"] == loc_id]
                            if not loc_aqi.empty:
                                latest = loc_aqi.iloc[0]
                                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                                st.write("**🏭 Latest Air Quality:**")
                                st.write(f"💨 AQI: {latest['aqi']:.0f}")
                                st.write(f"🌫️ PM2.5: {latest['pm25']:.1f}")
                                st.write(f"🌪️ PM10: {latest['pm10']:.1f}")
                                st.markdown('</div>', unsafe_allow_html=True)
        
        # ML Model Status
        st.markdown("## 🤖 ML Model Training Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.write("**🎯 Target Variables:**")
            st.write("• Temperature")
            st.write("• Humidity") 
            st.write("• PM2.5")
            st.write("• PM10")
            st.write("• AQI")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.write("**🧠 ML Models:**")
            st.write("• XGBoost (Optimized)")
            st.write("• LightGBM (Optimized)")
            st.write("• Random Forest")
            st.write("• LSTM Neural Network")
            st.write("• Ensemble Voting")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.write("**⚡ Optimization:**")
            st.write("• Optuna Hyperparameter Tuning")
            st.write("• 50+ Trial Optimization")
            st.write("• Cross-validation")
            st.write("• Feature Selection")
            st.write("• Ensemble Weighting")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data Quality Metrics
        st.markdown("## 📊 Data Quality for ML Training")
        
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        
        with quality_col1:
            weather_completeness = (
                (1 - data["weather"].isnull().sum().sum() / data["weather"].size) * 100
                if not data["weather"].empty
                else 0
            )
            st.metric("🌤️ Weather Data Completeness", f"{weather_completeness:.1f}%")

        with quality_col2:
            aqi_completeness = (
                (1 - data["aqi"].isnull().sum().sum() / data["aqi"].size) * 100
                if not data["aqi"].empty
                else 0
            )
            st.metric("🏭 AQI Data Completeness", f"{aqi_completeness:.1f}%")

        with quality_col3:
            # Total data points for ML training
            weather_points = len(data["weather"]) if not data["weather"].empty else 0
            aqi_points = len(data["aqi"]) if not data["aqi"].empty else 0
            total_points = weather_points + aqi_points
            st.metric("📈 Total ML Training Points", f"{total_points:,}")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("*🤖 Enhanced dashboard with real-time ML training data visualization*")
        st.markdown("*🔄 Auto-refresh enabled for continuous monitoring*")


def main():
    """Main function to run the enhanced dashboard"""
    dashboard = AdvancedDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()