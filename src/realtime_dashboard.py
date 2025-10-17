"""
Real-time Dashboard for Climate and AQI Monitoring and Prediction
Streamlit-based interactive dashboard with live updates and IoT integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import asyncio
import json
import time
from typing import Dict, List, Optional

from config import PUNE_LOCATIONS, DATABASE_CONFIG
from advanced_ml_models import AdvancedMLModels
from realtime_data_collector import RealtimeDataCollector
from enhanced_data_collector import EnhancedDataCollector

# Page configuration
st.set_page_config(
    page_title="Pune Climate & AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RealtimeDashboard:
    """Real-time dashboard for climate and AQI monitoring"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG['sqlite_path']
        self.ml_models = AdvancedMLModels()
        self.realtime_collector = RealtimeDataCollector()
        
        # Load models if available
        try:
            self.ml_models.load_models()
        except Exception as e:
            st.warning(f"Could not load ML models: {e}")
    
    def load_recent_data(self, hours: int = 24) -> Dict[str, pd.DataFrame]:
        """Load recent data from all sources"""
        conn = sqlite3.connect(self.db_path)
        
        # Real-time weather data
        weather_query = f'''
            SELECT * FROM realtime_weather 
            WHERE timestamp > datetime('now', '-{hours} hours')
            ORDER BY timestamp DESC
        '''
        weather_df = pd.read_sql_query(weather_query, conn)
        
        # Real-time air quality data
        aqi_query = f'''
            SELECT * FROM realtime_air_quality 
            WHERE timestamp > datetime('now', '-{hours} hours')
            ORDER BY timestamp DESC
        '''
        aqi_df = pd.read_sql_query(aqi_query, conn)
        
        # Historical data for comparison
        historical_weather_query = f'''
            SELECT * FROM weather_historical 
            WHERE timestamp > datetime('now', '-{hours} hours')
            ORDER BY timestamp DESC
        '''
        historical_weather_df = pd.read_sql_query(historical_weather_query, conn)
        
        conn.close()
        
        # Convert timestamps
        for df in [weather_df, aqi_df, historical_weather_df]:
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return {
            'weather': weather_df,
            'aqi': aqi_df,
            'historical_weather': historical_weather_df
        }
    
    def create_location_map(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create interactive map showing all monitoring locations"""
        fig = go.Figure()
        
        # Add location markers
        for location_id, location_config in PUNE_LOCATIONS.items():
            # Get latest data for this location
            latest_temp = None
            latest_aqi = None
            
            if not data['weather'].empty:
                location_weather = data['weather'][data['weather']['location_id'] == location_id]
                if not location_weather.empty:
                    latest_temp = location_weather.iloc[0]['temperature']
            
            if not data['aqi'].empty:
                location_aqi = data['aqi'][data['aqi']['location_id'] == location_id]
                if not location_aqi.empty:
                    latest_aqi = location_aqi.iloc[0]['pm25']
            
            # Color based on AQI level
            color = 'green'
            if latest_aqi:
                if latest_aqi > 100:
                    color = 'red'
                elif latest_aqi > 50:
                    color = 'orange'
                elif latest_aqi > 25:
                    color = 'yellow'
            
            hover_text = f"<b>{location_config.name}</b><br>"
            hover_text += f"Zone: {location_config.zone}<br>"
            if latest_temp:
                hover_text += f"Temperature: {latest_temp:.1f}°C<br>"
            if latest_aqi:
                hover_text += f"PM2.5: {latest_aqi:.1f} µg/m³"
            
            fig.add_trace(go.Scattermapbox(
                lat=[location_config.lat],
                lon=[location_config.lon],
                mode='markers',
                marker=dict(size=15, color=color),
                text=hover_text,
                hoverinfo='text',
                name=location_config.name
            ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=18.5204, lon=73.8567),
                zoom=10
            ),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        return fig
    
    def create_time_series_chart(self, data: pd.DataFrame, y_column: str, title: str, color: str = 'blue') -> go.Figure:
        """Create time series chart for a specific metric"""
        fig = go.Figure()
        
        if not data.empty and y_column in data.columns:
            for location_id in data['location_id'].unique():
                location_data = data[data['location_id'] == location_id].sort_values('timestamp')
                location_name = PUNE_LOCATIONS.get(location_id, {}).name if location_id in PUNE_LOCATIONS else location_id
                
                fig.add_trace(go.Scatter(
                    x=location_data['timestamp'],
                    y=location_data[y_column],
                    mode='lines+markers',
                    name=location_name,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_column.replace('_', ' ').title(),
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    
    def create_aqi_gauge(self, aqi_value: float) -> go.Figure:
        """Create AQI gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=aqi_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Air Quality Index"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 300]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "yellow"},
                    {'range': [100, 150], 'color': "orange"},
                    {'range': [150, 200], 'color': "red"},
                    {'range': [200, 300], 'color': "purple"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    
    def create_prediction_chart(self, location_id: str, target_variable: str, days: int = 7) -> go.Figure:
        """Create prediction chart for a specific variable"""
        fig = go.Figure()
        
        try:
            # Get historical data
            conn = sqlite3.connect(self.db_path)
            
            if target_variable in ['temperature', 'humidity', 'pressure']:
                table = 'weather_historical'
            else:
                table = 'air_quality_historical'
            
            query = f'''
                SELECT timestamp, {target_variable} 
                FROM {table} 
                WHERE location_id = ? AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp
            '''
            
            historical_df = pd.read_sql_query(query, conn, params=(location_id,))
            conn.close()
            
            if not historical_df.empty:
                historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
                
                # Plot historical data
                fig.add_trace(go.Scatter(
                    x=historical_df['timestamp'],
                    y=historical_df[target_variable],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Generate predictions
                if hasattr(self.ml_models, 'models') and target_variable in self.ml_models.models:
                    future_dates = []
                    predictions = []
                    
                    for i in range(1, days + 1):
                        future_date = datetime.now() + timedelta(days=i)
                        future_dates.append(future_date)
                        
                        try:
                            pred_result = self.ml_models.predict(location_id, target_variable, horizon_days=i)
                            predictions.append(pred_result['prediction'])
                        except:
                            predictions.append(None)
                    
                    # Plot predictions
                    valid_predictions = [(date, pred) for date, pred in zip(future_dates, predictions) if pred is not None]
                    
                    if valid_predictions:
                        pred_dates, pred_values = zip(*valid_predictions)
                        
                        fig.add_trace(go.Scatter(
                            x=pred_dates,
                            y=pred_values,
                            mode='lines+markers',
                            name='Predictions',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(symbol='diamond', size=8)
                        ))
        
        except Exception as e:
            st.error(f"Error creating prediction chart: {e}")
        
        fig.update_layout(
            title=f"{target_variable.replace('_', ' ').title()} - Historical & Predictions",
            xaxis_title="Date",
            yaxis_title=target_variable.replace('_', ' ').title(),
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for environmental variables"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Environmental Variables Correlation",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def display_iot_status(self):
        """Display IoT sensor status"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get sensor registry
            sensor_query = '''
                SELECT sensor_id, location_id, sensor_type, status, installation_date
                FROM sensor_registry
                ORDER BY location_id, sensor_type
            '''
            
            sensors_df = pd.read_sql_query(sensor_query, conn)
            
            # Get recent data counts
            recent_data_query = '''
                SELECT location_id, sensor_type, COUNT(*) as data_points
                FROM iot_sensor_data
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY location_id, sensor_type
            '''
            
            recent_data_df = pd.read_sql_query(recent_data_query, conn)
            conn.close()
            
            if not sensors_df.empty:
                st.subheader("IoT Sensor Status")
                
                # Merge data
                status_df = sensors_df.merge(
                    recent_data_df, 
                    on=['location_id', 'sensor_type'], 
                    how='left'
                )
                status_df['data_points'] = status_df['data_points'].fillna(0)
                status_df['status_indicator'] = status_df['data_points'].apply(
                    lambda x: "🟢 Active" if x > 0 else "🔴 Inactive"
                )
                
                # Display as table
                display_df = status_df[['sensor_id', 'location_id', 'sensor_type', 'status_indicator', 'data_points']]
                display_df.columns = ['Sensor ID', 'Location', 'Type', 'Status', 'Data Points (1h)']
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No IoT sensors registered yet")
                
        except Exception as e:
            st.error(f"Error displaying IoT status: {e}")
    
    def run_dashboard(self):
        """Main dashboard function"""
        st.title("🌍 Pune Climate & AQI Prediction Dashboard")
        st.markdown("Real-time monitoring and prediction of climate and air quality across Pune")
        
        # Sidebar controls
        st.sidebar.header("Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "Data Time Range",
            ["Last 6 hours", "Last 24 hours", "Last 3 days", "Last 7 days"],
            index=1
        )
        
        hours_map = {
            "Last 6 hours": 6,
            "Last 24 hours": 24,
            "Last 3 days": 72,
            "Last 7 days": 168
        }
        hours = hours_map[time_range]
        
        # Location selector
        selected_locations = st.sidebar.multiselect(
            "Select Locations",
            list(PUNE_LOCATIONS.keys()),
            default=list(PUNE_LOCATIONS.keys())[:3]
        )
        
        # Load data
        with st.spinner("Loading data..."):
            data = self.load_recent_data(hours)
        
        # Filter data by selected locations
        if selected_locations:
            for key in data:
                if not data[key].empty and 'location_id' in data[key].columns:
                    data[key] = data[key][data[key]['location_id'].isin(selected_locations)]
        
        # Main dashboard layout
        col1, col2, col3, col4 = st.columns(4)
        
        # Key metrics
        with col1:
            if not data['weather'].empty:
                avg_temp = data['weather']['temperature'].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
            else:
                st.metric("Avg Temperature", "N/A")
        
        with col2:
            if not data['weather'].empty:
                avg_humidity = data['weather']['humidity'].mean()
                st.metric("Avg Humidity", f"{avg_humidity:.1f}%")
            else:
                st.metric("Avg Humidity", "N/A")
        
        with col3:
            if not data['aqi'].empty:
                avg_pm25 = data['aqi']['pm25'].mean()
                st.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
            else:
                st.metric("Avg PM2.5", "N/A")
        
        with col4:
            if not data['aqi'].empty:
                avg_aqi = data['aqi']['aqi'].mean()
                st.metric("Avg AQI", f"{avg_aqi:.0f}")
            else:
                st.metric("Avg AQI", "N/A")
        
        # Location map
        st.subheader("📍 Monitoring Locations")
        map_fig = self.create_location_map(data)
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Time series charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌡️ Temperature Trends")
            if not data['weather'].empty:
                temp_fig = self.create_time_series_chart(data['weather'], 'temperature', 'Temperature Over Time')
                st.plotly_chart(temp_fig, use_container_width=True)
            else:
                st.info("No temperature data available")
        
        with col2:
            st.subheader("💨 Air Quality Trends")
            if not data['aqi'].empty:
                pm25_fig = self.create_time_series_chart(data['aqi'], 'pm25', 'PM2.5 Over Time', 'red')
                st.plotly_chart(pm25_fig, use_container_width=True)
            else:
                st.info("No air quality data available")
        
        # AQI Gauge
        if not data['aqi'].empty:
            st.subheader("🎯 Current AQI Status")
            latest_aqi = data['aqi']['aqi'].iloc[0] if not data['aqi'].empty else 0
            aqi_gauge = self.create_aqi_gauge(latest_aqi)
            st.plotly_chart(aqi_gauge, use_container_width=True)
        
        # Predictions section
        st.subheader("🔮 Predictions")
        
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            pred_location = st.selectbox("Select Location for Predictions", selected_locations)
            pred_variable = st.selectbox("Select Variable", ['temperature', 'humidity', 'pm25', 'pm10', 'aqi'])
        
        with pred_col2:
            pred_days = st.slider("Prediction Days", 1, 7, 3)
        
        if pred_location and pred_variable:
            pred_fig = self.create_prediction_chart(pred_location, pred_variable, pred_days)
            st.plotly_chart(pred_fig, use_container_width=True)
        
        # Correlation analysis
        if not data['weather'].empty and not data['aqi'].empty:
            st.subheader("📊 Environmental Variables Correlation")
            
            # Merge weather and AQI data for correlation
            merged_data = pd.merge(
                data['weather'], 
                data['aqi'], 
                on=['timestamp', 'location_id'], 
                how='inner'
            )
            
            if not merged_data.empty:
                corr_fig = self.create_correlation_heatmap(merged_data)
                st.plotly_chart(corr_fig, use_container_width=True)
        
        # IoT sensor status
        self.display_iot_status()
        
        # Data quality metrics
        st.subheader("📈 Data Quality Metrics")
        
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        
        with quality_col1:
            weather_completeness = (1 - data['weather'].isnull().sum().sum() / data['weather'].size) * 100 if not data['weather'].empty else 0
            st.metric("Weather Data Completeness", f"{weather_completeness:.1f}%")
        
        with quality_col2:
            aqi_completeness = (1 - data['aqi'].isnull().sum().sum() / data['aqi'].size) * 100 if not data['aqi'].empty else 0
            st.metric("AQI Data Completeness", f"{aqi_completeness:.1f}%")
        
        with quality_col3:
            iot_data_points = len(data['iot']) if not data['iot'].empty else 0
            st.metric("IoT Data Points", f"{iot_data_points}")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("*Dashboard updates every 30 seconds when auto-refresh is enabled*")


def main():
    """Main function to run the dashboard"""
    dashboard = RealtimeDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()