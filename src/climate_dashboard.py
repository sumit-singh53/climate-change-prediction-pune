#!/usr/bin/env python3
"""
Comprehensive Climate Change Dashboard for Pune
Shows historical trends, current status, and future predictions
"""

import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_CONFIG, PUNE_LOCATIONS

# Page configuration
st.set_page_config(
    page_title="🌍 Pune Climate Change Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .trend-up {
        color: #FF6B6B;
        font-weight: bold;
    }
    
    .trend-down {
        color: #4ECDC4;
        font-weight: bold;
    }
    
    .trend-stable {
        color: #95E1D3;
        font-weight: bold;
    }
    
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class ClimateChangeDashboard:
    """Comprehensive climate change dashboard"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG["sqlite_path"]
    
    def load_data(self) -> tuple:
        """Load weather and AQI data from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Load weather data
        weather_query = """
            SELECT * FROM weather_historical 
            ORDER BY timestamp
        """
        weather_df = pd.read_sql_query(weather_query, conn)
        
        # Load AQI data
        aqi_query = """
            SELECT * FROM air_quality_historical 
            ORDER BY timestamp
        """
        aqi_df = pd.read_sql_query(aqi_query, conn)
        
        conn.close()
        
        # Convert timestamps
        for df in [weather_df, aqi_df]:
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['year'] = df['timestamp'].dt.year
                df['month'] = df['timestamp'].dt.month
                df['season'] = df['month'].apply(self.get_season)
        
        return weather_df, aqi_df
    
    def get_season(self, month):
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'
    
    def calculate_trends(self, df, variable, years_back=10):
        """Calculate climate trends"""
        recent_data = df[df['year'] >= (datetime.now().year - years_back)]
        if len(recent_data) < 2:
            return 0, 0, "stable"
        
        yearly_avg = recent_data.groupby('year')[variable].mean()
        if len(yearly_avg) < 2:
            return 0, 0, "stable"
        
        # Linear regression for trend
        X = yearly_avg.index.values.reshape(-1, 1)
        y = yearly_avg.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_per_year = model.coef_[0]
        trend_total = trend_per_year * years_back
        
        # Determine trend direction
        if abs(trend_per_year) < 0.01:
            direction = "stable"
        elif trend_per_year > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return trend_per_year, trend_total, direction
    
    def create_temperature_trend_chart(self, weather_df, analysis_mode="All Locations (Combined)", selected_locations=None):
        """Create temperature trend visualization with location context"""
        
        if analysis_mode == "Compare Locations" and len(selected_locations) > 1:
            # Create comparison chart for multiple locations
            fig = go.Figure()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            
            for i, location_id in enumerate(selected_locations):
                location_data = weather_df[weather_df['location_id'] == location_id]
                yearly_temp = location_data.groupby('year')['temperature'].mean().reset_index()
                
                fig.add_trace(go.Scatter(
                    x=yearly_temp['year'],
                    y=yearly_temp['temperature'],
                    mode='lines+markers',
                    name=PUNE_LOCATIONS[location_id].name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))
            
            title = f"🌡️ Temperature Comparison - {len(selected_locations)} Locations"
        else:
            # Single location or combined data
            yearly_temp = weather_df.groupby('year').agg({
                'temperature': ['mean', 'min', 'max']
            }).round(1)
            yearly_temp.columns = ['avg_temp', 'min_temp', 'max_temp']
            yearly_temp = yearly_temp.reset_index()
            
            fig = go.Figure()
            
            # Average temperature line
            fig.add_trace(go.Scatter(
                x=yearly_temp['year'],
                y=yearly_temp['avg_temp'],
                mode='lines+markers',
                name='Average Temperature',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=6)
            ))
            
            # Min/Max range
            fig.add_trace(go.Scatter(
                x=yearly_temp['year'],
                y=yearly_temp['max_temp'],
                mode='lines',
                name='Max Temperature',
                line=dict(color='#FF9999', width=1),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=yearly_temp['year'],
                y=yearly_temp['min_temp'],
                mode='lines',
                name='Temperature Range',
                line=dict(color='#FF9999', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.2)'
            ))
            
            # Add trend line
            if len(yearly_temp) > 1:
                X = yearly_temp['year'].values.reshape(-1, 1)
                y = yearly_temp['avg_temp'].values
                model = LinearRegression()
                model.fit(X, y)
                trend_line = model.predict(X)
                
                fig.add_trace(go.Scatter(
                    x=yearly_temp['year'],
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='#333', width=2, dash='dash')
                ))
            
            if analysis_mode == "Specific Location" and selected_locations:
                location_name = PUNE_LOCATIONS[selected_locations[0]].name
                title = f"🌡️ Temperature Trends - {location_name}"
            else:
                title = "🌡️ Temperature Trends in Pune (2000-2024)"
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_rainfall_pattern_chart(self, weather_df):
        """Create rainfall pattern visualization"""
        # Monthly rainfall patterns by year
        monthly_rain = weather_df.groupby(['year', 'month'])['precipitation'].sum().reset_index()
        
        # Create heatmap data
        heatmap_data = monthly_rain.pivot(index='year', columns='month', values='precipitation')
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=heatmap_data.index,
            colorscale='Blues',
            colorbar=dict(title="Rainfall (mm)")
        ))
        
        fig.update_layout(
            title="🌧️ Monthly Rainfall Patterns (2000-2024)",
            xaxis_title="Month",
            yaxis_title="Year",
            height=500
        )
        
        return fig
    
    def create_aqi_trend_chart(self, aqi_df, analysis_mode="All Locations (Combined)", selected_locations=None):
        """Create AQI trend visualization with location context"""
        
        if analysis_mode == "Compare Locations" and len(selected_locations) > 1:
            # Create comparison chart for multiple locations
            fig = go.Figure()
            colors = ['#9B59B6', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#1ABC9C', '#E67E22', '#34495E']
            
            for i, location_id in enumerate(selected_locations):
                location_data = aqi_df[aqi_df['location_id'] == location_id]
                yearly_aqi = location_data.groupby('year')['aqi'].mean().reset_index()
                
                fig.add_trace(go.Scatter(
                    x=yearly_aqi['year'],
                    y=yearly_aqi['aqi'],
                    mode='lines+markers',
                    name=PUNE_LOCATIONS[location_id].name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))
            
            title = f"🏭 AQI Comparison - {len(selected_locations)} Locations"
        else:
            # Single location or combined data
            yearly_aqi = aqi_df.groupby('year')['aqi'].mean().reset_index()
            
            fig = go.Figure()
            
            # AQI trend line
            fig.add_trace(go.Scatter(
                x=yearly_aqi['year'],
                y=yearly_aqi['aqi'],
                mode='lines+markers',
                name='Average AQI',
                line=dict(color='#9B59B6', width=3),
                marker=dict(size=8)
            ))
            
            if analysis_mode == "Specific Location" and selected_locations:
                location_name = PUNE_LOCATIONS[selected_locations[0]].name
                title = f"🏭 Air Quality Index - {location_name}"
            else:
                title = "🏭 Air Quality Index Trends in Pune"
        
        # AQI categories (for all chart types)
        fig.add_hline(y=50, line_dash="dash", line_color="green", 
                     annotation_text="Good (0-50)")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", 
                     annotation_text="Moderate (51-100)")
        fig.add_hline(y=150, line_dash="dash", line_color="orange", 
                     annotation_text="Unhealthy for Sensitive (101-150)")
        fig.add_hline(y=200, line_dash="dash", line_color="red", 
                     annotation_text="Unhealthy (151-200)")
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="AQI",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_seasonal_comparison(self, weather_df):
        """Create seasonal comparison charts"""
        seasonal_data = weather_df.groupby(['year', 'season']).agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'precipitation': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature by Season', 'Humidity by Season', 
                          'Rainfall by Season', 'Seasonal Trends'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        seasons = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
        
        for i, season in enumerate(seasons):
            season_data = seasonal_data[seasonal_data['season'] == season]
            
            # Temperature
            fig.add_trace(go.Scatter(
                x=season_data['year'],
                y=season_data['temperature'],
                mode='lines',
                name=f'{season} Temp',
                line=dict(color=colors[i]),
                showlegend=False
            ), row=1, col=1)
            
            # Humidity
            fig.add_trace(go.Scatter(
                x=season_data['year'],
                y=season_data['humidity'],
                mode='lines',
                name=f'{season} Humidity',
                line=dict(color=colors[i]),
                showlegend=False
            ), row=1, col=2)
            
            # Rainfall
            fig.add_trace(go.Scatter(
                x=season_data['year'],
                y=season_data['precipitation'],
                mode='lines',
                name=season,
                line=dict(color=colors[i])
            ), row=2, col=1)
        
        # Overall trends
        yearly_avg = weather_df.groupby('year').agg({
            'temperature': 'mean',
            'humidity': 'mean'
        }).reset_index()
        
        fig.add_trace(go.Scatter(
            x=yearly_avg['year'],
            y=yearly_avg['temperature'],
            mode='lines+markers',
            name='Avg Temperature',
            line=dict(color='#E74C3C', width=2)
        ), row=2, col=2)
        
        fig.update_layout(height=800, title_text="🌍 Seasonal Climate Patterns")
        return fig
    
    def predict_future_climate(self, weather_df, aqi_df):
        """Predict future climate using machine learning"""
        # Prepare data for prediction
        yearly_weather = weather_df.groupby('year').agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'precipitation': 'sum'
        }).reset_index()
        
        yearly_aqi = aqi_df.groupby('year')['aqi'].mean().reset_index()
        
        # Combine data
        combined_data = pd.merge(yearly_weather, yearly_aqi, on='year')
        
        # Features and targets
        X = combined_data[['year']].values
        
        # Train models
        models = {}
        predictions = {}
        
        for variable in ['temperature', 'humidity', 'precipitation', 'aqi']:
            y = combined_data[variable].values
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            models[variable] = model
            
            # Predict future years (2025-2050)
            future_years = np.array([[year] for year in range(2025, 2051)])
            future_pred = model.predict(future_years)
            predictions[variable] = future_pred
        
        return models, predictions
    
    def create_future_predictions_chart(self, weather_df, aqi_df, predictions):
        """Create future predictions visualization"""
        # Historical data
        yearly_weather = weather_df.groupby('year').agg({
            'temperature': 'mean',
            'precipitation': 'sum'
        }).reset_index()
        
        yearly_aqi = aqi_df.groupby('year')['aqi'].mean().reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Predictions', 'Rainfall Predictions',
                          'AQI Predictions', 'Climate Change Summary'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        future_years = list(range(2025, 2051))
        
        # Temperature predictions
        fig.add_trace(go.Scatter(
            x=yearly_weather['year'],
            y=yearly_weather['temperature'],
            mode='lines+markers',
            name='Historical Temperature',
            line=dict(color='#3498DB')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=future_years,
            y=predictions['temperature'],
            mode='lines+markers',
            name='Predicted Temperature',
            line=dict(color='#E74C3C', dash='dash')
        ), row=1, col=1)
        
        # Rainfall predictions
        fig.add_trace(go.Scatter(
            x=yearly_weather['year'],
            y=yearly_weather['precipitation'],
            mode='lines+markers',
            name='Historical Rainfall',
            line=dict(color='#2ECC71'),
            showlegend=False
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=future_years,
            y=predictions['precipitation'],
            mode='lines+markers',
            name='Predicted Rainfall',
            line=dict(color='#F39C12', dash='dash'),
            showlegend=False
        ), row=1, col=2)
        
        # AQI predictions
        fig.add_trace(go.Scatter(
            x=yearly_aqi['year'],
            y=yearly_aqi['aqi'],
            mode='lines+markers',
            name='Historical AQI',
            line=dict(color='#9B59B6'),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=future_years,
            y=predictions['aqi'],
            mode='lines+markers',
            name='Predicted AQI',
            line=dict(color='#E67E22', dash='dash'),
            showlegend=False
        ), row=2, col=1)
        
        # Summary chart - temperature change
        current_temp = yearly_weather['temperature'].iloc[-5:].mean()
        future_temp_2030 = predictions['temperature'][5]  # 2030
        future_temp_2050 = predictions['temperature'][-1]  # 2050
        
        fig.add_trace(go.Bar(
            x=['Current (2020-2024)', '2030 Prediction', '2050 Prediction'],
            y=[current_temp, future_temp_2030, future_temp_2050],
            name='Temperature Change',
            marker_color=['#3498DB', '#F39C12', '#E74C3C'],
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(height=800, title_text="🔮 Future Climate Predictions for Pune")
        return fig
    
    def display_location_specific_analysis(self, weather_df, aqi_df, location_id):
        """Display detailed analysis for a specific location"""
        location_config = PUNE_LOCATIONS[location_id]
        
        st.subheader(f"📍 Detailed Analysis: {location_config.name}")
        
        # Location overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            **Location Details:**
            - **District:** {location_config.district}
            - **Zone:** {location_config.zone}
            - **Coordinates:** {location_config.lat:.4f}°N, {location_config.lon:.4f}°E
            """)
        
        with col2:
            # Quick stats
            if not weather_df.empty and not aqi_df.empty:
                recent_weather = weather_df[weather_df['year'] >= 2020]
                recent_aqi = aqi_df[aqi_df['year'] >= 2020]
                
                avg_temp_recent = recent_weather['temperature'].mean()
                avg_aqi_recent = recent_aqi['aqi'].mean()
                
                st.metric("Recent Avg Temp", f"{avg_temp_recent:.1f}°C")
                st.metric("Recent Avg AQI", f"{avg_aqi_recent:.0f}")
        
        # Seasonal breakdown for this location
        if not weather_df.empty:
            seasonal_stats = weather_df.groupby('season').agg({
                'temperature': 'mean',
                'humidity': 'mean',
                'precipitation': 'sum'
            }).round(1)
            
            st.markdown("**Seasonal Climate Profile:**")
            
            cols = st.columns(4)
            seasons = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
            season_icons = ['❄️', '☀️', '🌧️', '🍂']
            
            for i, season in enumerate(seasons):
                if season in seasonal_stats.index:
                    with cols[i]:
                        st.markdown(f"""
                        **{season_icons[i]} {season}**
                        - Temp: {seasonal_stats.loc[season, 'temperature']:.1f}°C
                        - Humidity: {seasonal_stats.loc[season, 'humidity']:.1f}%
                        - Rain: {seasonal_stats.loc[season, 'precipitation']:.0f}mm
                        """)
        
        # Climate change impact for this location
        if len(weather_df) > 100:  # Ensure sufficient data
            temp_trend, _, temp_direction = self.calculate_trends(weather_df, 'temperature')
            aqi_trend, _, aqi_direction = self.calculate_trends(aqi_df, 'aqi')
            
            st.markdown("**Climate Change Impact:**")
            
            impact_col1, impact_col2 = st.columns(2)
            
            with impact_col1:
                temp_color = "🔴" if temp_direction == "increasing" else "🟢" if temp_direction == "decreasing" else "🟡"
                st.markdown(f"{temp_color} **Temperature:** {temp_direction} at {temp_trend:+.3f}°C/year")
            
            with impact_col2:
                aqi_color = "🔴" if aqi_direction == "increasing" else "🟢" if aqi_direction == "decreasing" else "🟡"
                st.markdown(f"{aqi_color} **Air Quality:** {aqi_direction} at {aqi_trend:+.1f} AQI/year")

    def display_climate_metrics(self, weather_df, aqi_df):
        """Display key climate metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Current year data
        current_year = datetime.now().year
        current_weather = weather_df[weather_df['year'] == current_year]
        current_aqi = aqi_df[aqi_df['year'] == current_year]
        
        # Calculate trends
        temp_trend, temp_total, temp_direction = self.calculate_trends(weather_df, 'temperature')
        rain_trend, rain_total, rain_direction = self.calculate_trends(weather_df, 'precipitation')
        aqi_trend, aqi_total, aqi_direction = self.calculate_trends(aqi_df, 'aqi')
        humidity_trend, humidity_total, humidity_direction = self.calculate_trends(weather_df, 'humidity')
        
        with col1:
            avg_temp = current_weather['temperature'].mean() if not current_weather.empty else weather_df['temperature'].mean()
            trend_class = f"trend-{temp_direction.replace('increasing', 'up').replace('decreasing', 'down').replace('stable', 'stable')}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌡️ Temperature</h3>
                <h2>{avg_temp:.1f}°C</h2>
                <p class="{trend_class}">
                    {temp_direction.title()}: {temp_trend:+.2f}°C/year
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_rain = current_weather['precipitation'].sum() if not current_weather.empty else weather_df.groupby('year')['precipitation'].sum().mean()
            trend_class = f"trend-{rain_direction.replace('increasing', 'up').replace('decreasing', 'down').replace('stable', 'stable')}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌧️ Annual Rainfall</h3>
                <h2>{avg_rain:.0f}mm</h2>
                <p class="{trend_class}">
                    {rain_direction.title()}: {rain_trend:+.1f}mm/year
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_aqi = current_aqi['aqi'].mean() if not current_aqi.empty else aqi_df['aqi'].mean()
            trend_class = f"trend-{aqi_direction.replace('increasing', 'up').replace('decreasing', 'down').replace('stable', 'stable')}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏭 Air Quality Index</h3>
                <h2>{avg_aqi:.0f}</h2>
                <p class="{trend_class}">
                    {aqi_direction.title()}: {aqi_trend:+.1f}/year
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_humidity = current_weather['humidity'].mean() if not current_weather.empty else weather_df['humidity'].mean()
            trend_class = f"trend-{humidity_direction.replace('increasing', 'up').replace('decreasing', 'down').replace('stable', 'stable')}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>💧 Humidity</h3>
                <h2>{avg_humidity:.1f}%</h2>
                <p class="{trend_class}">
                    {humidity_direction.title()}: {humidity_trend:+.1f}%/year
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def run_dashboard(self):
        """Main dashboard function"""
        st.markdown('<h1 class="main-header">🌍 Pune Climate Change Dashboard</h1>', unsafe_allow_html=True)
        
        # Load data
        with st.spinner("Loading climate data..."):
            weather_df, aqi_df = self.load_data()
        
        if weather_df.empty or aqi_df.empty:
            st.error("❌ No data found! Please run the data generator first.")
            st.code("python src/comprehensive_data_generator.py")
            return
        
        # Display metrics
        st.subheader("📊 Current Climate Status")
        self.display_climate_metrics(weather_df, aqi_df)
        
        # Sidebar controls
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Date range selector
        min_year = int(weather_df['year'].min())
        max_year = int(weather_df['year'].max())
        
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        # Filter data
        filtered_weather = weather_df[
            (weather_df['year'] >= year_range[0]) & 
            (weather_df['year'] <= year_range[1])
        ]
        filtered_aqi = aqi_df[
            (aqi_df['year'] >= year_range[0]) & 
            (aqi_df['year'] <= year_range[1])
        ]
        
        # Location selector with enhanced options
        st.sidebar.subheader("📍 Location Analysis")
        
        # Analysis mode selector
        analysis_mode = st.sidebar.radio(
            "Analysis Mode",
            ["All Locations (Combined)", "Specific Location", "Compare Locations"]
        )
        
        if analysis_mode == "All Locations (Combined)":
            selected_locations = list(PUNE_LOCATIONS.keys())
            location_info = "Showing combined data from all 8 Pune locations"
        elif analysis_mode == "Specific Location":
            location_names = {k: v.name for k, v in PUNE_LOCATIONS.items()}
            selected_location = st.sidebar.selectbox(
                "Choose Location",
                options=list(PUNE_LOCATIONS.keys()),
                format_func=lambda x: location_names[x]
            )
            selected_locations = [selected_location]
            location_config = PUNE_LOCATIONS[selected_location]
            location_info = f"📍 **{location_config.name}**\n- District: {location_config.district}\n- Zone: {location_config.zone}\n- Coordinates: {location_config.lat:.3f}°N, {location_config.lon:.3f}°E"
        else:  # Compare Locations
            selected_locations = st.sidebar.multiselect(
                "Select Locations to Compare",
                options=list(PUNE_LOCATIONS.keys()),
                default=list(PUNE_LOCATIONS.keys())[:3],
                format_func=lambda x: PUNE_LOCATIONS[x].name
            )
            location_info = f"Comparing {len(selected_locations)} locations: {', '.join([PUNE_LOCATIONS[loc].name for loc in selected_locations])}"
        
        # Display location info
        st.sidebar.info(location_info)
        
        # Filter data based on selection
        if selected_locations:
            filtered_weather = filtered_weather[filtered_weather['location_id'].isin(selected_locations)]
            filtered_aqi = filtered_aqi[filtered_aqi['location_id'].isin(selected_locations)]
        
        # Show data summary for selected locations
        if analysis_mode == "Specific Location":
            st.sidebar.markdown("### 📊 Location Statistics")
            loc_weather = filtered_weather
            loc_aqi = filtered_aqi
            
            if not loc_weather.empty and not loc_aqi.empty:
                avg_temp = loc_weather['temperature'].mean()
                avg_rain = loc_weather['precipitation'].sum() / len(loc_weather['year'].unique())
                avg_aqi = loc_aqi['aqi'].mean()
                
                st.sidebar.metric("Avg Temperature", f"{avg_temp:.1f}°C")
                st.sidebar.metric("Annual Rainfall", f"{avg_rain:.0f}mm")
                st.sidebar.metric("Avg AQI", f"{avg_aqi:.0f}")
                
                # Air quality category
                if avg_aqi <= 50:
                    aqi_category = "🟢 Good"
                elif avg_aqi <= 100:
                    aqi_category = "🟡 Moderate"
                elif avg_aqi <= 150:
                    aqi_category = "🟠 Unhealthy for Sensitive"
                else:
                    aqi_category = "🔴 Unhealthy"
                
                st.sidebar.markdown(f"**Air Quality:** {aqi_category}")
        
        # Main charts with location context
        if analysis_mode == "Specific Location":
            location_name = PUNE_LOCATIONS[selected_locations[0]].name
            st.subheader(f"📈 Climate Trends for {location_name}")
        else:
            st.subheader("📈 Historical Climate Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temp_chart = self.create_temperature_trend_chart(filtered_weather, analysis_mode, selected_locations)
            st.plotly_chart(temp_chart, use_container_width=True)
        
        with col2:
            aqi_chart = self.create_aqi_trend_chart(filtered_aqi, analysis_mode, selected_locations)
            st.plotly_chart(aqi_chart, use_container_width=True)
        
        # Location-specific detailed analysis
        if analysis_mode == "Specific Location" and selected_locations:
            self.display_location_specific_analysis(filtered_weather, filtered_aqi, selected_locations[0])
        
        # Rainfall patterns
        if analysis_mode == "Specific Location":
            location_name = PUNE_LOCATIONS[selected_locations[0]].name
            st.subheader(f"🌧️ Rainfall Patterns - {location_name}")
        else:
            st.subheader("🌧️ Rainfall Patterns")
        rainfall_chart = self.create_rainfall_pattern_chart(filtered_weather)
        st.plotly_chart(rainfall_chart, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("🌍 Seasonal Climate Analysis")
        seasonal_chart = self.create_seasonal_comparison(filtered_weather)
        st.plotly_chart(seasonal_chart, use_container_width=True)
        
        # Future predictions
        st.subheader("🔮 Future Climate Predictions")
        
        with st.spinner("Training ML models and generating predictions..."):
            models, predictions = self.predict_future_climate(weather_df, aqi_df)
        
        future_chart = self.create_future_predictions_chart(weather_df, aqi_df, predictions)
        st.plotly_chart(future_chart, use_container_width=True)
        
        # Climate change summary
        st.subheader("📋 Climate Change Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **🌡️ Temperature Trends:**
            - Current average: {weather_df['temperature'].mean():.1f}°C
            - 10-year trend: {self.calculate_trends(weather_df, 'temperature')[1]:+.1f}°C
            - 2050 prediction: {predictions['temperature'][-1]:.1f}°C
            """)
            
            st.warning(f"""
            **🌧️ Rainfall Patterns:**
            - Annual average: {weather_df.groupby('year')['precipitation'].sum().mean():.0f}mm
            - 10-year trend: {self.calculate_trends(weather_df, 'precipitation')[1]:+.0f}mm
            - 2050 prediction: {predictions['precipitation'][-1]:.0f}mm
            """)
        
        with col2:
            st.error(f"""
            **🏭 Air Quality:**
            - Current AQI: {aqi_df['aqi'].mean():.0f}
            - 10-year trend: {self.calculate_trends(aqi_df, 'aqi')[1]:+.0f} points
            - 2050 prediction: {predictions['aqi'][-1]:.0f}
            """)
            
            st.success(f"""
            **💧 Humidity Levels:**
            - Current average: {weather_df['humidity'].mean():.1f}%
            - 10-year trend: {self.calculate_trends(weather_df, 'humidity')[1]:+.1f}%
            - 2050 prediction: {predictions['humidity'][-1]:.1f}%
            """)
        
        # Data summary
        st.subheader("📊 Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Weather Records", f"{len(weather_df):,}")
        with col2:
            st.metric("AQI Records", f"{len(aqi_df):,}")
        with col3:
            st.metric("Years of Data", f"{max_year - min_year + 1}")

def main():
    """Main function"""
    dashboard = ClimateChangeDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()