"""
Enhanced Visualization Module for Pune Climate Dashboard
Creates interactive and visually appealing charts using Plotly Express
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ClimateVisualizationEngine:
    """
    Comprehensive visualization engine for climate dashboard
    Creates all required chart types with interactive features
    """
    
    def __init__(self):
        self.color_palette = {
            'temperature': '#FF6B6B',
            'rainfall': '#4ECDC4', 
            'humidity': '#45B7D1',
            'aqi': '#96CEB4',
            'co2': '#FFEAA7',
            'wind_speed': '#DDA0DD',
            'pressure': '#98D8C8',
            'solar_radiation': '#FFB347'
        }
        
        self.season_colors = {
            'Winter': '#87CEEB',
            'Summer': '#FFB347', 
            'Monsoon': '#98FB98',
            'Post-Monsoon': '#DDA0DD'
        }
    
    def create_line_chart_historical_vs_predicted(self, historical_data: pd.DataFrame, 
                                                predicted_data: pd.DataFrame, 
                                                variable: str) -> go.Figure:
        """
        Line Chart → Historical vs Predicted Temperature/Rainfall
        """
        print(f"📈 Creating line chart for {variable}: Historical vs Predicted")
        
        fig = go.Figure()
        
        # Historical data line
        if variable in historical_data.columns:
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data[variable],
                mode='lines',
                name=f'Historical {variable.title()}',
                line=dict(color=self.color_palette.get(variable, '#1f77b4'), width=2),
                hovertemplate=f'<b>Historical {variable.title()}</b><br>' +
                            'Date: %{x}<br>' +
                            f'Value: %{{y:.2f}}<br>' +
                            '<extra></extra>'
            ))
        
        # Predicted data line
        pred_col = f'{variable}_predicted'
        if pred_col in predicted_data.columns:
            fig.add_trace(go.Scatter(
                x=predicted_data['date'],
                y=predicted_data[pred_col],
                mode='lines',
                name=f'Predicted {variable.title()}',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate=f'<b>Predicted {variable.title()}</b><br>' +
                            'Date: %{x}<br>' +
                            f'Value: %{{y:.2f}}<br>' +
                            '<extra></extra>'
            ))
            
            # Add confidence intervals if available
            if f'{variable}_lower' in predicted_data.columns and f'{variable}_upper' in predicted_data.columns:
                fig.add_trace(go.Scatter(
                    x=predicted_data['date'],
                    y=predicted_data[f'{variable}_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=predicted_data['date'],
                    y=predicted_data[f'{variable}_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
        
        # Add vertical separator line
        if len(historical_data) > 0 and len(predicted_data) > 0:
            try:
                last_historical = historical_data['date'].max()
                fig.add_vline(
                    x=last_historical,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="Prediction Start",
                    annotation_position="top"
                )
            except Exception as e:
                print(f"⚠️ Could not add separator line: {e}")
        
        fig.update_layout(
            title=dict(
                text=f"{variable.title()} - Historical vs Predicted Trends",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="Date",
            yaxis_title=f"{variable.title()} ({'°C' if variable == 'temperature' else 'mm' if variable == 'rainfall' else 'units'})",
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_bar_chart_yearly_rainfall(self, data: pd.DataFrame) -> go.Figure:
        """
        Bar Chart → Yearly Rainfall Totals
        """
        print("🌧️ Creating yearly rainfall bar chart")
        
        if 'rainfall' not in data.columns or 'year' not in data.columns:
            raise ValueError("Rainfall or year data not available")
        
        # Calculate yearly rainfall totals
        yearly_rainfall = data.groupby('year')['rainfall'].sum().reset_index()
        yearly_rainfall['rainfall_mm'] = yearly_rainfall['rainfall'].round(1)
        
        # Create bar chart with color gradient
        fig = px.bar(
            yearly_rainfall,
            x='year',
            y='rainfall',
            title="Annual Rainfall Totals by Year",
            color='rainfall',
            color_continuous_scale='Blues',
            hover_data={'rainfall': ':.1f'},
            labels={'rainfall': 'Rainfall (mm)', 'year': 'Year'}
        )
        
        # Add trend line
        z = np.polyfit(yearly_rainfall['year'], yearly_rainfall['rainfall'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=yearly_rainfall['year'],
            y=p(yearly_rainfall['year']),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Customize layout
        fig.update_layout(
            height=500,
            xaxis_title="Year",
            yaxis_title="Rainfall (mm)",
            showlegend=True,
            hovermode='x'
        )
        
        # Add annotations for highest and lowest years
        max_year = yearly_rainfall.loc[yearly_rainfall['rainfall'].idxmax()]
        min_year = yearly_rainfall.loc[yearly_rainfall['rainfall'].idxmin()]
        
        fig.add_annotation(
            x=max_year['year'],
            y=max_year['rainfall'],
            text=f"Highest: {max_year['rainfall']:.0f}mm",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            bgcolor="lightgreen"
        )
        
        fig.add_annotation(
            x=min_year['year'],
            y=min_year['rainfall'],
            text=f"Lowest: {min_year['rainfall']:.0f}mm",
            showarrow=True,
            arrowhead=2,
            arrowcolor="orange",
            bgcolor="lightyellow"
        )
        
        return fig
    
    def create_heatmap_monthly_temp_humidity(self, data: pd.DataFrame) -> go.Figure:
        """
        Heatmap → Monthly Average Temperature and Humidity
        """
        print("🔥 Creating monthly temperature & humidity heatmap")
        
        if not all(col in data.columns for col in ['temperature', 'humidity', 'year', 'month']):
            raise ValueError("Required columns (temperature, humidity, year, month) not available")
        
        # Create subplots for temperature and humidity
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Average Temperature (°C)', 'Monthly Average Humidity (%)'),
            vertical_spacing=0.1
        )
        
        # Calculate monthly averages
        monthly_temp = data.groupby(['year', 'month'])['temperature'].mean().unstack()
        monthly_humidity = data.groupby(['year', 'month'])['humidity'].mean().unstack()
        
        # Temperature heatmap
        fig.add_trace(
            go.Heatmap(
                z=monthly_temp.values,
                x=[f"Month {i}" for i in monthly_temp.columns],
                y=monthly_temp.index,
                colorscale='RdYlBu_r',
                name='Temperature',
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Temperature: %{z:.1f}°C<extra></extra>',
                colorbar=dict(title="Temperature (°C)", x=1.02)
            ),
            row=1, col=1
        )
        
        # Humidity heatmap
        fig.add_trace(
            go.Heatmap(
                z=monthly_humidity.values,
                x=[f"Month {i}" for i in monthly_humidity.columns],
                y=monthly_humidity.index,
                colorscale='Blues',
                name='Humidity',
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Humidity: %{z:.1f}%<extra></extra>',
                colorbar=dict(title="Humidity (%)", x=1.15)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(
                text="Monthly Climate Patterns - Temperature & Humidity",
                x=0.5,
                font=dict(size=20)
            ),
            height=800
        )
        
        return fig
    
    def create_scatter_plot_co2_temperature(self, data: pd.DataFrame) -> go.Figure:
        """
        Scatter Plot → CO₂ vs Temperature Correlation
        """
        print("🌿 Creating CO₂ vs Temperature scatter plot")
        
        if not all(col in data.columns for col in ['co2', 'temperature']):
            raise ValueError("CO₂ or temperature data not available")
        
        # Create scatter plot with trend line
        fig = px.scatter(
            data,
            x='co2',
            y='temperature',
            title="CO₂ Levels vs Temperature Correlation",
            color='year' if 'year' in data.columns else None,
            size='aqi' if 'aqi' in data.columns else None,
            hover_data=['date', 'humidity'] if 'humidity' in data.columns else ['date'],
            trendline="ols",
            labels={
                'co2': 'CO₂ Concentration (ppm)',
                'temperature': 'Temperature (°C)'
            }
        )
        
        # Calculate correlation coefficient
        correlation = data['co2'].corr(data['temperature'])
        
        # Add correlation annotation
        fig.add_annotation(
            x=data['co2'].min() + (data['co2'].max() - data['co2'].min()) * 0.05,
            y=data['temperature'].max() - (data['temperature'].max() - data['temperature'].min()) * 0.05,
            text=f"Correlation: {correlation:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=14)
        )
        
        fig.update_layout(
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def create_forecast_curve_future_predictions(self, historical_data: pd.DataFrame,
                                               predictions: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Forecast Curve → Prophet or LSTM Future Predictions
        """
        print("🔮 Creating future predictions forecast curves")
        
        fig = go.Figure()
        
        # Add historical data for context
        for variable in ['temperature', 'rainfall']:
            if variable in historical_data.columns:
                fig.add_trace(go.Scatter(
                    x=historical_data['date'],
                    y=historical_data[variable],
                    mode='lines',
                    name=f'Historical {variable.title()}',
                    line=dict(color=self.color_palette.get(variable, '#1f77b4'), width=2),
                    opacity=0.7
                ))
        
        # Add prediction curves
        for variable, pred_data in predictions.items():
            if not pred_data.empty:
                pred_col = f'{variable}_predicted'
                if pred_col in pred_data.columns:
                    fig.add_trace(go.Scatter(
                        x=pred_data['date'],
                        y=pred_data[pred_col],
                        mode='lines+markers',
                        name=f'Predicted {variable.title()}',
                        line=dict(
                            color=self.color_palette.get(variable, '#1f77b4'),
                            width=3,
                            dash='dash'
                        ),
                        marker=dict(size=4)
                    ))
                    
                    # Add confidence intervals if available
                    if f'{variable}_lower' in pred_data.columns and f'{variable}_upper' in pred_data.columns:
                        fig.add_trace(go.Scatter(
                            x=pred_data['date'],
                            y=pred_data[f'{variable}_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=pred_data['date'],
                            y=pred_data[f'{variable}_lower'],
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=f'rgba({",".join(map(str, [int(self.color_palette.get(variable, "#1f77b4")[1:3], 16), int(self.color_palette.get(variable, "#1f77b4")[3:5], 16), int(self.color_palette.get(variable, "#1f77b4")[5:7], 16)]))},0.2)',
                            name=f'{variable.title()} Confidence',
                            hoverinfo='skip'
                        ))
        
        # Add prediction start line
        if len(historical_data) > 0:
            prediction_start = historical_data['date'].max()
            fig.add_vline(
                x=prediction_start,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top"
            )
        
        fig.update_layout(
            title=dict(
                text="Climate Forecast Curves - Future Predictions",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="Date",
            yaxis_title="Climate Variables",
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    def create_pie_chart_seasonal_rainfall(self, data: pd.DataFrame) -> go.Figure:
        """
        Pie Chart → Seasonal Rainfall Distribution
        """
        print("🥧 Creating seasonal rainfall distribution pie chart")
        
        if not all(col in data.columns for col in ['rainfall', 'season']):
            raise ValueError("Rainfall or season data not available")
        
        # Calculate seasonal rainfall totals
        seasonal_rainfall = data.groupby('season')['rainfall'].sum().reset_index()
        seasonal_rainfall['percentage'] = (seasonal_rainfall['rainfall'] / seasonal_rainfall['rainfall'].sum() * 100).round(1)
        
        # Create pie chart
        fig = px.pie(
            seasonal_rainfall,
            values='rainfall',
            names='season',
            title="Seasonal Rainfall Distribution",
            color='season',
            color_discrete_map=self.season_colors,
            hover_data=['percentage']
        )
        
        # Customize pie chart
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                         'Rainfall: %{value:.0f}mm<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            )
        )
        
        # Add annotations with rainfall amounts
        for i, row in seasonal_rainfall.iterrows():
            fig.add_annotation(
                text=f"{row['rainfall']:.0f}mm",
                x=0.5, y=-0.15 - i*0.05,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12, color=self.season_colors.get(row['season'], 'black'))
            )
        
        return fig
    
    def create_comprehensive_dashboard(self, historical_data: pd.DataFrame,
                                     predictions: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, go.Figure]:
        """
        Create all visualization types for the dashboard
        """
        print("🎨 Creating comprehensive visualization dashboard")
        
        dashboard_plots = {}
        
        try:
            # 1. Line Chart - Historical vs Predicted (Temperature)
            if predictions and 'temperature' in predictions:
                dashboard_plots['line_chart_temperature'] = self.create_line_chart_historical_vs_predicted(
                    historical_data, predictions['temperature'], 'temperature'
                )
            
            # 2. Line Chart - Historical vs Predicted (Rainfall)
            if predictions and 'rainfall' in predictions:
                dashboard_plots['line_chart_rainfall'] = self.create_line_chart_historical_vs_predicted(
                    historical_data, predictions['rainfall'], 'rainfall'
                )
            
            # 3. Bar Chart - Yearly Rainfall
            if 'rainfall' in historical_data.columns and 'year' in historical_data.columns:
                dashboard_plots['bar_chart_rainfall'] = self.create_bar_chart_yearly_rainfall(historical_data)
            
            # 4. Heatmap - Monthly Temperature & Humidity
            if all(col in historical_data.columns for col in ['temperature', 'humidity', 'year', 'month']):
                dashboard_plots['heatmap_monthly'] = self.create_heatmap_monthly_temp_humidity(historical_data)
            
            # 5. Scatter Plot - CO₂ vs Temperature
            if all(col in historical_data.columns for col in ['co2', 'temperature']):
                dashboard_plots['scatter_co2_temp'] = self.create_scatter_plot_co2_temperature(historical_data)
            
            # 6. Forecast Curve - Future Predictions
            if predictions:
                dashboard_plots['forecast_curve'] = self.create_forecast_curve_future_predictions(
                    historical_data, predictions
                )
            
            # 7. Pie Chart - Seasonal Rainfall
            if all(col in historical_data.columns for col in ['rainfall', 'season']):
                dashboard_plots['pie_chart_seasonal'] = self.create_pie_chart_seasonal_rainfall(historical_data)
            
        except Exception as e:
            print(f"⚠️ Error creating visualization: {e}")
        
        print(f"✅ Created {len(dashboard_plots)} visualizations")
        return dashboard_plots
    
    def create_climate_summary_chart(self, data: pd.DataFrame) -> go.Figure:
        """
        Create a summary chart showing all key climate variables
        """
        print("📊 Creating climate summary chart")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Trends', 'Rainfall Patterns', 'Air Quality Index', 'Humidity Levels'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Temperature
        if 'temperature' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['temperature'],
                    mode='lines',
                    name='Temperature',
                    line=dict(color='#FF6B6B', width=2)
                ),
                row=1, col=1
            )
        
        # Rainfall
        if 'rainfall' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['rainfall'],
                    mode='lines',
                    name='Rainfall',
                    line=dict(color='#4ECDC4', width=2)
                ),
                row=1, col=2
            )
        
        # AQI
        if 'aqi' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['aqi'],
                    mode='lines',
                    name='AQI',
                    line=dict(color='#96CEB4', width=2)
                ),
                row=2, col=1
            )
        
        # Humidity
        if 'humidity' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['humidity'],
                    mode='lines',
                    name='Humidity',
                    line=dict(color='#45B7D1', width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(
                text="Pune Climate Variables Overview",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=False
        )
        
        return fig


def create_all_visualizations(historical_data: pd.DataFrame,
                            predictions: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, go.Figure]:
    """
    Main function to create all required visualizations
    
    Args:
        historical_data: Historical climate data DataFrame
        predictions: Dictionary of prediction DataFrames by variable
    
    Returns:
        Dictionary of Plotly figures for all visualization types
    """
    print("🎨 CREATING ALL CLIMATE VISUALIZATIONS")
    print("=" * 60)
    
    visualizer = ClimateVisualizationEngine()
    
    # Create comprehensive dashboard
    all_plots = visualizer.create_comprehensive_dashboard(historical_data, predictions)
    
    # Add summary chart
    if not historical_data.empty:
        all_plots['climate_summary'] = visualizer.create_climate_summary_chart(historical_data)
    
    print(f"\n📊 VISUALIZATION SUMMARY:")
    print(f"   📈 Total visualizations created: {len(all_plots)}")
    print(f"   📅 Data period: {historical_data['date'].min()} to {historical_data['date'].max()}")
    
    if predictions:
        print(f"   🔮 Predictions included for: {list(predictions.keys())}")
    
    available_charts = list(all_plots.keys())
    print(f"   📋 Available charts: {', '.join(available_charts)}")
    
    return all_plots


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    print("🧪 Testing visualization engine...")
    
    # Sample historical data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    sample_data = {
        'date': dates,
        'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates)),
        'aqi': 70 + np.random.normal(0, 15, len(dates)),
        'co2': 410 + np.arange(len(dates)) * 0.01 + np.random.normal(0, 2, len(dates)),
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'season': ['Winter' if m in [12,1,2] else 'Summer' if m in [3,4,5] else 'Monsoon' if m in [6,7,8,9] else 'Post-Monsoon' for m in [d.month for d in dates]]
    }
    
    historical_df = pd.DataFrame(sample_data)
    
    # Sample prediction data
    pred_dates = pd.date_range('2025-01-01', '2030-12-31', freq='M')
    predictions = {
        'temperature': pd.DataFrame({
            'date': pred_dates,
            'temperature_predicted': 26 + 3 * np.sin(2 * np.pi * np.arange(len(pred_dates)) / 12) + np.random.normal(0, 1, len(pred_dates)),
            'temperature_lower': 24 + 3 * np.sin(2 * np.pi * np.arange(len(pred_dates)) / 12) + np.random.normal(0, 1, len(pred_dates)),
            'temperature_upper': 28 + 3 * np.sin(2 * np.pi * np.arange(len(pred_dates)) / 12) + np.random.normal(0, 1, len(pred_dates))
        }),
        'rainfall': pd.DataFrame({
            'date': pred_dates,
            'rainfall_predicted': 50 + 30 * np.sin(2 * np.pi * np.arange(len(pred_dates)) / 12) + np.random.normal(0, 10, len(pred_dates)),
            'rainfall_lower': 40 + 30 * np.sin(2 * np.pi * np.arange(len(pred_dates)) / 12) + np.random.normal(0, 10, len(pred_dates)),
            'rainfall_upper': 60 + 30 * np.sin(2 * np.pi * np.arange(len(pred_dates)) / 12) + np.random.normal(0, 10, len(pred_dates))
        })
    }
    
    # Test visualization creation
    all_visualizations = create_all_visualizations(historical_df, predictions)
    
    print(f"\n✅ Visualization testing completed!")
    print(f"Created {len(all_visualizations)} different chart types")
    print(f"Charts available: {list(all_visualizations.keys())}")