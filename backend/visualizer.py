"""
Visualization Module
Creates interactive visualizations: line charts, bar graphs, heatmaps, scatter plots, forecast curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ClimateVisualizer:
    """Advanced visualization system for climate data and predictions"""
    
    def __init__(self):
        self.color_palette = {
            'temperature': '#FF6B6B',
            'rainfall': '#4ECDC4',
            'humidity': '#45B7D1',
            'aqi': '#96CEB4',
            'co2': '#FFEAA7',
            'wind_speed': '#DDA0DD',
            'pressure': '#98D8C8'
        }
        
        self.season_colors = {
            'Winter': '#87CEEB',
            'Summer': '#FFB347',
            'Monsoon': '#98FB98',
            'Post-Monsoon': '#DDA0DD'
        }
    
    def create_time_series_plot(self, df: pd.DataFrame, variables: List[str], 
                               title: str = "Climate Time Series") -> go.Figure:
        """Create interactive time series plot"""
        print(f"📈 Creating time series plot for {len(variables)} variables...")
        
        fig = make_subplots(
            rows=len(variables), cols=1,
            subplot_titles=[f"{var.title()} Over Time" for var in variables],
            vertical_spacing=0.08
        )
        
        for i, var in enumerate(variables, 1):
            if var in df.columns:
                color = self.color_palette.get(var, '#1f77b4')
                
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[var],
                        mode='lines',
                        name=var.title(),
                        line=dict(color=color, width=2),
                        hovertemplate=f'<b>{var.title()}</b><br>' +
                                    'Date: %{x}<br>' +
                                    f'Value: %{{y:.2f}}<br>' +
                                    '<extra></extra>'
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=300 * len(variables),
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date")
        
        print("✅ Time series plot created")
        return fig
    
    def create_seasonal_analysis(self, df: pd.DataFrame, variable: str) -> go.Figure:
        """Create seasonal analysis visualization"""
        print(f"🌱 Creating seasonal analysis for {variable}...")
        
        # Calculate seasonal statistics
        seasonal_stats = df.groupby('season')[variable].agg(['mean', 'std', 'min', 'max']).round(2)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"Average {variable.title()} by Season",
                f"{variable.title()} Distribution by Season",
                f"Monthly {variable.title()} Pattern",
                f"Seasonal {variable.title()} Trends Over Years"
            ],
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Seasonal averages
        seasons = seasonal_stats.index
        colors = [self.season_colors.get(season, '#1f77b4') for season in seasons]
        
        fig.add_trace(
            go.Bar(
                x=seasons,
                y=seasonal_stats['mean'],
                name='Average',
                marker_color=colors,
                error_y=dict(type='data', array=seasonal_stats['std']),
                hovertemplate='<b>%{x}</b><br>Average: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Box plots by season
        for season in seasons:
            season_data = df[df['season'] == season][variable]
            fig.add_trace(
                go.Box(
                    y=season_data,
                    name=season,
                    marker_color=self.season_colors.get(season, '#1f77b4'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Monthly pattern
        monthly_avg = df.groupby('month')[variable].mean()
        fig.add_trace(
            go.Scatter(
                x=monthly_avg.index,
                y=monthly_avg.values,
                mode='lines+markers',
                name='Monthly Average',
                line=dict(color=self.color_palette.get(variable, '#1f77b4'), width=3),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Yearly trends by season
        yearly_seasonal = df.groupby(['year', 'season'])[variable].mean().unstack()
        for season in yearly_seasonal.columns:
            fig.add_trace(
                go.Scatter(
                    x=yearly_seasonal.index,
                    y=yearly_seasonal[season],
                    mode='lines+markers',
                    name=season,
                    line=dict(color=self.season_colors.get(season, '#1f77b4')),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(text=f"Seasonal Analysis - {variable.title()}", x=0.5, font=dict(size=20)),
            height=800,
            showlegend=True
        )
        
        print("✅ Seasonal analysis created")
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, variables: List[str]) -> go.Figure:
        """Create correlation heatmap"""
        print(f"🔥 Creating correlation heatmap for {len(variables)} variables...")
        
        # Calculate correlation matrix
        correlation_matrix = df[variables].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Climate Variables Correlation Matrix", x=0.5, font=dict(size=20)),
            width=600,
            height=600
        )
        
        print("✅ Correlation heatmap created")
        return fig
    
    def create_prediction_comparison(self, historical_df: pd.DataFrame, 
                                   predictions_df: pd.DataFrame, 
                                   variable: str) -> go.Figure:
        """Create prediction vs historical comparison"""
        print(f"🔮 Creating prediction comparison for {variable}...")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=historical_df['date'],
                y=historical_df[variable],
                mode='lines',
                name='Historical Data',
                line=dict(color=self.color_palette.get(variable, '#1f77b4'), width=2),
                hovertemplate=f'<b>Historical {variable.title()}</b><br>' +
                            'Date: %{x}<br>' +
                            f'Value: %{{y:.2f}}<br>' +
                            '<extra></extra>'
            )
        )
        
        # Predictions
        pred_col = f'{variable}_predicted'
        if pred_col in predictions_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions_df['date'],
                    y=predictions_df[pred_col],
                    mode='lines',
                    name='Predictions',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate=f'<b>Predicted {variable.title()}</b><br>' +
                                'Date: %{x}<br>' +
                                f'Value: %{{y:.2f}}<br>' +
                                '<extra></extra>'
                )
            )
            
            # Add confidence intervals if available
            if f'{variable}_lower' in predictions_df.columns and f'{variable}_upper' in predictions_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=predictions_df['date'],
                        y=predictions_df[f'{variable}_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=predictions_df['date'],
                        y=predictions_df[f'{variable}_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        name='Confidence Interval',
                        hoverinfo='skip'
                    )
                )
        
        # Add vertical line to separate historical and predicted data
        if len(historical_df) > 0 and len(predictions_df) > 0:
            try:
                last_historical_date = historical_df['date'].max()
                fig.add_vline(
                    x=last_historical_date,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="Prediction Start"
                )
            except Exception as e:
                print(f"⚠️ Could not add separation line: {e}")
        
        fig.update_layout(
            title=dict(text=f"{variable.title()} - Historical vs Predicted", x=0.5, font=dict(size=20)),
            xaxis_title="Date",
            yaxis_title=variable.title(),
            height=500,
            hovermode='x unified'
        )
        
        print("✅ Prediction comparison created")
        return fig
    
    def create_climate_dashboard(self, df: pd.DataFrame, 
                               predictions: Dict[str, pd.DataFrame] = None) -> Dict[str, go.Figure]:
        """Create comprehensive climate dashboard"""
        print("🎛️ Creating comprehensive climate dashboard...")
        
        dashboard_plots = {}
        
        # 1. Main climate variables time series
        main_vars = ['temperature', 'rainfall', 'humidity', 'aqi']
        available_vars = [var for var in main_vars if var in df.columns]
        
        if available_vars:
            dashboard_plots['time_series'] = self.create_time_series_plot(
                df, available_vars, "Climate Variables Over Time"
            )
        
        # 2. Seasonal analysis for temperature and rainfall
        for var in ['temperature', 'rainfall']:
            if var in df.columns:
                dashboard_plots[f'seasonal_{var}'] = self.create_seasonal_analysis(df, var)
        
        # 3. Correlation heatmap
        numeric_vars = df.select_dtypes(include=[np.number]).columns
        climate_vars = [var for var in numeric_vars if var not in ['year', 'month', 'day', 'day_of_year']]
        
        if len(climate_vars) > 2:
            dashboard_plots['correlation'] = self.create_correlation_heatmap(df, climate_vars[:8])
        
        # 4. Prediction comparisons
        if predictions:
            for var, pred_df in predictions.items():
                if not pred_df.empty:
                    dashboard_plots[f'prediction_{var}'] = self.create_prediction_comparison(
                        df, pred_df, var
                    )
        
        # 5. Climate change trends
        if 'year' in df.columns:
            dashboard_plots['trends'] = self.create_climate_trends(df)
        
        print(f"✅ Dashboard created with {len(dashboard_plots)} plots")
        return dashboard_plots
    
    def create_climate_trends(self, df: pd.DataFrame) -> go.Figure:
        """Create climate change trends visualization"""
        print("📊 Creating climate change trends...")
        
        # Calculate yearly averages
        yearly_stats = df.groupby('year').agg({
            'temperature': 'mean',
            'rainfall': 'sum',
            'aqi': 'mean',
            'co2': 'mean' if 'co2' in df.columns else lambda x: np.nan
        }).round(2)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Average Temperature Trend",
                "Annual Rainfall Trend", 
                "Air Quality Index Trend",
                "CO₂ Levels Trend"
            ]
        )
        
        # Temperature trend
        fig.add_trace(
            go.Scatter(
                x=yearly_stats.index,
                y=yearly_stats['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Rainfall trend
        fig.add_trace(
            go.Scatter(
                x=yearly_stats.index,
                y=yearly_stats['rainfall'],
                mode='lines+markers',
                name='Rainfall',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # AQI trend
        fig.add_trace(
            go.Scatter(
                x=yearly_stats.index,
                y=yearly_stats['aqi'],
                mode='lines+markers',
                name='AQI',
                line=dict(color='#96CEB4', width=3),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # CO2 trend (if available)
        if 'co2' in yearly_stats.columns and not yearly_stats['co2'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=yearly_stats.index,
                    y=yearly_stats['co2'],
                    mode='lines+markers',
                    name='CO₂',
                    line=dict(color='#FFEAA7', width=3),
                    marker=dict(size=6)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(text="Climate Change Trends Over Time", x=0.5, font=dict(size=20)),
            height=800,
            showlegend=False
        )
        
        print("✅ Climate trends created")
        return fig


def generate_visuals(df: pd.DataFrame, predictions: Dict[str, pd.DataFrame] = None,
                    variables: List[str] = None) -> Dict[str, go.Figure]:
    """
    Main function to generate comprehensive visualizations
    
    Args:
        df: Historical climate data
        predictions: Dictionary of prediction DataFrames for different variables
        variables: List of variables to visualize
    
    Returns:
        Dictionary of Plotly figures
    """
    print("🎨 GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 60)
    
    if variables is None:
        variables = ['temperature', 'rainfall', 'humidity', 'aqi']
    
    visualizer = ClimateVisualizer()
    
    # Generate dashboard
    visuals = visualizer.create_climate_dashboard(df, predictions)
    
    print(f"\n📊 VISUALIZATION SUMMARY:")
    print(f"   📈 Total plots created: {len(visuals)}")
    print(f"   🎯 Variables analyzed: {variables}")
    print(f"   📅 Data period: {df['date'].min()} to {df['date'].max()}")
    
    if predictions:
        print(f"   🔮 Predictions included: {list(predictions.keys())}")
    
    return visuals


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    sample_data = {
        'date': dates,
        'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 5, len(dates)),
        'aqi': 70 + np.random.normal(0, 15, len(dates)),
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'season': ['Winter' if m in [12,1,2] else 'Summer' if m in [3,4,5] else 'Monsoon' if m in [6,7,8,9] else 'Post-Monsoon' for m in [d.month for d in dates]]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Test visualization generation
    visuals = generate_visuals(df_sample)
    
    print(f"\n✅ Visualization test completed!")
    print(f"Generated {len(visuals)} visualizations")