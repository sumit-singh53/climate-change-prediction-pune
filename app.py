"""
Streamlit Climate Prediction Web Application
Main application file for the Pune Climate Change Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
from data_collector import fetch_city_data
from data_preprocessor import clean_and_preprocess
from model_trainer import train_model, train_all_models
from predictor import predict_future, predict_multiple_targets
from evaluator import evaluate_model, evaluate_multiple_models
from visualizer import generate_visuals
from report_generator import generate_report

# Page configuration
st.set_page_config(
    page_title="Pune Climate Prediction System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'predictions_generated' not in st.session_state:
    st.session_state.predictions_generated = False

@st.cache_data
def load_climate_data(start_year: int, end_year: int, include_current: bool = True):
    """Load and cache climate data"""
    try:
        # Run async function in sync context
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

@st.cache_data
def preprocess_data(data: pd.DataFrame):
    """Preprocess and cache data"""
    try:
        results = clean_and_preprocess(
            data, 
            target_variables=['temperature', 'rainfall'],
            scaling_method='robust',
            create_features=True
        )
        return results
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return {}

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌡️ Pune Climate Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Climate Forecasting & Analysis for Pune, India</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Data Configuration
    st.sidebar.header("📊 Data Configuration")
    start_year = st.sidebar.slider("Start Year", 2000, 2023, 2020)
    end_year = st.sidebar.slider("End Year", 2021, 2024, 2024)
    include_current = st.sidebar.checkbox("Include Current Data", True)
    
    # Model Configuration
    st.sidebar.header("🤖 Model Configuration")
    target_variables = st.sidebar.multiselect(
        "Target Variables",
        ['temperature', 'rainfall', 'humidity', 'aqi'],
        default=['temperature', 'rainfall']
    )
    
    model_types = st.sidebar.multiselect(
        "Model Types",
        ['linear', 'random_forest', 'prophet', 'lstm'],
        default=['linear', 'random_forest', 'prophet']
    )
    
    # Prediction Configuration
    st.sidebar.header("🔮 Prediction Configuration")
    prediction_start = st.sidebar.slider("Prediction Start Year", 2025, 2030, 2026)
    prediction_end = st.sidebar.slider("Prediction End Year", 2030, 2050, 2030)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "🔮 Predictions", 
        "📈 Visualizations", 
        "📄 Reports"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown('<h2 class="sub-header">📊 Climate Data Overview</h2>', unsafe_allow_html=True)
        
        if st.button("🔄 Load Climate Data", type="primary"):
            with st.spinner("Loading climate data..."):
                data = load_climate_data(start_year, end_year, include_current)
                
                if not data.empty:
                    st.session_state.climate_data = data
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(data):,} records successfully!")
                else:
                    st.error("❌ Failed to load data")
        
        if st.session_state.data_loaded and 'climate_data' in st.session_state:
            data = st.session_state.climate_data
            
            # Data summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📅 Total Records", f"{len(data):,}")
            with col2:
                st.metric("🗓️ Date Range", f"{data['date'].min().year}-{data['date'].max().year}")
            with col3:
                st.metric("🌡️ Avg Temperature", f"{data['temperature'].mean():.1f}°C")
            with col4:
                st.metric("🌧️ Total Rainfall", f"{data['rainfall'].sum():.0f}mm")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Basic statistics
            st.subheader("📊 Statistical Summary")
            numeric_cols = ['temperature', 'rainfall', 'humidity', 'aqi', 'wind_speed']
            available_cols = [col for col in numeric_cols if col in data.columns]
            
            if available_cols:
                st.dataframe(data[available_cols].describe(), use_container_width=True)
            
            # Data quality check
            st.subheader("🔍 Data Quality")
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                st.warning("⚠️ Missing values detected:")
                st.write(missing_data[missing_data > 0])
            else:
                st.success("✅ No missing values detected")
    
    # Tab 2: Model Training
    with tab2:
        st.markdown('<h2 class="sub-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first in the Data Overview tab")
        else:
            data = st.session_state.climate_data
            
            # Preprocessing
            st.subheader("🔧 Data Preprocessing")
            if st.button("🔄 Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    preprocessed_results = preprocess_data(data)
                    
                    if preprocessed_results:
                        st.session_state.preprocessed_data = preprocessed_results
                        st.success("✅ Data preprocessing completed!")
                        
                        # Show preprocessing summary
                        final_data = preprocessed_results['final']
                        st.info(f"📊 Processed dataset: {len(final_data):,} records, {len(final_data.columns)} features")
            
            # Model Training
            st.subheader("🎯 Model Training")
            
            col1, col2 = st.columns(2)
            with col1:
                optimize_models = st.checkbox("🔧 Optimize Hyperparameters", False)
            with col2:
                save_models = st.checkbox("💾 Save Trained Models", True)
            
            if st.button("🚀 Train All Models", type="primary"):
                if 'preprocessed_data' not in st.session_state:
                    st.error("❌ Please preprocess data first")
                else:
                    processed_data = st.session_state.preprocessed_data['final']
                    
                    with st.spinner("Training models... This may take a few minutes."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        all_models = {}
                        total_models = len(target_variables) * len(model_types)
                        current_model = 0
                        
                        for target in target_variables:
                            if target in processed_data.columns:
                                all_models[target] = {}
                                
                                for model_type in model_types:
                                    try:
                                        current_model += 1
                                        status_text.text(f"Training {model_type} for {target}... ({current_model}/{total_models})")
                                        
                                        save_path = f"outputs/models/{target}_{model_type}_model.pkl" if save_models else None
                                        
                                        model_info = train_model(
                                            processed_data, 
                                            target, 
                                            model_type, 
                                            optimize=optimize_models,
                                            save_path=save_path
                                        )
                                        
                                        all_models[target][model_type] = model_info
                                        progress_bar.progress(current_model / total_models)
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error training {model_type} for {target}: {e}")
                                        continue
                        
                        st.session_state.trained_models = all_models
                        st.session_state.models_trained = True
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ All models trained successfully!")
                        st.success(f"🎉 Training completed! {len(all_models)} targets, {sum(len(models) for models in all_models.values())} models total.")
            
            # Model Performance Summary
            if st.session_state.models_trained and 'trained_models' in st.session_state:
                st.subheader("📊 Model Performance Summary")
                
                models = st.session_state.trained_models
                
                for target, target_models in models.items():
                    st.write(f"**{target.title()} Models:**")
                    
                    performance_data = []
                    for model_type, model_info in target_models.items():
                        performance_data.append({
                            'Model': model_type.title(),
                            'R² Score': f"{model_info.get('test_r2', 0):.3f}",
                            'RMSE': f"{model_info.get('test_rmse', 0):.3f}",
                            'MAE': f"{model_info.get('test_mae', 0):.3f}"
                        })
                    
                    if performance_data:
                        performance_df = pd.DataFrame(performance_data)
                        st.dataframe(performance_df, use_container_width=True)
                    
                    st.write("---")
    
    # Tab 3: Predictions
    with tab3:
        st.markdown('<h2 class="sub-header">🔮 Future Climate Predictions</h2>', unsafe_allow_html=True)
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first in the Model Training tab")
        else:
            st.subheader("🎯 Generate Predictions")
            
            future_years = list(range(prediction_start, prediction_end + 1))
            st.info(f"📅 Prediction period: {prediction_start} - {prediction_end} ({len(future_years)} years)")
            
            if st.button("🔮 Generate Predictions", type="primary"):
                models = st.session_state.trained_models
                historical_data = st.session_state.preprocessed_data['final']
                
                with st.spinner("Generating predictions..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_predictions = {}
                    total_predictions = sum(len(target_models) for target_models in models.values())
                    current_prediction = 0
                    
                    for target, target_models in models.items():
                        all_predictions[target] = {}
                        
                        for model_type, model_info in target_models.items():
                            try:
                                current_prediction += 1
                                status_text.text(f"Predicting {target} with {model_type}... ({current_prediction}/{total_predictions})")
                                
                                pred_df = predict_future(model_info, future_years, historical_data)
                                all_predictions[target][model_type] = pred_df
                                
                                progress_bar.progress(current_prediction / total_predictions)
                                
                            except Exception as e:
                                st.error(f"❌ Error predicting {target} with {model_type}: {e}")
                                continue
                    
                    st.session_state.predictions = all_predictions
                    st.session_state.predictions_generated = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ All predictions generated successfully!")
                    st.success(f"🎉 Predictions completed for {len(all_predictions)} variables!")
            
            # Display Predictions
            if st.session_state.predictions_generated and 'predictions' in st.session_state:
                st.subheader("📊 Prediction Results")
                
                predictions = st.session_state.predictions
                
                # Prediction summary
                for target, target_predictions in predictions.items():
                    st.write(f"**{target.title()} Predictions:**")
                    
                    # Show prediction statistics
                    for model_type, pred_df in target_predictions.items():
                        if not pred_df.empty:
                            pred_col = f'{target}_predicted'
                            if pred_col in pred_df.columns:
                                avg_prediction = pred_df[pred_col].mean()
                                min_prediction = pred_df[pred_col].min()
                                max_prediction = pred_df[pred_col].max()
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Model", model_type.title())
                                with col2:
                                    st.metric("Average", f"{avg_prediction:.2f}")
                                with col3:
                                    st.metric("Minimum", f"{min_prediction:.2f}")
                                with col4:
                                    st.metric("Maximum", f"{max_prediction:.2f}")
                    
                    st.write("---")
    
    # Tab 4: Visualizations
    with tab4:
        st.markdown('<h2 class="sub-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
        else:
            data = st.session_state.climate_data
            predictions = st.session_state.predictions if st.session_state.predictions_generated else None
            
            if st.button("🎨 Generate Visualizations"):
                with st.spinner("Creating visualizations..."):
                    # Prepare predictions for visualization
                    viz_predictions = {}
                    if predictions:
                        for target, target_preds in predictions.items():
                            # Use the best performing model's predictions
                            if target_preds:
                                best_model = list(target_preds.keys())[0]  # Use first available model
                                viz_predictions[target] = target_preds[best_model]
                    
                    visuals = generate_visuals(data, viz_predictions, target_variables)
                    st.session_state.visuals = visuals
                    st.success("✅ Visualizations created!")
            
            # Display visualizations
            if 'visuals' in st.session_state:
                visuals = st.session_state.visuals
                
                # Time series plots
                if 'time_series' in visuals:
                    st.subheader("📈 Climate Variables Time Series")
                    st.plotly_chart(visuals['time_series'], use_container_width=True)
                
                # Seasonal analysis
                for var in ['temperature', 'rainfall']:
                    seasonal_key = f'seasonal_{var}'
                    if seasonal_key in visuals:
                        st.subheader(f"🌱 {var.title()} Seasonal Analysis")
                        st.plotly_chart(visuals[seasonal_key], use_container_width=True)
                
                # Correlation heatmap
                if 'correlation' in visuals:
                    st.subheader("🔥 Variables Correlation Matrix")
                    st.plotly_chart(visuals['correlation'], use_container_width=True)
                
                # Prediction comparisons
                for var in target_variables:
                    pred_key = f'prediction_{var}'
                    if pred_key in visuals:
                        st.subheader(f"🔮 {var.title()} Predictions vs Historical")
                        st.plotly_chart(visuals[pred_key], use_container_width=True)
                
                # Climate trends
                if 'trends' in visuals:
                    st.subheader("📊 Climate Change Trends")
                    st.plotly_chart(visuals['trends'], use_container_width=True)
    
    # Tab 5: Reports
    with tab5:
        st.markdown('<h2 class="sub-header">📄 Comprehensive Reports</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
        else:
            st.subheader("📊 Report Generation")
            
            # Report configuration
            col1, col2 = st.columns(2)
            with col1:
                include_predictions = st.checkbox("Include Predictions", True)
                include_model_performance = st.checkbox("Include Model Performance", True)
            with col2:
                include_visualizations = st.checkbox("Include Visualizations", False)  # PDF limitations
                report_format = st.selectbox("Report Format", ["PDF", "HTML"])
            
            if st.button("📄 Generate Report", type="primary"):
                data = st.session_state.climate_data
                predictions = st.session_state.predictions if include_predictions and st.session_state.predictions_generated else None
                model_results = None
                visuals = st.session_state.visuals if include_visualizations and 'visuals' in st.session_state else None
                
                # Prepare model results for report
                if include_model_performance and st.session_state.models_trained:
                    models = st.session_state.trained_models
                    model_results = {}
                    
                    for target, target_models in models.items():
                        # Evaluate models
                        models_predictions = {}
                        for model_type, model_info in target_models.items():
                            if 'y_test' in model_info and 'y_pred_test' in model_info:
                                models_predictions[model_type] = (model_info['y_test'], model_info['y_pred_test'])
                        
                        if models_predictions:
                            model_results[target] = evaluate_multiple_models(models_predictions, target)
                
                with st.spinner("Generating comprehensive report..."):
                    try:
                        # Prepare simplified predictions for report
                        report_predictions = {}
                        if predictions:
                            for target, target_preds in predictions.items():
                                if target_preds:
                                    # Use best model's predictions
                                    best_model = list(target_preds.keys())[0]
                                    report_predictions[target] = target_preds[best_model]
                        
                        report_path = generate_report(
                            data=data,
                            predictions=report_predictions,
                            model_results=model_results,
                            visuals=visuals,
                            output_path=f"pune_climate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        )
                        
                        st.success(f"✅ Report generated successfully!")
                        
                        # Provide download link
                        if os.path.exists(report_path):
                            with open(report_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Report",
                                    data=file.read(),
                                    file_name=os.path.basename(report_path),
                                    mime="application/pdf"
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {e}")
            
            # Report preview/summary
            if st.session_state.data_loaded:
                st.subheader("📋 Report Preview")
                
                data = st.session_state.climate_data
                
                # Basic statistics for preview
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Data Records", f"{len(data):,}")
                    st.metric("📅 Years Covered", f"{data['date'].max().year - data['date'].min().year + 1}")
                
                with col2:
                    if 'temperature' in data.columns:
                        temp_trend = data.groupby('year')['temperature'].mean().diff().mean() if 'year' in data.columns else 0
                        st.metric("🌡️ Temperature Trend", f"{temp_trend:.3f}°C/year")
                    
                    if 'aqi' in data.columns:
                        st.metric("💨 Average AQI", f"{data['aqi'].mean():.0f}")
                
                with col3:
                    if st.session_state.models_trained:
                        total_models = sum(len(models) for models in st.session_state.trained_models.values())
                        st.metric("🤖 Models Trained", total_models)
                    
                    if st.session_state.predictions_generated:
                        st.metric("🔮 Predictions", "Available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌍 Pune Climate Prediction System | Built with Streamlit & Advanced ML Models</p>
        <p>📊 Data-driven insights for climate adaptation and planning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()