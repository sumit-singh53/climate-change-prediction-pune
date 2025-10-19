"""
Modular ML Pipeline Backend for Climate Prediction
Comprehensive backend system for Pune climate forecasting
"""

__version__ = "1.0.0"
__author__ = "Climate Prediction Team"

from .data_collector import fetch_city_data
from .data_preprocessor import clean_and_preprocess
from .model_trainer import train_model
from .predictor import predict_future
from .evaluator import evaluate_model
from .visualizer import generate_visuals
from .report_generator import generate_report
from .insights_generator import generate_insights
from .risk_calculator import calculate_climate_risk

__all__ = [
    'fetch_city_data',
    'clean_and_preprocess', 
    'train_model',
    'predict_future',
    'evaluate_model',
    'generate_visuals',
    'generate_report',
    'generate_insights',
    'calculate_climate_risk'
]