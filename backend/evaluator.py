"""
Model Evaluation Module
Returns RMSE, MAE, and R² metrics for model performance assessment
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation system"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.comparison_results = {}
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic regression metrics"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'mse': mean_squared_error(y_true, y_pred)
        }
        
        # Additional metrics
        metrics['mean_error'] = np.mean(y_pred - y_true)
        metrics['std_error'] = np.std(y_pred - y_true)
        metrics['max_error'] = np.max(np.abs(y_pred - y_true))
        
        return metrics
    
    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate advanced statistical metrics"""
        residuals = y_pred - y_true
        
        # Correlation coefficient
        correlation, p_value = stats.pearsonr(y_true, y_pred)
        
        # Normalized metrics
        rmse_normalized = np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))
        mae_normalized = mean_absolute_error(y_true, y_pred) / (np.max(y_true) - np.min(y_true))
        
        # Directional accuracy (for time series)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction)
        else:
            directional_accuracy = np.nan
        
        advanced_metrics = {
            'correlation': correlation,
            'correlation_p_value': p_value,
            'rmse_normalized': rmse_normalized,
            'mae_normalized': mae_normalized,
            'directional_accuracy': directional_accuracy,
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals)
        }
        
        return advanced_metrics
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        print(f"📊 Evaluating {model_name} performance...")
        
        # Basic metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        
        # Advanced metrics
        advanced_metrics = self.calculate_advanced_metrics(y_true, y_pred)
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **advanced_metrics}
        
        # Performance classification
        r2_score_val = all_metrics['r2']
        if r2_score_val >= 0.9:
            performance_grade = "Excellent"
        elif r2_score_val >= 0.8:
            performance_grade = "Good"
        elif r2_score_val >= 0.6:
            performance_grade = "Fair"
        else:
            performance_grade = "Poor"
        
        evaluation_result = {
            'model_name': model_name,
            'metrics': all_metrics,
            'performance_grade': performance_grade,
            'sample_size': len(y_true),
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Store results
        self.evaluation_results[model_name] = evaluation_result
        
        print(f"✅ {model_name} evaluation completed - Grade: {performance_grade}")
        return evaluation_result
    
    def compare_models(self, models_results: Dict[str, Dict[str, Any]], 
                      metric: str = 'r2') -> pd.DataFrame:
        """Compare multiple models based on a specific metric"""
        print(f"🔍 Comparing models based on {metric}...")
        
        comparison_data = []
        
        for model_name, result in models_results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                comparison_data.append({
                    'model': model_name,
                    'r2': metrics.get('r2', np.nan),
                    'rmse': metrics.get('rmse', np.nan),
                    'mae': metrics.get('mae', np.nan),
                    'mape': metrics.get('mape', np.nan),
                    'correlation': metrics.get('correlation', np.nan),
                    'performance_grade': result.get('performance_grade', 'Unknown')
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by the specified metric (higher is better for r2, correlation; lower is better for errors)
        if metric in ['r2', 'correlation', 'directional_accuracy']:
            comparison_df = comparison_df.sort_values(metric, ascending=False)
        else:
            comparison_df = comparison_df.sort_values(metric, ascending=True)
        
        # Add ranking
        comparison_df['rank'] = range(1, len(comparison_df) + 1)
        
        self.comparison_results[metric] = comparison_df
        
        print(f"✅ Model comparison completed")
        return comparison_df
    
    def generate_evaluation_report(self, model_results: Dict[str, Any], 
                                 target_name: str = "Target") -> str:
        """Generate a comprehensive evaluation report"""
        report = f"""
🎯 MODEL EVALUATION REPORT - {target_name.upper()}
{'='*60}

📊 PERFORMANCE SUMMARY:
   Model: {model_results['model_name']}
   Grade: {model_results['performance_grade']}
   Sample Size: {model_results['sample_size']:,}
   
📈 KEY METRICS:
   R² Score: {model_results['metrics']['r2']:.4f}
   RMSE: {model_results['metrics']['rmse']:.4f}
   MAE: {model_results['metrics']['mae']:.4f}
   MAPE: {model_results['metrics']['mape']:.2f}%
   
🔍 DETAILED ANALYSIS:
   Correlation: {model_results['metrics']['correlation']:.4f}
   Mean Error: {model_results['metrics']['mean_error']:.4f}
   Max Error: {model_results['metrics']['max_error']:.4f}
   Directional Accuracy: {model_results['metrics'].get('directional_accuracy', 'N/A'):.2%}
   
📊 RESIDUAL STATISTICS:
   Mean: {model_results['metrics']['residual_mean']:.4f}
   Std Dev: {model_results['metrics']['residual_std']:.4f}
   Skewness: {model_results['metrics']['residual_skewness']:.4f}
   Kurtosis: {model_results['metrics']['residual_kurtosis']:.4f}

💡 INTERPRETATION:
   - R² of {model_results['metrics']['r2']:.3f} means the model explains {model_results['metrics']['r2']*100:.1f}% of variance
   - MAPE of {model_results['metrics']['mape']:.1f}% indicates average prediction error
   - Performance grade: {model_results['performance_grade']}
        """
        
        return report


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  model_name: str = "Model") -> Dict[str, Any]:
    """
    Main function to evaluate model performance
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model being evaluated
    
    Returns:
        Dictionary containing comprehensive evaluation results
    """
    print(f"🔬 EVALUATING MODEL: {model_name.upper()}")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    
    # Perform evaluation
    results = evaluator.evaluate_model_performance(y_true, y_pred, model_name)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, model_name)
    results['report'] = report
    
    print(report)
    
    return results


def evaluate_multiple_models(models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                           target_name: str = "Target") -> Dict[str, Any]:
    """
    Evaluate multiple models and compare their performance
    
    Args:
        models_predictions: Dict with model_name: (y_true, y_pred) pairs
        target_name: Name of the target variable
    
    Returns:
        Dictionary containing all evaluation results and comparisons
    """
    print(f"🚀 EVALUATING MULTIPLE MODELS FOR {target_name.upper()}")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    all_results = {}
    
    # Evaluate each model
    for model_name, (y_true, y_pred) in models_predictions.items():
        try:
            result = evaluator.evaluate_model_performance(y_true, y_pred, model_name)
            all_results[model_name] = result
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Compare models
    if len(all_results) > 1:
        comparison_df = evaluator.compare_models(all_results)
        
        print(f"\n🏆 MODEL RANKING (by R² Score):")
        print("-" * 40)
        for _, row in comparison_df.iterrows():
            print(f"   {row['rank']}. {row['model']}: R²={row['r2']:.3f}, RMSE={row['rmse']:.3f}")
        
        # Find best model
        best_model = comparison_df.iloc[0]['model']
        print(f"\n🥇 BEST MODEL: {best_model}")
        
        return {
            'individual_results': all_results,
            'comparison': comparison_df,
            'best_model': best_model,
            'target_name': target_name
        }
    else:
        return {
            'individual_results': all_results,
            'target_name': target_name
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    y_true = np.random.normal(25, 5, 1000)  # True temperature values
    y_pred1 = y_true + np.random.normal(0, 2, 1000)  # Model 1 predictions
    y_pred2 = y_true + np.random.normal(0, 3, 1000)  # Model 2 predictions
    
    # Test single model evaluation
    print("Testing single model evaluation...")
    result = evaluate_model(y_true, y_pred1, "Linear Regression")
    
    # Test multiple model evaluation
    print("\n" + "="*60)
    print("Testing multiple model evaluation...")
    models_predictions = {
        'Linear Regression': (y_true, y_pred1),
        'Random Forest': (y_true, y_pred2)
    }
    
    multi_results = evaluate_multiple_models(models_predictions, "Temperature")
    
    print("\n✅ Model evaluation tests completed!")