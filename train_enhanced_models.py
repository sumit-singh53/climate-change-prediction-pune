#!/usr/bin/env python3
"""
Enhanced Model Training Script
Train high-accuracy ML models for climate and AQI prediction
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from enhanced_ml_models import EnhancedMLModels
from config import MODEL_CONFIG

def main():
    """Train enhanced models for all target variables"""
    
    print("🚀 Enhanced Climate & AQI Model Training")
    print("=" * 60)
    print(f"📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize enhanced ML models
    ml_models = EnhancedMLModels()
    
    # Target variables to train
    target_variables = MODEL_CONFIG['target_variables']
    
    print(f"🎯 Training models for {len(target_variables)} target variables:")
    for var in target_variables:
        print(f"   - {var}")
    print()
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs/models', exist_ok=True)
    
    # Training results
    all_results = {}
    
    for i, target_var in enumerate(target_variables, 1):
        print(f"🧠 Training models for {target_var} ({i}/{len(target_variables)})")
        print("-" * 40)
        
        try:
            # Train enhanced models
            results = ml_models.train_enhanced_models(target_var)
            all_results[target_var] = results
            
            # Save models
            model_filepath = f'outputs/models/{target_var}_enhanced_models.joblib'
            ml_models.save_models(target_var, model_filepath)
            
            print(f"✅ Models saved to {model_filepath}")
            print()
            
            # Display best model performance
            best_model = max(results['performances'].items(), key=lambda x: x[1]['r2'])
            best_name, best_metrics = best_model
            
            print(f"🏆 Best model for {target_var}: {best_name.upper()}")
            print(f"   📊 Accuracy: {best_metrics['accuracy']:.1f}%")
            print(f"   📉 RMSE: {best_metrics['rmse']:.3f}")
            print(f"   📈 R²: {best_metrics['r2']:.3f}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to train models for {target_var}: {str(e)}")
            print()
            continue
    
    # Summary report
    print("=" * 60)
    print("📊 TRAINING SUMMARY REPORT")
    print("=" * 60)
    
    if all_results:
        # Create summary DataFrame
        summary_data = []
        for target_var, results in all_results.items():
            for model_name, metrics in results['performances'].items():
                summary_data.append({
                    'Target Variable': target_var,
                    'Model': model_name.upper(),
                    'Accuracy (%)': f"{metrics['accuracy']:.1f}%",
                    'RMSE': f"{metrics['rmse']:.3f}",
                    'R²': f"{metrics['r2']:.3f}",
                    'MAE': f"{metrics['mae']:.3f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary report
        summary_df.to_csv('outputs/models/training_summary.csv', index=False)
        print(f"\n📄 Summary report saved to outputs/models/training_summary.csv")
        
        # Best models for each variable
        print("\n🏆 BEST MODELS BY VARIABLE:")
        print("-" * 30)
        for target_var, results in all_results.items():
            best_model = max(results['performances'].items(), key=lambda x: x[1]['r2'])
            best_name, best_metrics = best_model
            print(f"{target_var:12} | {best_name.upper():12} | {best_metrics['accuracy']:6.1f}% | R²: {best_metrics['r2']:.3f}")
        
        # Feature importance
        print("\n🔍 TOP FEATURES BY VARIABLE:")
        print("-" * 30)
        for target_var, results in all_results.items():
            if 'feature_importance' in results and results['feature_importance']:
                print(f"\n{target_var.upper()}:")
                # Get feature importance from best performing tree-based model
                for model_name, importance_list in results['feature_importance'].items():
                    if importance_list:  # If there are features
                        print(f"  {model_name.upper()} top 5 features:")
                        for feature, importance in importance_list[:5]:
                            print(f"    - {feature}: {importance:.3f}")
                        break
    
    else:
        print("❌ No models were successfully trained.")
    
    print("\n" + "=" * 60)
    print(f"🎉 Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Usage instructions
    print("\n💡 NEXT STEPS:")
    print("1. 🌐 Run the enhanced dashboard: python src/enhanced_dashboard.py")
    print("2. 📊 Or use streamlit: streamlit run src/enhanced_dashboard.py")
    print("3. 🔮 Generate predictions using the trained models")
    print("4. 📈 Monitor model performance and retrain as needed")

if __name__ == "__main__":
    main()