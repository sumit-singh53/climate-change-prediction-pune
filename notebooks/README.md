# Jupyter Notebooks

This directory contains Jupyter notebooks for data analysis, model development, and research.

## Available Notebooks

### 1. EDA.ipynb
**Exploratory Data Analysis**
- Data distribution analysis
- Correlation studies
- Seasonal patterns
- Location-wise comparisons

### 2. Modeling.ipynb
**Model Development**
- Feature engineering experiments
- Model comparison studies
- Hyperparameter tuning
- Performance evaluation

### 3. Modeling_phase6.ipynb
**Advanced Modeling**
- Ensemble model development
- Deep learning experiments
- Time series analysis
- Prediction accuracy improvements

## Getting Started

1. **Install Jupyter**
   ```bash
   pip install jupyter notebook
   ```

2. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

3. **Open notebooks**
   Navigate to the notebooks directory and open any `.ipynb` file

## Requirements

All notebooks use the same dependencies as the main project:
- pandas, numpy for data manipulation
- matplotlib, seaborn, plotly for visualization
- scikit-learn, xgboost, tensorflow for modeling
- See `requirements.txt` for complete list

## Data Access

Notebooks automatically access data from:
- `../data/` directory for raw and processed data
- SQLite database for operational data
- Live APIs for real-time data (when available)

## Best Practices

- **Version Control**: Notebooks are included in git with outputs cleared
- **Documentation**: Each notebook includes markdown explanations
- **Reproducibility**: Set random seeds for consistent results
- **Performance**: Use data sampling for large datasets during development