import pytest
import pandas as pd
import numpy as np
from src.model import prepare_features, get_pipelines

def test_pipeline_structure():
    """Ensure pipelines dict contains the expected models."""
    pipelines, _ = get_pipelines()
    expected_models = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"]
    for model in expected_models:
        assert model in pipelines, f"{model} is missing from pipelines"

def test_feature_preparation():
    """Test if split works and noise column is dropped."""
    # Create dummy data
    data = {
        "temp_c": np.random.rand(100),
        "wind_kph": np.random.rand(100),
        "solar_radiation": np.random.rand(100),
        "price_1h_ago": np.random.rand(100),
        "price_24h_ago": np.random.rand(100),
        "avg_price_last_24h": np.random.rand(100),
        "hour_of_day": np.random.randint(0, 24, 100),
        "day_of_week": np.random.randint(0, 7, 100),
        "month": np.random.randint(1, 12, 100),
        "hour_day_x_day_week": np.random.rand(100), # Noise column
        "price_actual": np.random.rand(100)
    }
    df = pd.DataFrame(data)
    
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    # Check shape (80/20 split)
    assert len(X_train) == 80
    assert len(X_test) == 20
    
    # Check noise column is gone
    assert "hour_day_x_day_week" not in X_train.columns
    
    # Check target separation
    assert "price_actual" not in X_train.columns