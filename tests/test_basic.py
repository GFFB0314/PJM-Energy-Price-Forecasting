"""Unit tests for model pipeline structure and feature preparation."""

import numpy as np
import pandas as pd

from src.model import get_pipeline, prepare_features


def test_pipeline_structure():
    """Ensure pipeline is GradientBoostingRegressor and contains expected steps."""
    pipeline, param_grid = get_pipeline()

    # Check that 'gbr' step exists in the pipeline
    assert "gbr" in pipeline.named_steps
    assert type(pipeline.named_steps["gbr"]).__name__ == "GradientBoostingRegressor"

    # Check that param_grid contains GBR specific hyperparams
    assert "gbr__n_estimators" in param_grid
    assert "gbr__learning_rate" in param_grid
    assert "gbr__max_depth" in param_grid


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
        "hour_day_x_day_week": np.random.rand(100),  # Noise column
        "price_actual": np.random.rand(100),
    }
    df = pd.DataFrame(data)

    X_train, X_test, _y_train, _y_test = prepare_features(df)

    # Check shape (80/20 split)
    assert len(X_train) == 80
    assert len(X_test) == 20

    # Check noise column is gone
    assert "hour_day_x_day_week" not in X_train.columns

    # Check target separation
    assert "price_actual" not in X_train.columns
