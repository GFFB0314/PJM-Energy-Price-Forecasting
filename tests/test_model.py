"""Unit tests for model training and simulation pipeline."""

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.model import train_and_evaluate, run_pnl_simulation


def generate_mock_data():
    """Creates a small mock dataset for testing training loop."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    X = pd.DataFrame(
        {
            "temp_c": np.random.rand(100),
            "wind_kph": np.random.rand(100),
            "solar_radiation": np.random.rand(100),
            "price_1h_ago": np.random.rand(100),
            "price_24h_ago": np.random.rand(100),
            "avg_price_last_24h": np.random.rand(100),
            "hour_of_day": np.random.randint(0, 24, 100),
            "day_of_week": np.random.randint(0, 7, 100),
            "month": np.random.randint(1, 12, 100),
        },
        index=dates,
    )

    y = pd.Series(np.random.rand(100) * 100, index=dates, name="price_actual")
    return X, y


def test_train_and_evaluate_mocked():
    """Test the training loop using mock GridSearchCV to speed up execution."""
    X, y = generate_mock_data()
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    with patch("src.model.GridSearchCV") as MockGridSearch:
        mock_model = MagicMock()
        mock_model.best_score_ = -10.5  # mock negative RMSE
        mock_model.best_params_ = {"gbr__n_estimators": 10}

        # Add a mock predict method return
        mock_model.best_estimator_.predict.return_value = np.random.rand(20) * 100

        MockGridSearch.return_value = mock_model

        best_model, leaderboard = train_and_evaluate(X_train, y_train, X_test, y_test)

        # Verify GridSearchCV was called
        MockGridSearch.assert_called_once()
        mock_model.fit.assert_called_once_with(X_train, y_train)

        # Verify outputs
        assert best_model == mock_model.best_estimator_
        assert not leaderboard.empty
        assert leaderboard.iloc[0]["Model"] == "GradientBoostingRegressor"
        assert leaderboard.iloc[0]["CV_RMSE"] == 10.5


def test_run_pnl_simulation():
    """Test PnL simulation output format without error."""
    X, y = generate_mock_data()
    X_test, y_test = X.iloc[:24], y.iloc[:24]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.rand(24) * 100

    with patch("src.model.logger") as mock_logger:
        run_pnl_simulation(mock_model, X_test, y_test)

        # Check that it reached the final logging statements
        mock_logger.info.assert_any_call("Running PnL Simulation...")
        mock_logger.info.assert_any_call("-" * 30)
