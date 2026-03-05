"""
Model training, evaluation, and PnL simulation logic for the Energy Arbitrage Project.
This module handles loading processed data, feature engineering, pipeline construction,
grid search training, and financial impact analysis.
"""

from typing import Any, Dict, Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score as R2
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline


from .config import PROCESSED_DATA_PATH
from .logging_utils import get_logger

# Setup logging
logger = get_logger(__name__)


def load_processed_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Loads the required training CSV data."""
    logger.info("Loading data from %s...", path)
    df = pd.read_csv(path)

    # Converting the datetime_beginning_ept back to Datetime
    df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    # Setting datetime_beginning_ept as index and sorting
    df.set_index("datetime_beginning_ept", inplace=True)
    df.sort_index(inplace=True)

    return df


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares X and y, splits data, and drops noise columns.
    Matches the exact logic from the notebook.
    """
    logger.info("Preparing features and target...")

    # 1. Feature Selection
    weather_cols: list[str] = ["temp_c", "wind_kph", "solar_radiation"]
    other_cols: list[str] = [
        "price_1h_ago",
        "price_24h_ago",
        "avg_price_last_24h",
        "hour_of_day",
        "day_of_week",
        "month",
    ]

    # Drop the noise column if it exists (Safety check)
    if "hour_day_x_day_week" in df.columns:
        df = df.drop(columns=["hour_day_x_day_week"])

    X: pd.DataFrame = df[weather_cols + other_cols]
    y: pd.Series = df["price_actual"]

    # 2. Split (Shuffle=False is CRITICAL)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    logger.info("Data Split: Train %s, Test %s", X_train.shape, X_test.shape)
    return (X_train, X_test, y_train, y_test)


def get_pipeline() -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Defines the Pipeline and Param Grid for Gradient Boosting.
    """
    weather_cols: list[str] = ["temp_c", "wind_kph", "solar_radiation"]
    other_cols: list[str] = [
        "price_1h_ago",
        "price_24h_ago",
        "avg_price_last_24h",
        "hour_of_day",
        "day_of_week",
        "month",
    ]

    # Preprocessors
    processor_trees = ColumnTransformer(
        transformers=[("all_columns", "passthrough", weather_cols + other_cols)],
        remainder="drop",
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("process", processor_trees),
            ("gbr", GradientBoostingRegressor(random_state=42)),
        ]
    )

    # Param Grid
    param_grid: Dict[str, Any] = {
        "gbr__n_estimators": [100, 200],
        "gbr__learning_rate": [0.05, 0.1],
        "gbr__max_depth": [2, 3, 4],
    }

    return (pipeline, param_grid)


def train_and_evaluate(
    X_train, y_train, X_test, y_test
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Runs the GridSearch Training logic.
    """
    pipeline, param_grid = get_pipeline()

    # CV Strategy (The 60 splits logic)
    tscv = TimeSeriesSplit(
        n_splits=60, test_size=24
    )  # Forecast horizon of 24 hours over a period of 60 days (1440 hours)

    logger.info("Starting Training for GradientBoostingRegressor...")

    model = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    # Fitting the model on the entire training data
    model.fit(X_train, y_train)
    final_model: Pipeline = (
        model.best_estimator_
    )  # Best pipeline with best hyperparameters
    best_score = round(-model.best_score_, 4)  # Best CV score (lowest RMSE)
    best_params = model.best_params_  # Best hyperparameters

    y_pred = final_model.predict(X_test)
    test_rmse = round(RMSE(y_test, y_pred), 4)
    test_r2 = round(R2(y_test, y_pred), 4)

    results = []
    results.append(
        {
            "Model": "GradientBoostingRegressor",
            "CV_RMSE": best_score,
            "TEST_RMSE": test_rmse,
            "R2_TEST": test_r2,
            "Best Params": best_params,
        }
    )

    logger.info(
        "  -> GradientBoostingRegressor | CV_RMSE: %.4f | TEST_RMSE: %.4f",
        best_score,
        test_rmse,
    )

    # Create the leaderboard dataframe
    leaderb_df: pd.DataFrame = pd.DataFrame(results)

    logger.info("Winner: GradientBoostingRegressor")
    return (final_model, leaderb_df)


def run_pnl_simulation(model, X_test, y_test):
    """
    Calculates the PnL (Profit and Loss).
    """
    logger.info("Running PnL Simulation...")
    df_sim = X_test.copy()
    df_sim["price_actual"] = y_test
    df_sim["price_predicted"] = model.predict(X_test)

    # We assume "datetime_beginning_ept" was the index or needs to be
    # recovered for date grouping.
    # Since X_test lost the index name in some splits, we re-verify or just
    # use row grouping if ordered. Ideally, X_test index is datetime.

    # Simple hack to group by chunks of 24 if index is lost,
    # BUT assuming X_test kept its datetime index from the ETL:
    try:
        df_sim["date"] = df_sim.index.date
    except AttributeError:
        # If index was reset, we need to handle it.
        # For now, we assume the index is correct as per notebook
        logger.warning("Index is not datetime. PnL grouping might fail.")
        return

    def simulate_day(daily_data):
        """Simulates a day of trading."""
        buy_price_perfect = daily_data["price_actual"].min()
        sell_price_perfect = daily_data["price_actual"].max()
        profit_perfect = (sell_price_perfect - buy_price_perfect) * 100

        buy_hour_pred = daily_data["price_predicted"].idxmin()
        sell_hour_pred = daily_data["price_predicted"].idxmax()

        buy_price_model = daily_data.loc[buy_hour_pred, "price_actual"]
        sell_price_model = daily_data.loc[sell_hour_pred, "price_actual"]

        profit_model = (sell_price_model - buy_price_model) * 100
        return pd.Series(
            [profit_perfect, profit_model], index=["Perfect_Profit", "Model_Profit"]
        )

    daily_profits = df_sim.groupby("date").apply(simulate_day, include_groups=False)

    total_realized = daily_profits["Model_Profit"].sum()  # Model Proft
    total_potential = daily_profits["Perfect_Profit"].sum()  # Perfect Profit
    efficiency = (total_realized / total_potential) * 100

    logger.info("-" * 30)
    logger.info("PnL RESULTS:")
    logger.info("Total Potential Profit:       $%.2f", total_potential)
    logger.info("Total Realized Profit:         $%.2f", total_realized)
    logger.info("Efficiency:                         %.1f%%", efficiency)
    logger.info("-" * 30)


def save_model(model, filename="best_estimator.pkl"):
    """Saves the model object."""
    path = f"src/{filename}"
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
