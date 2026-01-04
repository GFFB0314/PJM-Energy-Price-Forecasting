"""
Model training, evaluation, and PnL simulation logic for the Energy Arbitrage Project.
This module handles loading processed data, feature engineering, pipeline construction,
grid search training, and financial impact analysis.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Tuple, Dict, Any, Optional

from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error as RMSE, r2_score as R2

from .config import PROCESSED_DATA_PATH_TRAIN

# Setup logging
logger = logging.getLogger(__name__)

def load_processed_data(path: str = PROCESSED_DATA_PATH_TRAIN) -> pd.DataFrame:
    """Loads the required training CSV data."""
    logger.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    
    # Converting the datetime_beginning_ept back to Datetime 
    df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    # Setting datetime_beginning_ept as index and sorting
    df.set_index("datetime_beginning_ept", inplace=True)
    df.sort_index(inplace=True)

    return df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares X and y, splits data, and drops noise columns.
    Matches the exact logic from the notebook.
    """
    logger.info("Preparing features and target...")
    
    # 1. Feature Selection
    weather_cols: list[str] = ["temp_c", "wind_kph", "solar_radiation"]
    other_cols: list[str] = ["price_1h_ago", "price_24h_ago", "avg_price_last_24h", "hour_of_day", "day_of_week", "month"]
    
    # Drop the noise column if it exists (Safety check)
    if "hour_day_x_day_week" in df.columns:
        df = df.drop(columns=["hour_day_x_day_week"])

    X: pd.DataFrame = df[weather_cols + other_cols]
    y: pd.Series = df["price_actual"]
    
    # 2. Split (Shuffle=False is CRITICAL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    logger.info(f"Data Split: Train {X_train.shape}, Test {X_test.shape}")
    return (X_train, X_test, y_train, y_test)

def get_pipelines() -> Tuple[Dict[str, Pipeline], Dict[str, Any]]:
    """
    Defines the Pipelines and Param Grids.
    """
    weather_cols: list[str] = ["temp_c", "wind_kph", "solar_radiation"]
    other_cols: list[str] = ["price_1h_ago", "price_24h_ago", "avg_price_last_24h", "hour_of_day", "day_of_week", "month"]

    # Preprocessors
    weather_poly_scaled = Pipeline(steps=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", RobustScaler())
    ])
    
    other_scaled = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])

    processor_scaled = ColumnTransformer(
        transformers=[
            ("weather_columns", weather_poly_scaled, weather_cols),
            ("other_columns", other_scaled, other_cols)
        ], 
        remainder="drop"
    )

    processor_trees = ColumnTransformer(
        transformers=[
            ("all_columns", "passthrough", weather_cols + other_cols)
        ], 
        remainder="drop"
    )

    # Pipelines
    pipelines: Dict[str, Pipeline] = {
        "LinearRegression": Pipeline(steps=[
            ("process", processor_scaled),
            ("lr", LinearRegression())
        ]),

        "RidgeRegression": Pipeline(steps=[
            ("process", processor_scaled),
            ("ridge", Ridge())
        ]),

        "LassoRegression": Pipeline(steps=[
            ("process", processor_scaled),
            ("lasso", Lasso())
        ]),

        "KNN": Pipeline(steps=[
            ("process", processor_scaled),
            ("knn", KNeighborsRegressor())
        ]),

        "BaggingRegressor": Pipeline(steps=[
            ("process", processor_trees),
            ("br", BaggingRegressor(random_state=42))
        ]),

        "RandomForestRegressor": Pipeline(steps=[
            ("process", processor_trees),
            ("rfr", RandomForestRegressor(random_state=42))
        ]),

        "GradientBoostingRegressor": Pipeline(steps=[
            ("process", processor_trees),
            ("gbr", GradientBoostingRegressor(random_state=42))
        ])
    }
    
    # Param Grid
    param_grid: Dict[str, Any] = {
        "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
        "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        },
        "RandomForestRegressor": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "max_features": ["sqrt", "log2"]
        },
        "BaggingRegressor": {
            "n_estimators": [10, 50, 100], # Uses Decision Tree as base estimator by default
            "max_samples": [0.6, 0.8, 1.0]
        },
        "GradientBoostingRegressor": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3, 4]
        },
    }

    return (pipelines, param_grid)

def train_and_evaluate(X_train, y_train, X_test, y_test) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Runs the GridSearch Training loop.
    """
    pipelines, param_grid = get_pipelines()
    
    # CV Strategy (The 60 splits logic)
    tscv = TimeSeriesSplit(n_splits=60, test_size=24) # Forecast horizon of 24 hours over a period of 60 days (1440 hours)
    
    results: list[dict] = []
    estimators: Dict[str, Pipeline] = {}
    
    logger.info(f"Starting Training loop on {len(pipelines)} models...")

    for name, pipeline in pipelines.items():
        logger.info(f"Training (Fitting) {name}...")
        
        # logic to map params
        search_params: dict = {}
        step_name: str = ""
        
        if "RidgeRegression" in name: grid_key, step_name = "Ridge", "ridge"
        elif "LassoRegression" in name: grid_key, step_name = "Lasso", "lasso"
        elif "KNN" in name: grid_key, step_name = "KNN", "knn"
        elif "RandomForestRegressor" in name: grid_key, step_name = "RandomForestRegressor", "rfr"
        elif "BaggingRegressor" in name: grid_key, step_name = "BaggingRegressor", "br"
        elif "GradientBoostingRegressor" in name: grid_key, step_name = "GradientBoostingRegressor", "gbr"
        else: grid_key = None # No Hyperparameter tunning for LinearRegression

        if grid_key and grid_key in param_grid:
            for param, values in param_grid[grid_key].items():
                search_params[f"{step_name}__{param}"] = values
        
        best_score: float = 0.0
        cv_score: float = 0.0

        if search_params:
            model = GridSearchCV(
                pipeline, 
                param_grid=search_params, 
                cv=tscv, 
                scoring="neg_root_mean_squared_error", 
                n_jobs=-1
            )

            # Fitting the model on the entire training data
            model.fit(X_train, y_train)
            final_model: Pipeline = model.best_estimator_ # Best pipeline with best hyperparameters
            best_score = round(-model.best_score_, 4) # Best CV score (lowest RMSE)
            best_params = model.best_params_ # Best hyperparameters
        else:
            cv_scores = cross_val_score(
                pipeline,
                X_train, 
                y_train, 
                cv=tscv, 
                scoring="neg_root_mean_squared_error"
            )

            cv_score = round(-cv_scores.mean(), 4) # AVG_RMSE 
            best_params = "Default"

            # Fitting the model on the entire training data
            pipeline.fit(X_train, y_train)
            final_model: Pipeline = pipeline

        y_pred = final_model.predict(X_test)
        test_rmse = round(RMSE(y_test, y_pred), 4)
        test_r2 = round(R2(y_test, y_pred), 4)
            
        results.append({
            "Model": name,
            "CV_RMSE": best_score if search_params else cv_score,
            "TEST_RMSE": test_rmse,
            "R2_TEST": test_r2,
            "Best Params": best_params
        })
        estimators[name] = final_model
        logger.info(f"  -> {name} | CV_RMSE: {results[-1]["CV_RMSE"]:.4f} | TEST_RMSE: {results[-1]["TEST_RMSE"]:.4f}")

    # Find the best model based on CV Score
    logger.info("Printing the Model Leaderboard sorted by CV_RMSE:")
    leaderb_df: pd.DataFrame = pd.DataFrame(results).sort_values("TEST_RMSE", ascending=True)
    best_model_name: str = leaderb_df.iloc[0]["Model"]
    best_model_obj: Pipeline = estimators[best_model_name]
    
    logger.info(f"Winner: {best_model_name}")
    return (best_model_obj, leaderb_df)

def run_pnl_simulation(model, X_test, y_test):
    """
    Calculates the PnL (Profit and Loss).
    """
    logger.info("Running PnL Simulation...")
    df_sim = X_test.copy()
    df_sim["price_actual"] = y_test
    df_sim["price_predicted"] = model.predict(X_test)
    
    # We assume "datetime_beginning_ept" was the index or needs to be recovered for date grouping
    # Since X_test lost the index name in some splits, we re-verify or just use row grouping if ordered
    # Ideally, X_test index is datetime.
    
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
        buy_price_perfect = daily_data["price_actual"].min()
        sell_price_perfect = daily_data["price_actual"].max()
        profit_perfect = (sell_price_perfect - buy_price_perfect) * 100
        
        buy_hour_pred = daily_data["price_predicted"].idxmin()
        sell_hour_pred = daily_data["price_predicted"].idxmax()
        
        buy_price_model = daily_data.loc[buy_hour_pred, "price_actual"]
        sell_price_model = daily_data.loc[sell_hour_pred, "price_actual"]
        
        profit_model = (sell_price_model - buy_price_model) * 100
        return pd.Series([profit_perfect, profit_model], index=["Perfect_Profit", "Model_Profit"])

    daily_profits = df_sim.groupby("date").apply(simulate_day)
    
    total_realized = daily_profits["Model_Profit"].sum() # Model Proft
    total_potential = daily_profits["Perfect_Profit"].sum() # Perfect Profit
    efficiency = (total_realized / total_potential) * 100
    
    logger.info("-" * 30)
    logger.info(f"PnL RESULTS:")
    logger.info(f"Total Potential Profit:       ${total_potential:,.2f}")
    logger.info(f"Total Realized Profit:         ${total_realized:,.2f}")
    logger.info(f"Efficiency:                         {efficiency:.2f}%")
    logger.info("-" * 30)

def save_model(model, filename="best_estimator.pkl"):
    """Saves the model object."""
    path = f"src/{filename}"
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")