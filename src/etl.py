"""ETL (Extract, Transform, Load) functions for the Energy Arbitrage Project."""

import logging
import os
from typing import Dict, Any

import pandas as pd
import requests

from .config import (
    DB_CONNECTION_STRING,
    WEATHER_API_URL,
    WEATHER_API_PARAMS,
    PROCESSED_DATA_PATH,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_weather_data(
    url: str = WEATHER_API_URL, params: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Fetches historical weather data from Open-Meteo API.
    """
    if params is None:
        params = WEATHER_API_PARAMS
    logger.info("Fetching weather data from %s...", url)
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hourly = data["hourly"]
        df_weather = pd.DataFrame(
            {
                "time_ept": hourly["time"],
                "temp_c": hourly["temperature_2m"],
                "wind_kph": hourly["wind_speed_10m"],
                "solar_radiation": hourly["shortwave_radiation"],
            }
        )

        # Convert to datetime
        df_weather["time_ept"] = pd.to_datetime(df_weather["time_ept"])

        logger.info("Successfully downloaded %d rows of weather data.", len(df_weather))
        return df_weather

    except requests.exceptions.RequestException as e:
        logger.error("Error fetching weather data: %s", e)
        raise


def fetch_price_data(connection_string: str = DB_CONNECTION_STRING) -> pd.DataFrame:
    """
    Fetches price and feature data from the PostgreSQL database.
    """
    logger.info("Connecting to Database to fetch price data...")
    query = "SELECT * FROM pjm_market.features_v"

    try:
        df_prices = pd.read_sql(query, con=connection_string)

        # Ensure timestamps are datetime
        if "datetime_beginning_ept" in df_prices.columns:
            df_prices["datetime_beginning_ept"] = pd.to_datetime(
                df_prices["datetime_beginning_ept"]
            )

        logger.info("Loaded %d rows from SQL.", len(df_prices))
        return df_prices
    except Exception as e:
        logger.error("Error fetching data from database: %s", e)
        raise


def merge_data(df_prices: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merges price data with weather data on timestamp.
    """
    logger.info("Merging price and weather data...")

    # Merge (Inner Join)
    df_merged = df_prices.merge(
        df_weather, left_on="datetime_beginning_ept", right_on="time_ept", how="inner"
    )

    # Drop duplicate time column
    if "time_ept" in df_merged.columns:
        df_merged = df_merged.drop("time_ept", axis=1)

    logger.info("Final Dataset Shape: %s", df_merged.shape)
    return df_merged


def save_data(df: pd.DataFrame, output_path: str = PROCESSED_DATA_PATH) -> None:
    """
    Saves the processed dataframe to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Data successfully saved to %s", output_path)
    except Exception as exc:
        logger.error("Error saving data: %s", exc)
        raise


def run_etl_pipeline():
    """
    Runs the full ETL pipeline.
    """
    logger.info("Starting ETL Pipeline...")

    try:
        df_weather = fetch_weather_data()
        df_prices = fetch_price_data()
        df_merged = merge_data(df_prices, df_weather)
        save_data(df_merged)
        logger.info("ETL Pipeline completed successfully.")

    except Exception:
        logger.error("ETL Pipeline failed.")
        raise


if __name__ == "__main__":
    run_etl_pipeline()
