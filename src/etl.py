"""ETL (Extract, Transform, Load) functions for the Energy Arbitrage Project."""

import requests
import pandas as pd
from typing import Dict, Any, Optional
import os
import logging
from .config import DB_CONNECTION_STRING, WEATHER_API_URL, WEATHER_API_PARAMS, PROCESSED_DATA_PATH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_weather_data(url: str = WEATHER_API_URL, params: Dict[str, Any] = WEATHER_API_PARAMS) -> pd.DataFrame:
    """
    Fetches historical weather data from Open-Meteo API.
    """
    logger.info(f"Fetching weather data from {url}...")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data["hourly"]
        df_weather = pd.DataFrame({
            "time_ept": hourly["time"],
            "temp_c": hourly["temperature_2m"],
            "wind_kph": hourly["wind_speed_10m"],
            "solar_radiation": hourly["shortwave_radiation"]
        })
        
        # Convert to datetime
        df_weather["time_ept"] = pd.to_datetime(df_weather["time_ept"])
        
        logger.info(f"Successfully downloaded {len(df_weather)} rows of weather data.")
        return df_weather
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
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
            df_prices["datetime_beginning_ept"] = pd.to_datetime(df_prices["datetime_beginning_ept"])
            
        logger.info(f"Loaded {len(df_prices)} rows from SQL.")
        return df_prices
    except Exception as e:
        logger.error(f"Error fetching data from database: {e}")
        raise

def merge_data(df_prices: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merges price data with weather data on timestamp.
    """
    logger.info("Merging price and weather data...")
    
    # Merge (Inner Join)
    df_merged = df_prices.merge(
        df_weather,
        left_on="datetime_beginning_ept",
        right_on="time_ept",
        how="inner"
    )
    
    # Drop duplicate time column
    if "time_ept" in df_merged.columns:
        df_merged = df_merged.drop("time_ept", axis=1)
        
    logger.info(f"Final Dataset Shape: {df_merged.shape}")
    return df_merged

def save_data(df: pd.DataFrame, output_path: str = PROCESSED_DATA_PATH) -> None:
    """
    Saves the processed dataframe to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Data successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
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
        
    except Exception as e:
        logger.error("ETL Pipeline failed.")

if __name__ == "__main__":
    run_etl_pipeline()