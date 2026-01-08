"""Configuration and DB settings for the Energy Arbitrage Project."""

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


# Database Credentials from environment variables
user = os.getenv("DB_USER", "").strip()
password = os.getenv("DB_PASS", "").strip()
host = os.getenv("DB_HOST", "").strip()
port = os.getenv("DB_PORT", "").strip()
dbname = os.getenv("DB_NAME", "").strip()

# Format postgres://user:password@host:port/dbname
DB_CONNECTION_STRING: str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# Weather API Configuration
WEATHER_API_URL: str = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_API_PARAMS: dict = {
    "latitude": 40.27,
    "longitude": -76.88,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",  # Radiation = Sunlight
    "timezone": "America/New_York",  # Critical: Match PJM EPT Timezone
}

# Data Paths
DATA_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
PROCESSED_DATA_PATH: str = os.path.join(DATA_DIR, "processed", "merged_data.csv")
PROCESSED_DATA_PATH_TRAIN: str = os.path.join(DATA_DIR, "processed", "trained_data.csv")
