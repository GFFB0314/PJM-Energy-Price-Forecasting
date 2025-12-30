"""Configuration and DB settings for the Energy Arbitrage Project."""

import os 
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


# Safe version
user = os.getenv("DB_USER", "").strip()
password = os.getenv("DB_PASS", "").strip()
host = os.getenv("DB_HOST", "").strip()
port = os.getenv("DB_PORT", "").strip()
dbname = os.getenv("DB_NAME", "").strip()

# Format postgres://user:password@host:port/dbname
DB_CONNECTION_STRING: str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"