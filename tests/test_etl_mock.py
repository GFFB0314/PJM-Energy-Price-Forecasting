"""Unit tests for ETL functions with mocked external dependencies."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.etl import fetch_price_data, fetch_weather_data, merge_data

# --- Fixtures ---


@pytest.fixture
def mock_weather_response():
    """Fixture providing mock weather API response data."""
    return {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [10.5, 9.8],
            "wind_speed_10m": [15.2, 14.1],
            "shortwave_radiation": [0.0, 0.0],
        }
    }


@pytest.fixture
def sample_price_data():
    """Fixture providing sample price DataFrame for testing."""
    return pd.DataFrame(
        {
            "datetime_beginning_ept": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 01:00:00"]
            ),
            "price": [30.5, 32.1],
            "node_id": [1, 1],
        }
    )


@pytest.fixture
def sample_weather_data():
    """Fixture providing sample weather DataFrame for testing."""
    return pd.DataFrame(
        {
            "time_ept": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
            "temp_c": [10.5, 9.8],
            "wind_kph": [15.2, 14.1],
            "solar_radiation": [0.0, 0.0],
        }
    )


# --- Tests ---


def test_fetch_weather_data_success(mock_weather_response):
    """Test successful weather data fetching with mocked API."""
    with patch("src.etl.requests.get") as mock_get:
        # Configure the mock to return our fixture data
        mock_response = MagicMock()
        mock_response.json.return_value = mock_weather_response
        mock_get.return_value = mock_response

        # Call the function
        df = fetch_weather_data()

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "temp_c" in df.columns
        assert df.iloc[0]["temp_c"] == 10.5
        mock_get.assert_called_once()


def test_fetch_weather_data_failure():
    """Test that the function raises an error when the API request fails."""
    with patch("src.etl.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        with pytest.raises(requests.exceptions.RequestException):
            fetch_weather_data()


def test_fetch_price_data_mocked():
    """Test database retrieval by mocking pandas read_sql."""
    # We mock pd.read_sql directly to avoid needing a real DB connection
    with patch("src.etl.pd.read_sql") as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame(
            {"datetime_beginning_ept": ["2024-01-01 00:00:00"], "price": [50.0]}
        )

        df = fetch_price_data("dummy_connection_string")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["price"] == 50.0
        mock_read_sql.assert_called_once()


def test_merge_data(sample_price_data, sample_weather_data):
    """Test that merging works correctly and handles mismatched rows if any."""

    # Create a mismatch to test inner join (add a row to prices that isn't in weather)
    extra_price_row = pd.DataFrame(
        {
            "datetime_beginning_ept": pd.to_datetime(["2024-01-01 02:00:00"]),
            "price": [28.0],
            "node_id": [1],
        }
    )
    prices_with_extra = pd.concat([sample_price_data, extra_price_row])

    # Merge should only keep the overlapping timestamps (inner join)
    merged_df = merge_data(prices_with_extra, sample_weather_data)

    # Expect 2 rows, not 3 (since the 3rd row has no matching weather data)
    assert len(merged_df) == 2
    assert "temp_c" in merged_df.columns
    assert "price" in merged_df.columns
    # Ensure duplication of timestamp column is handled (should not have 'time_ept')
    assert "time_ept" not in merged_df.columns
