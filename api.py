"""
FastAPI application for Energy Price Prediction API.

This module provides REST endpoints for predicting energy prices
using the trained Gradient Boosting model.
"""

import logging
from datetime import datetime
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Energy Price Prediction API",
    description="Predicts PJM Western Hub energy prices using weather and historical data",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """Input features for price prediction."""

    temp_c: float = Field(..., description="Temperature in Celsius", ge=-40, le=50)
    wind_kph: float = Field(..., description="Wind speed in km/h", ge=0, le=200)
    solar_radiation: float = Field(
        ..., description="Solar radiation in W/m²", ge=0, le=1500
    )
    price_1h_ago: float = Field(..., description="Price 1 hour ago in $/MWh", ge=0)
    price_24h_ago: float = Field(..., description="Price 24 hours ago in $/MWh", ge=0)
    avg_price_last_24h: float = Field(
        ..., description="Average price last 24h in $/MWh", ge=0
    )
    hour_of_day: int = Field(..., description="Hour of day (0-23)", ge=0, le=23)
    day_of_week: int = Field(..., description="Day of week (0-6, 0=Sunday)", ge=0, le=6)
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)

    class Config:
        json_schema_extra = {
            "example": {
                "temp_c": 25.0,
                "wind_kph": 15.2,
                "solar_radiation": 450.0,
                "price_1h_ago": 35.5,
                "price_24h_ago": 32.1,
                "avg_price_last_24h": 33.8,
                "hour_of_day": 17,
                "day_of_week": 2,
                "month": 7,
            }
        }


class PredictionOutput(BaseModel):
    """Output prediction response."""

    predicted_price: float = Field(..., description="Predicted price in $/MWh")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    timestamp: str


# Load the trained model at startup
MODEL_PATH = "src/best_estimator.pkl"
model = None


@app.on_event("startup")
async def load_model():
    """Load the trained model on application startup."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.error("Model file not found at %s", MODEL_PATH)
        logger.warning("API will run without model - predictions will fail")
    except Exception as e:
        logger.error("Error loading model: %s", e)
        logger.warning("API will run without model - predictions will fail")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Energy Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify API status."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_price(input_data: PredictionInput):
    """
    Predict energy price based on weather and historical data.

    Args:
        input_data: Input features including weather and historical prices

    Returns:
        Predicted price in $/MWh

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later."
        )

    try:
        # Convert input to DataFrame (model expects this format)
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])

        # Ensure column order matches training data
        feature_columns = [
            "temp_c",
            "wind_kph",
            "solar_radiation",
            "price_1h_ago",
            "price_24h_ago",
            "avg_price_last_24h",
            "hour_of_day",
            "day_of_week",
            "month",
        ]
        df = df[feature_columns]

        # Make prediction
        prediction = model.predict(df)[0]

        logger.info("Prediction made: %.2f $/MWh", prediction)

        return PredictionOutput(
            predicted_price=round(float(prediction), 2),
            timestamp=datetime.now().isoformat(),
            model_version="GradientBoostingRegressor-v1.0",
        )

    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(input_data: List[PredictionInput]):
    """
    Predict energy prices for multiple inputs (batch prediction).

    Args:
        input_data: List of input features

    Returns:
        List of predictions

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later."
        )

    try:
        # Convert list of inputs to DataFrame
        input_dicts = [item.dict() for item in input_data]
        df = pd.DataFrame(input_dicts)

        # Ensure column order
        feature_columns = [
            "temp_c",
            "wind_kph",
            "solar_radiation",
            "price_1h_ago",
            "price_24h_ago",
            "avg_price_last_24h",
            "hour_of_day",
            "day_of_week",
            "month",
        ]
        df = df[feature_columns]

        # Make predictions
        predictions = model.predict(df)

        results = [
            {
                "predicted_price": round(float(pred), 2),
                "input_index": idx,
            }
            for idx, pred in enumerate(predictions)
        ]

        logger.info("Batch prediction made for %d inputs", len(input_data))

        return {
            "predictions": results,
            "timestamp": datetime.now().isoformat(),
            "model_version": "GradientBoostingRegressor-v1.0",
        }

    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
