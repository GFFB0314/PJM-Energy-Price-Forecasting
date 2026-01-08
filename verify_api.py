"""
Quick test script for the FastAPI application.
Run this to verify the API works locally before deploying.
"""

import requests

# Test data matching the example from the API
test_data = {
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

if __name__ == "__main__":
    print("🔍 Testing Energy Price Prediction API locally...")
    print("Make sure the API is running (uvicorn api:app --reload)")
    print("-" * 50)

    base_url = "http://localhost:8000"

    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"\n✅ Health Check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Health check failed: {e}")
        print("   Please start the API first: uvicorn api:app --reload")
        exit(1)

    # Test 2: Single prediction
    try:
        response = requests.post(
            f"{base_url}/predict", json=test_data, timeout=5
        )
        print(f"\n✅ Prediction: {response.status_code}")
        result = response.json()
        print(f"   Predicted Price: ${result['predicted_price']:.2f}/MWh")
        print(f"   Timestamp: {result['timestamp']}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Prediction failed: {e}")

    # Test 3: Batch prediction
    try:
        batch_data = [test_data, test_data]  # Two identical requests
        response = requests.post(
            f"{base_url}/predict/batch", json=batch_data, timeout=5
        )
        print(f"\n✅ Batch Prediction: {response.status_code}")
        result = response.json()
        print(f"   Predictions: {len(result['predictions'])} items")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Batch prediction failed: {e}")

    print("\n" + "=" * 50)
    print("All tests completed! 🎉")
    print("Next step: Deploy to Render/Railway for public endpoint")
