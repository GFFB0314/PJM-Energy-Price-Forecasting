# FastAPI Deployment Guide - Free Hosting

This guide will help you deploy your Energy Price Prediction API for **FREE** and get a public endpoint for your LinkedIn post.

## Quick Start (Local Testing)

1. **Install FastAPI dependencies:**
   ```bash
   pip install fastapi uvicorn pydantic
   ```

2. **Run locally:**
   ```bash
   uvicorn api:app --reload
   ```

3. **Test the API:**
   - Visit: `http://localhost:8000/docs` (Interactive Swagger UI)
   - Health check: `http://localhost:8000/health`

---

## Option 1: Render (Recommended - Easiest)

**Pros:** Free tier, auto-deploys from GitHub, custom domain support  
**Cons:** Spins down after 15 min of inactivity (cold start delay)

### Steps:

1. **Add these files to your repo:**

   **`render.yaml`** (in project root):
   ```yaml
   services:
     - type: web
       name: energy-price-api
       env: python
       buildCommand: "pip install -r requirements.txt"
       startCommand: "uvicorn api:app --host 0.0.0.0 --port $PORT"
       plan: free
   ```

   **Update `requirements.txt`** (add these lines):
   ```
   fastapi==0.115.6
   uvicorn[standard]==0.34.0
   pydantic==2.10.5
   ```

2. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add FastAPI deployment"
   git push origin main
   ```

3. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Sign in with GitHub
   - Click "New" → "Web Service"
   - Select your repo
   - Render auto-detects `render.yaml` and deploys!

4. **Get your endpoint:**
   - Copy URL: `https://energy-price-api.onrender.com`
   - Test: `https://energy-price-api.onrender.com/docs`

---

## Option 2: Railway

**Pros:** Very fast, generous free tier, no sleep  
**Cons:** Requires credit card (won't charge on free tier)

### Steps:

1. **Create `Procfile`** (in project root):
   ```
   web: uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

2. **Push to GitHub**

3. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app)
   - Click "Deploy from GitHub"
   - Select your repo
   - Railway auto-deploys!

4. **Get endpoint:**
   - Settings → Generate Domain
   - Example: `https://energy-api-production.up.railway.app`

---

## Option 3: Hugging Face Spaces (ML-Specific)

**Pros:** Built for ML models, no cold starts, great for portfolio  
**Cons:** Requires `Dockerfile`

### Steps:

1. **Create `Dockerfile`:**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 7860

   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

2. **Create Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - New Space → Docker → Create
   - Upload your files

3. **Endpoint:**
   - `https://huggingface.co/spaces/YOUR_USERNAME/energy-price-api`

---

## LinkedIn Post Template

Once deployed, use this template:

```
🚀 Excited to share my latest project: Energy Price Prediction API!

Built a production-ready ML system that forecasts PJM energy prices with 47.3% efficiency ($202K captured profit in backtesting).

🔧 Tech Stack:
- Gradient Boosting Regressor (sklearn)
- FastAPI for REST endpoints
- PostgreSQL with Window Functions
- Time-series CV (no data leakage!)

🌐 Live API: https://YOUR-ENDPOINT.onrender.com/docs

📊 Try the prediction endpoint with real-time weather data!

#MachineLearning #DataScience #EnergyMarkets #Python #FastAPI
```

---

## Testing Your API

**cURL Example:**
```bash
curl -X POST "https://YOUR-ENDPOINT.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "temp_c": 25.0,
    "wind_kph": 15.2,
    "solar_radiation": 450.0,
    "price_1h_ago": 35.5,
    "price_24h_ago": 32.1,
    "avg_price_last_24h": 33.8,
    "hour_of_day": 17,
    "day_of_week": 2,
    "month": 7
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "https://YOUR-ENDPOINT.onrender.com/predict",
    json={
        "temp_c": 25.0,
        "wind_kph": 15.2,
        "solar_radiation": 450.0,
        "price_1h_ago": 35.5,
        "price_24h_ago": 32.1,
        "avg_price_last_24h": 33.8,
        "hour_of_day": 17,
        "day_of_week": 2,
        "month": 7
    }
)
print(response.json())
```

---

## Next Steps

1. Choose a platform (I recommend **Render** for beginners)
2. Add the deployment files to your repo
3. Push to GitHub
4. Deploy and get your endpoint
5. Test it works: `/health` and `/predict`
6. Add to LinkedIn post with a screenshot!

Need help? Check the platform-specific docs or ask me!
