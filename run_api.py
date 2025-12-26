"""
Ambulance Demand Prediction API
===============================
Standalone API runner for Docker
"""

import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import h3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import uvicorn

# =========================================================
# CONFIGURATION
# =========================================================

# Model path (Docker or Local)
if os.path.exists("data/models/xgboost_model.joblib"):
    MODEL_PATH = "data/models/xgboost_model.joblib"
else:
    MODEL_PATH = "D:/GitHub_Works/emergency-vehicle-mlops/data/models/xgboost_model.joblib"

H3_RESOLUTION = 8

# =========================================================
# LOAD MODEL
# =========================================================

print("🔄 Loading model...")
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="🚑 Ambulance Demand Prediction API",
    description="Predict ambulance demand by location and time using ML",
    version="1.0.0",
)

# =========================================================
# SCHEMAS
# =========================================================

class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=12.8, le=13.2)
    longitude: float = Field(..., ge=77.4, le=77.8)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)

    class Config:
        schema_extra = {
            "example": {
                "latitude": 12.9352,
                "longitude": 77.6245,
                "hour": 14,
                "day_of_week": 2,
                "month": 6,
            }
        }


class PredictionResponse(BaseModel):
    h3_index: str
    predicted_demand: float
    demand_level: str
    latitude: float
    longitude: float
    hour: int
    day_of_week: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_demand_level(demand: float) -> str:
    if demand < 1:
        return "LOW"
    elif demand < 3:
        return "MEDIUM"
    elif demand < 5:
        return "HIGH"
    else:
        return "CRITICAL"


def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    features = {
        "hour": request.hour,
        "day_of_week": request.day_of_week,
        "is_weekend": 1 if request.day_of_week >= 5 else 0,
        "month": request.month,
        "demand_lag_1h": 1.0,
        "demand_lag_24h": 1.0,
        "demand_rolling_3h": 1.0,
        "demand_rolling_24h": 1.0,
    }
    return pd.DataFrame([features])


# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "🚑 Ambulance Demand Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_demand(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    h3_index = h3.latlng_to_cell(request.latitude, request.longitude, H3_RESOLUTION)
    features_df = prepare_features(request)
    prediction = max(0, model.predict(features_df)[0])
    demand_level = get_demand_level(prediction)

    return PredictionResponse(
        h3_index=h3_index,
        predicted_demand=round(prediction, 2),
        demand_level=demand_level,
        latitude=request.latitude,
        longitude=request.longitude,
        hour=request.hour,
        day_of_week=request.day_of_week,
        timestamp=datetime.now().isoformat(),
    )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)