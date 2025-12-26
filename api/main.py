"""
FastAPI Application
===================
REST API for ambulance demand prediction.

Usage:
    uvicorn src.emergency_forecast_mlops.api.main:app --reload
"""

import joblib
import h3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================
import os
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "D:/GitHub_Works/emergency-vehicle-mlops/data/models/xgboost_model.joblib"))

# MODEL_PATH = Path("D:/GitHub_Works/emergency-vehicle-mlops/data/models/xgboost_model.joblib")
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
# REQUEST/RESPONSE SCHEMAS
# =========================================================

class PredictionRequest(BaseModel):
    """Input schema for prediction."""
    
    latitude: float = Field(..., ge=12.8, le=13.2, description="Latitude (Bangalore)")
    longitude: float = Field(..., ge=77.4, le=77.8, description="Longitude (Bangalore)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    
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
    """Output schema for prediction."""
    
    h3_index: str
    predicted_demand: float
    demand_level: str
    latitude: float
    longitude: float
    hour: int
    day_of_week: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    timestamp: str


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_demand_level(demand: float) -> str:
    """Convert demand number to category."""
    if demand < 1:
        return "LOW"
    elif demand < 3:
        return "MEDIUM"
    elif demand < 5:
        return "HIGH"
    else:
        return "CRITICAL"


def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    """Prepare features for model prediction."""
    
    features = {
        "hour": request.hour,
        "day_of_week": request.day_of_week,
        "is_weekend": 1 if request.day_of_week >= 5 else 0,
        "month": request.month,
        "demand_lag_1h": 1.0,      # Default values for demo
        "demand_lag_24h": 1.0,
        "demand_rolling_3h": 1.0,
        "demand_rolling_24h": 1.0,
    }
    
    return pd.DataFrame([features])


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/", tags=["General"])
async def root():
    """Welcome message."""
    return {
        "message": "🚑 Ambulance Demand Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_demand(request: PredictionRequest):
    """Predict ambulance demand for a location and time."""
    
    # Check model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert lat/lon to H3 hexagon
    h3_index = h3.latlng_to_cell(request.latitude, request.longitude, H3_RESOLUTION)
    
    # Prepare features
    features_df = prepare_features(request)
    
    # Make prediction
    prediction = model.predict(features_df)[0]
    prediction = max(0, prediction)  # Ensure non-negative
    
    # Get demand level
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
# RUN SERVER (for direct execution)
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)