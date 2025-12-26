# 🚑 Emergency Vehicle Demand Forecasting

> Spatiotemporal ambulance demand prediction using H3 hexagonal grids, XGBoost, and MLOps best practices.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-purple)

---

## 🎯 Project Overview

This project predicts **ambulance demand by location and time** to help emergency services optimize fleet positioning and reduce response times.

| Component | Technology |
|-----------|------------|
| Spatial Indexing | Uber H3 Hexagonal Grids |
| ML Model | XGBoost Regressor |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently AI |
| API Framework | FastAPI |
| Containerization | Docker |

---

## 🏗️ Architecture
┌─────────────┐ ┌──────────────┐ ┌─────────────┐
│ Raw Data │────▶│ Features │────▶│ Model │
│ (GPS/Time) │ │ (H3 + Time) │ │ (XGBoost) │
└─────────────┘ └──────────────┘ └─────────────┘
│
▼
┌─────────────┐
│ FastAPI │
│ (REST API) │
└─────────────┘


---

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/emergency-vehicle-mlops.git
cd emergency-vehicle-mlops

# Create environment
conda create -n ambulance python=3.10 -y
conda activate ambulance

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python src/emergency_forecast_mlops/data/generate_data.py

# Train model
python src/emergency_forecast_mlops/models/train.py

# Run API
python run_api.py



---

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/emergency-vehicle-mlops.git
cd emergency-vehicle-mlops

# Create environment
conda create -n ambulance python=3.10 -y
conda activate ambulance

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python src/emergency_forecast_mlops/data/generate_data.py

# Train model
python src/emergency_forecast_mlops/models/train.py

# Run API
python run_api.py


Docker
# Build image
docker build -t ambulance-forecast-api .

# Run container
docker run -p 8000:8000 ambulance-forecast-api


📡 API Usage
Health Check
bash
curl http://localhost:8000/health
Predict Demand
bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.9352,
    "longitude": 77.6245,
    "hour": 14,
    "day_of_week": 2,
    "month": 6
  }'
Response
json
{
  "h3_index": "88624a5a5ffffff",
  "predicted_demand": 2.35,
  "demand_level": "MEDIUM",
  "latitude": 12.9352,
  "longitude": 77.6245,
  "hour": 14,
  "day_of_week": 2,
  "timestamp": "2024-01-15T14:30:00"
}
📁 Project Structure
text
emergency-vehicle-mlops/
├── src/emergency_forecast_mlops/
│   ├── data/           # Data generation & validation
│   ├── features/       # H3 & time feature engineering
│   ├── models/         # XGBoost training
│   └── api/            # FastAPI endpoints
├── data/
│   ├── raw/            # Raw GPS data
│   ├── features/       # Processed features
│   └── models/         # Trained models
├── config/             # Configuration files
├── Dockerfile          # Container setup
├── requirements.txt    # Dependencies
└── run_api.py          # API entry point


🔬 Key Features
1. H3 Hexagonal Grids
Divides Bangalore into ~150 hexagonal zones
Resolution 8 (~1km² per hexagon)
Uniform spatial aggregation
2. Time-Based Features
Hour of day, day of week
Weekend flag
Lag features (1h, 24h, 168h)
Rolling averages
3. MLOps Pipeline
MLflow experiment tracking
Model versioning
Reproducible training
4. Production-Ready API
FastAPI with auto-documentation
Input validation (Pydantic)
Health checks
Docker deployment
📊 Model Performance
Metric	Value
MAE	~0.45
RMSE	~0.68
R² Score	~0.72
🛠️ Tech Stack
Python 3.10
H3 - Uber's hexagonal spatial indexing
XGBoost - Gradient boosting model
MLflow - Experiment tracking
Evidently - Drift monitoring
FastAPI - REST API
Docker - Containerization
📈 Future Improvements
 Real-time data integration (IUDX API)
 Drift monitoring dashboard
 Multi-city support
 Weather feature integration
 CI/CD pipeline


👨‍💻 Author
Your Name

GitHub: @vaishnabala