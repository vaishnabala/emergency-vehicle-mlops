"""
Model Training Script
=====================
Trains XGBoost model to predict ambulance demand per hexagon.
Logs experiments to MLflow.

Usage:
    python src/emergency_forecast_mlops/models/train.py
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

# Paths
FEATURES_PATH = Path("D:/GitHub_Works/emergency-vehicle-mlops/data/features/features.csv")
MODEL_OUTPUT_PATH = Path("D:/GitHub_Works/emergency-vehicle-mlops/data/models/xgboost_model.joblib")
MLFLOW_TRACKING_URI = "sqlite:///D:/GitHub_Works/emergency-vehicle-mlops/mlflow.db"

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
}

# Feature columns (what model learns from)
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_rolling_3h",
    "demand_rolling_24h",
]

# Target column (what model predicts)
TARGET_COLUMN = "demand_count"


# =========================================================
# DATA PREPARATION
# =========================================================

def load_features():
    """Load feature dataset."""
    
    print("📂 Loading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"   ✅ Loaded {len(df):,} records")
    print(f"   ✅ Columns: {list(df.columns)}")
    
    return df


def prepare_data(df):
    """Prepare features and target for training."""
    
    print("\n🔧 Preparing data...")
    
    # Select features and target
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    print(f"   ✅ Features shape: {X.shape}")
    print(f"   ✅ Target shape: {y.shape}")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   ✅ Train size: {len(X_train):,}")
    print(f"   ✅ Test size: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test


# =========================================================
# MODEL TRAINING
# =========================================================

def train_model(X_train, y_train):
    """Train XGBoost model."""
    
    print("\n🤖 Training XGBoost model...")
    print(f"   Parameters: {MODEL_PARAMS}")
    
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    print("   ✅ Model trained successfully")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    
    print("\n📊 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    
    print(f"   ✅ MAE (Mean Absolute Error): {mae:.4f}")
    print(f"   ✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"   ✅ R² Score: {r2:.4f}")
    
    return metrics, y_pred


def get_feature_importance(model, feature_names):
    """Get feature importance ranking."""
    
    print("\n📈 Feature Importance:")
    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"   {row['feature']:20} {bar} {row['importance']:.3f}")
    
    return importance_df


# =========================================================
# MLFLOW TRACKING
# =========================================================

def setup_mlflow():
    """Setup MLflow tracking."""
    
    print("\n📊 Setting up MLflow...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ambulance_demand_forecast")
    
    print(f"   ✅ Tracking URI: {MLFLOW_TRACKING_URI}")
    print("   ✅ Experiment: ambulance_demand_forecast")


def log_to_mlflow(model, metrics, params, feature_importance):
    """Log experiment to MLflow."""
    
    print("\n📝 Logging to MLflow...")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.xgboost.log_model(model, "model") # type: ignore
        
        # Log feature importance as artifact
        importance_path = Path("feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path) # type: ignore
        importance_path.unlink()  # Delete temp file
        
        run_id = mlflow.active_run().info.run_id # type: ignore
        print(f"   ✅ Logged run: {run_id}")
    
    return run_id


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):
    """Save trained model to disk."""
    
    print("\n💾 Saving model...")
    
    # Ensure directory exists
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    
    print(f"   ✅ Model saved: {MODEL_OUTPUT_PATH}")


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_training():
    """Run the complete training pipeline."""
    
    print("=" * 60)
    print("🤖 MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load features
    df = load_features()
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Step 3: Setup MLflow
    setup_mlflow()
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Step 6: Feature importance
    importance_df = get_feature_importance(model, FEATURE_COLUMNS)
    
    # Step 7: Log to MLflow
    run_id = log_to_mlflow(model, metrics, MODEL_PARAMS, importance_df)
    
    # Step 8: Save model
    save_model(model)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"""
📊 Results Summary:
-------------------
   MAE:  {metrics['mae']:.4f}
   RMSE: {metrics['rmse']:.4f}
   R²:   {metrics['r2']:.4f}

📁 Artifacts:
-------------
   Model: {MODEL_OUTPUT_PATH}
   MLflow Run: {run_id}

🔍 View MLflow UI:
------------------
   Run: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}
   Open: http://localhost:5000

Next Step: Step 5 - API Development
Command: python src/emergency_forecast_mlops/api/main.py
""")
    
    return model, metrics


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    model, metrics = run_training()