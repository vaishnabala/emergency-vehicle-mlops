"""
Feature Engineering Script
==========================
Converts raw ambulance GPS data into ML-ready features using H3 hexagons.

Usage:
    python src/emergency_forecast_mlops/features/build_features.py
"""

import pandas as pd
import numpy as np
import h3
from pathlib import Path
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================

# Paths
RAW_DATA_PATH = Path("D:/GitHub_Works/emergency-vehicle-mlops/data/emergency_data.csv")
FEATURES_OUTPUT_PATH = Path("D:/GitHub_Works/emergency-vehicle-mlops/data/features/features.csv")
HEXAGON_OUTPUT_PATH = Path("D:/GitHub_Works/emergency-vehicle-mlops/data/features/hexagon_mapping.csv")

# H3 Resolution
# Resolution 8 = ~0.74 km² hexagons (good for city-level analysis)
# Resolution 9 = ~0.1 km² hexagons (more granular)
H3_RESOLUTION = 8


# =========================================================
# FEATURE ENGINEERING FUNCTIONS
# =========================================================

def load_raw_data():
    """Load and filter raw ambulance data."""
    
    print("📂 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   ✅ Loaded {len(df):,} records")
    
    # Filter only active ambulances (serviceOnDuty == YES)
    df_active = df[df["serviceOnDuty"] == "YES"].copy()
    print(f"   ✅ Filtered to {len(df_active):,} active records")
    
    return df_active


def add_h3_features(df):
    """Convert lat/lon to H3 hexagon IDs."""
    
    print("\n🔷 Adding H3 hexagon features...")
    
    # Convert to H3 hexagon ID
    df["h3_index"] = df.apply(
        lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION),
        axis=1
    )
    
    # Get hexagon center coordinates (useful for visualization)
    df["h3_center_lat"] = df["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
    df["h3_center_lon"] = df["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])
    
    unique_hexagons = df["h3_index"].nunique()
    print(f"   ✅ Created {unique_hexagons} unique hexagons")
    
    return df


def add_time_features(df):
    """Extract time-based features from datetime."""
    
    print("\n🕐 Adding time features...")
    
    # Parse datetime
    df["datetime"] = pd.to_datetime(df["observationDateTime"])
    
    # Extract time components
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    
    # Is weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Time of day category
    def get_time_category(hour):
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    df["time_of_day"] = df["hour"].apply(get_time_category)
    
    print("   ✅ Added: year, month, day, hour, day_of_week, week_of_year")
    print("   ✅ Added: is_weekend, time_of_day")
    
    return df


def aggregate_demand(df):
    """Aggregate ambulance demand per hexagon per hour."""
    
    print("\n📊 Aggregating demand per hexagon per hour...")
    
    # Create a datetime key for grouping (date + hour)
    df["datetime_hour"] = df["datetime"].dt.floor("H")
    
    # Aggregate: count ambulances per hexagon per hour
    demand_df = df.groupby(
        ["h3_index", "datetime_hour", "year", "month", "day", "hour", "day_of_week", "is_weekend"]
    ).agg(
        demand_count=("license_plate", "count"),
        unique_vehicles=("license_plate", "nunique"),
        h3_center_lat=("h3_center_lat", "first"),
        h3_center_lon=("h3_center_lon", "first"),
    ).reset_index()
    
    print(f"   ✅ Created {len(demand_df):,} aggregated records")
    print(f"   ✅ Demand range: {demand_df['demand_count'].min()} to {demand_df['demand_count'].max()}")
    
    return demand_df


def add_lag_features(df):
    """Add lag features for time series prediction."""
    
    print("\n⏮️  Adding lag features...")
    
    # Sort by hexagon and time
    df = df.sort_values(["h3_index", "datetime_hour"]).reset_index(drop=True)
    
    # Lag features (previous hour demand in same hexagon)
    df["demand_lag_1h"] = df.groupby("h3_index")["demand_count"].shift(1)
    df["demand_lag_24h"] = df.groupby("h3_index")["demand_count"].shift(24)
    df["demand_lag_168h"] = df.groupby("h3_index")["demand_count"].shift(168)  # 1 week
    
    # Rolling averages
    df["demand_rolling_3h"] = df.groupby("h3_index")["demand_count"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["demand_rolling_24h"] = df.groupby("h3_index")["demand_count"].transform(
        lambda x: x.rolling(window=24, min_periods=1).mean()
    )
    
    # Fill NaN values with 0 for lag features
    lag_columns = ["demand_lag_1h", "demand_lag_24h", "demand_lag_168h"]
    df[lag_columns] = df[lag_columns].fillna(0)
    
    print("   ✅ Added: demand_lag_1h, demand_lag_24h, demand_lag_168h")
    print("   ✅ Added: demand_rolling_3h, demand_rolling_24h")
    
    return df


def create_hexagon_mapping(df):
    """Create a reference table of all hexagons."""
    
    print("\n🗺️  Creating hexagon mapping table...")
    
    hexagon_df = df.groupby("h3_index").agg(
        center_lat=("h3_center_lat", "first"),
        center_lon=("h3_center_lon", "first"),
        total_demand=("demand_count", "sum"),
        avg_demand=("demand_count", "mean"),
        record_count=("demand_count", "count"),
    ).reset_index()
    
    print(f"   ✅ Created mapping for {len(hexagon_df)} hexagons")
    
    return hexagon_df


def save_features(features_df, hexagon_df):
    """Save feature datasets to CSV."""
    
    print("\n💾 Saving features...")
    
    # Ensure directories exist
    FEATURES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main features
    features_df.to_csv(FEATURES_OUTPUT_PATH, index=False)
    print(f"   ✅ Saved: {FEATURES_OUTPUT_PATH}")
    
    # Save hexagon mapping
    hexagon_df.to_csv(HEXAGON_OUTPUT_PATH, index=False)
    print(f"   ✅ Saved: {HEXAGON_OUTPUT_PATH}")


def display_summary(features_df, hexagon_df):
    """Display feature engineering summary."""
    
    print("\n" + "=" * 60)
    print("📊 FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    
    print(f"\n📋 Features Dataset:")
    print(f"   Shape: {features_df.shape[0]:,} rows × {features_df.shape[1]} columns")
    
    print(f"\n📋 Feature Columns:")
    for col in features_df.columns:
        print(f"   - {col}")
    
    print(f"\n📋 Hexagon Stats:")
    print(f"   Total hexagons: {len(hexagon_df)}")
    print(f"   Avg demand per hexagon: {hexagon_df['avg_demand'].mean():.2f}")
    
    print(f"\n📋 Sample Features (first 3 rows):")
    print(features_df.head(3).to_string(index=False))


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_feature_engineering():
    """Run the complete feature engineering pipeline."""
    
    print("=" * 60)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load raw data
    df = load_raw_data()
    
    # Step 2: Add H3 hexagon features
    df = add_h3_features(df)
    
    # Step 3: Add time features
    df = add_time_features(df)
    
    # Step 4: Aggregate demand per hexagon per hour
    demand_df = aggregate_demand(df)
    
    # Step 5: Add lag features
    features_df = add_lag_features(demand_df)
    
    # Step 6: Create hexagon mapping
    hexagon_df = create_hexagon_mapping(features_df)
    
    # Step 7: Save features
    save_features(features_df, hexagon_df)
    
    # Step 8: Display summary
    display_summary(features_df, hexagon_df)
    
    print("\n" + "=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 60)
    print("\nNext Step: Step 4 - Model Training")
    print("Command: python src/emergency_forecast_mlops/models/train.py")
    
    return features_df, hexagon_df


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    features_df, hexagon_df = run_feature_engineering()