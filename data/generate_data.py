"""
Synthetic Ambulance Data Generator
==================================
Generates realistic ambulance GPS data for Bangalore city.
Matches IUDX API schema for seamless transition to real data.

Usage:
    python src/emergency_forecast_mlops/data/generate_data.py
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

# Bangalore bounding box
LAT_MIN = 12.8
LAT_MAX = 13.2
LON_MIN = 77.4
LON_MAX = 77.8

# Data generation settings
NUM_AMBULANCES = 50
NUM_DAYS = 30
RECORDS_PER_DAY = 500

# Vehicle types
VEHICLE_SUPPORT_TYPES = ["BLS", "ALS", "Patient Transport"]
# BLS = Basic Life Support
# ALS = Advanced Life Support

# Output path
OUTPUT_PATH = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "emergency_data.csv"


# =========================================================
# HOTSPOT LOCATIONS (Realistic demand centers)
# =========================================================

# These represent areas with higher ambulance activity
HOTSPOTS = [
    {"name": "Koramangala", "lat": 12.9352, "lon": 77.6245, "weight": 0.15},
    {"name": "Whitefield", "lat": 12.9698, "lon": 77.7500, "weight": 0.12},
    {"name": "Electronic City", "lat": 12.8399, "lon": 77.6770, "weight": 0.10},
    {"name": "Marathahalli", "lat": 12.9591, "lon": 77.6974, "weight": 0.10},
    {"name": "Jayanagar", "lat": 12.9299, "lon": 77.5826, "weight": 0.08},
    {"name": "Indiranagar", "lat": 12.9784, "lon": 77.6408, "weight": 0.08},
    {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101, "weight": 0.07},
    {"name": "HSR Layout", "lat": 12.9081, "lon": 77.6476, "weight": 0.07},
    {"name": "Hebbal", "lat": 13.0358, "lon": 77.5970, "weight": 0.06},
    {"name": "Yeshwanthpur", "lat": 13.0285, "lon": 77.5416, "weight": 0.05},
    {"name": "Banashankari", "lat": 12.9255, "lon": 77.5468, "weight": 0.06},
    {"name": "Malleshwaram", "lat": 13.0035, "lon": 77.5647, "weight": 0.06},
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def generate_license_plate():
    """Generate Karnataka ambulance license plate."""
    # Format: KA XX Y XXXX
    district = random.choice(["01", "02", "03", "04", "05", "40", "41", "50", "51"])
    letter = random.choice("ABCDEFGHJKLMNPRSTUVWXY")
    number = random.randint(1000, 9999)
    return f"KA{district}{letter}{number}"


def generate_location():
    """Generate location biased towards hotspots."""
    
    # 70% chance to be near a hotspot
    if random.random() < 0.7:
        # Select hotspot based on weights
        weights = [h["weight"] for h in HOTSPOTS]
        hotspot = random.choices(HOTSPOTS, weights=weights, k=1)[0]
        
        # Add some noise (approximately 1-2 km radius)
        lat = hotspot["lat"] + random.gauss(0, 0.01)
        lon = hotspot["lon"] + random.gauss(0, 0.01)
    else:
        # Random location in Bangalore
        lat = random.uniform(LAT_MIN, LAT_MAX)
        lon = random.uniform(LON_MIN, LON_MAX)
    
    # Ensure within bounds
    lat = max(LAT_MIN, min(LAT_MAX, lat))
    lon = max(LON_MIN, min(LON_MAX, lon))
    
    return round(lat, 6), round(lon, 6)


def generate_datetime(base_date):
    """Generate datetime with realistic patterns."""
    
    # Hour distribution (more emergencies during certain hours)
    hour_weights = [
        0.02, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5 AM (low)
        0.04, 0.05, 0.06, 0.07, 0.07, 0.06,  # 6-11 AM (rising)
        0.05, 0.05, 0.05, 0.05, 0.06, 0.07,  # 12-5 PM (steady)
        0.08, 0.07, 0.06, 0.05, 0.04, 0.03,  # 6-11 PM (evening peak then decline)
    ]
    
    hour = random.choices(range(24), weights=hour_weights, k=1)[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    return base_date.replace(hour=hour, minute=minute, second=second)


def generate_service_status():
    """Generate service status (mostly YES for active ambulances)."""
    # 85% on duty, 15% off duty
    return "YES" if random.random() < 0.85 else "NO"


# =========================================================
# MAIN DATA GENERATION
# =========================================================

def generate_ambulance_data():
    """Generate complete synthetic ambulance dataset."""
    
    print("=" * 60)
    print("🚑 Generating Synthetic Ambulance Data")
    print("=" * 60)
    
    # Create ambulance fleet
    ambulances = []
    for i in range(NUM_AMBULANCES):
        ambulances.append({
            "license_plate": generate_license_plate(),
            "vehicleSupportType": random.choice(VEHICLE_SUPPORT_TYPES),
        })
    
    print(f"\n📋 Fleet size: {NUM_AMBULANCES} ambulances")
    print(f"📅 Generating {NUM_DAYS} days of data")
    print(f"📍 Records per day: ~{RECORDS_PER_DAY}")
    
    # Generate records
    records = []
    start_date = datetime(2024, 1, 1)
    
    for day in range(NUM_DAYS):
        current_date = start_date + timedelta(days=day)
        
        # Vary records per day (weekends have different patterns)
        is_weekend = current_date.weekday() >= 5
        day_records = int(RECORDS_PER_DAY * (0.8 if is_weekend else 1.0))
        day_records = int(day_records * random.uniform(0.9, 1.1))  # Add variance
        
        for _ in range(day_records):
            ambulance = random.choice(ambulances)
            lat, lon = generate_location()
            obs_datetime = generate_datetime(current_date)
            
            record = {
                "emergencyVehicleType": "Ambulance",
                "license_plate": ambulance["license_plate"],
                "vehicleSupportType": ambulance["vehicleSupportType"],
                "observationDateTime": obs_datetime.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                "longitude": lon,
                "latitude": lat,
                "serviceOnDuty": generate_service_status(),
            }
            records.append(record)
        
        # Progress indicator
        if (day + 1) % 10 == 0:
            print(f"   ✅ Generated day {day + 1}/{NUM_DAYS}")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by datetime
    df = df.sort_values("observationDateTime").reset_index(drop=True)
    
    print(f"\n📊 Total records generated: {len(df)}")
    
    return df


def save_data(df):
    """Save DataFrame to CSV."""
    
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Data saved to: {OUTPUT_PATH}")
    
    return OUTPUT_PATH


def display_summary(df):
    """Display data summary statistics."""
    
    print("\n" + "=" * 60)
    print("📊 DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n📋 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Columns:")
    for col in df.columns:
        print(f"   - {col}")
    
    print("\n📋 Sample Records (first 3):")
    print(df.head(3).to_string(index=False))
    
    print("\n📋 Service Status Distribution:")
    status_counts = df["serviceOnDuty"].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"   {status}: {count} ({pct:.1f}%)")
    
    print("\n📋 Vehicle Support Types:")
    type_counts = df["vehicleSupportType"].value_counts()
    for vtype, count in type_counts.items():
        pct = count / len(df) * 100
        print(f"   {vtype}: {count} ({pct:.1f}%)")
    
    print("\n📋 Location Bounds:")
    print(f"   Latitude:  {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"   Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    
    print("\n📋 Date Range:")
    print(f"   From: {df['observationDateTime'].min()}")
    print(f"   To:   {df['observationDateTime'].max()}")


# =========================================================
# MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":
    # Generate data
    df = generate_ambulance_data()
    
    # Save to CSV
    save_data(df)
    
    # Display summary
    display_summary(df)
    
    print("\n" + "=" * 60)
    print("✅ DATA GENERATION COMPLETE!")
    print("=" * 60)
    print("\nNext Step: Run feature engineering (Step 3)")
    print("Command: python src/emergency_forecast_mlops/features/build_features.py")