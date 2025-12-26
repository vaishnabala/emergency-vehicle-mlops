"""
Data Validation Script
======================
Validates the raw ambulance data before processing.

Usage:
    python src/emergency_forecast_mlops/data/validate_data.py
"""

import pandas as pd
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

# DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "emergency_data.csv"
DATA_PATH = Path("D:/GitHub_Works/emergency-vehicle-mlops/data/emergency_data.csv")
# Expected bounds for Bangalore
LAT_MIN, LAT_MAX = 12.8, 13.2
LON_MIN, LON_MAX = 77.4, 77.8

# Expected columns
EXPECTED_COLUMNS = [
    "emergencyVehicleType",
    "license_plate",
    "vehicleSupportType",
    "observationDateTime",
    "longitude",
    "latitude",
    "serviceOnDuty",
]


# =========================================================
# VALIDATION FUNCTIONS
# =========================================================

def validate_file_exists():
    """Check if data file exists."""
    print("1️⃣  Checking file exists...")
    
    if DATA_PATH.exists():
        print(f"   ✅ File found: {DATA_PATH}")
        return True
    else:
        print(f"   ❌ File NOT found: {DATA_PATH}")
        return False


def validate_columns(df):
    """Check if all expected columns exist."""
    print("\n2️⃣  Checking columns...")
    
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    
    if not missing:
        print("   ✅ All expected columns present")
    else:
        print(f"   ❌ Missing columns: {missing}")
    
    if extra:
        print(f"   ⚠️  Extra columns (ok): {extra}")
    
    return len(missing) == 0


def validate_no_nulls(df):
    """Check for null values."""
    print("\n3️⃣  Checking for null values...")
    
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        print("   ✅ No null values found")
        return True
    else:
        print(f"   ❌ Found {total_nulls} null values:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"      - {col}: {count} nulls")
        return False


def validate_coordinates(df):
    """Check if coordinates are within Bangalore bounds."""
    print("\n4️⃣  Checking coordinate bounds...")
    
    lat_valid = (df["latitude"] >= LAT_MIN) & (df["latitude"] <= LAT_MAX)
    lon_valid = (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX)
    
    lat_invalid = (~lat_valid).sum()
    lon_invalid = (~lon_valid).sum()
    
    if lat_invalid == 0 and lon_invalid == 0:
        print("   ✅ All coordinates within Bangalore bounds")
        print(f"      Latitude:  {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        print(f"      Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
        return True
    else:
        print(f"   ❌ Invalid coordinates found:")
        print(f"      - {lat_invalid} records outside latitude bounds")
        print(f"      - {lon_invalid} records outside longitude bounds")
        return False


def validate_vehicle_type(df):
    """Check emergency vehicle type."""
    print("\n5️⃣  Checking vehicle types...")
    
    types = df["emergencyVehicleType"].unique()
    
    if len(types) == 1 and types[0] == "Ambulance":
        print("   ✅ All records are Ambulance type")
        return True
    else:
        print(f"   ⚠️  Found types: {types}")
        return True  # Not a failure, just info


def validate_service_status(df):
    """Check service status values."""
    print("\n6️⃣  Checking service status...")
    
    valid_statuses = ["YES", "NO"]
    statuses = df["serviceOnDuty"].unique()
    
    invalid = [s for s in statuses if s not in valid_statuses]
    
    if not invalid:
        print("   ✅ Valid service statuses only")
        counts = df["serviceOnDuty"].value_counts()
        for status, count in counts.items():
            pct = count / len(df) * 100
            print(f"      - {status}: {count} ({pct:.1f}%)")
        return True
    else:
        print(f"   ❌ Invalid statuses found: {invalid}")
        return False


def validate_datetime(df):
    """Check datetime format and range."""
    print("\n7️⃣  Checking datetime format...")
    
    try:
        df["parsed_dt"] = pd.to_datetime(df["observationDateTime"])
        print("   ✅ Datetime format is valid")
        print(f"      From: {df['parsed_dt'].min()}")
        print(f"      To:   {df['parsed_dt'].max()}")
        return True
    except Exception as e:
        print(f"   ❌ Datetime parsing failed: {e}")
        return False


def validate_record_count(df):
    """Check if we have enough records."""
    print("\n8️⃣  Checking record count...")
    
    count = len(df)
    
    if count >= 1000:
        print(f"   ✅ Sufficient records: {count:,}")
        return True
    else:
        print(f"   ⚠️  Low record count: {count}")
        return True  # Warning, not failure


# =========================================================
# MAIN VALIDATION
# =========================================================

def run_validation():
    """Run all validation checks."""
    
    print("=" * 60)
    print("🔍 DATA VALIDATION")
    print("=" * 60)
    
    results = []
    
    # Check 1: File exists
    if not validate_file_exists():
        print("\n❌ VALIDATION FAILED: File not found")
        return False
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✅ Loaded {len(df):,} records")
    
    # Run all checks
    results.append(("Columns", validate_columns(df)))
    results.append(("No Nulls", validate_no_nulls(df)))
    results.append(("Coordinates", validate_coordinates(df)))
    results.append(("Vehicle Type", validate_vehicle_type(df)))
    results.append(("Service Status", validate_service_status(df)))
    results.append(("Datetime", validate_datetime(df)))
    results.append(("Record Count", validate_record_count(df)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("=" * 60)
        print("\n🎉 Data is ready for feature engineering!")
        print("\nNext Step: Step 3 - Feature Engineering")
        print("Command: python src/emergency_forecast_mlops/features/build_features.py")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("=" * 60)
        print("\nPlease fix the issues above before proceeding.")
    
    return all_passed


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_validation()