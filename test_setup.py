import sys
print(f"Python: {sys.version}")

try:
    import pandas
    print(f"✅ pandas: {pandas.__version__}")
except ImportError:
    print("❌ pandas failed")

try:
    import numpy
    print(f"✅ numpy: {numpy.__version__}")
except ImportError:
    print("❌ numpy failed")

try:
    import h3
    print(f"✅ h3: {h3.__version__}")
except ImportError:
    print("❌ h3 failed")

try:
    import xgboost
    print(f"✅ xgboost: {xgboost.__version__}")
except ImportError:
    print("❌ xgboost failed")

try:
    import mlflow
    print(f"✅ mlflow: {mlflow.__version__}")
except ImportError:
    print("❌ mlflow failed")

try:
    import fastapi
    print(f"✅ fastapi: {fastapi.__version__}")
except ImportError:
    print("❌ fastapi failed")

print("\n REQUIREMENTS INSTALLATION COMPLETED!")

# if any package failed to import use the below
# conda install pandas numpy scikit-learn xgboost -y
# conda install -c conda-forge h3-py folium -y
# pip install mlflow fastapi uvicorn python-dotenv evidently pytest black ruff joblib requests pydantic