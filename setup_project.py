import os
from pathlib import Path

print("=" * 60)
print("🚀 Emergency Forecast MLOps - Project Setup")
print("=" * 60)

# Get project root (where this script is located)
project_root = Path(__file__).parent
print(f"\n📍 Project root: {project_root}\n")

# =========================================================
# 1. DIRECTORIES TO CREATE
# =========================================================
directories = [
    "src/emergency_forecast_mlops/data",
    "src/emergency_forecast_mlops/features",
    "src/emergency_forecast_mlops/models",
    "src/emergency_forecast_mlops/pipeline",
    "src/emergency_forecast_mlops/api",
    "src/emergency_forecast_mlops/utils",
    "data/raw",
    "data/processed",
    "data/features",
    "data/models",
    "notebooks/01_exploration",
    "notebooks/02_spatial_analysis",
    "notebooks/03_modeling",
    "config",
    "docker",
    "tests/unit",
    "tests/integration",
    ".github/workflows",
]

print("📁 Creating directories...")
for dir_path in directories:
    full_path = project_root / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ {dir_path}")

# =========================================================
# 2. PYTHON __init__.py FILES
# =========================================================
init_files = [
    "src/emergency_forecast_mlops/__init__.py",
    "src/emergency_forecast_mlops/data/__init__.py",
    "src/emergency_forecast_mlops/features/__init__.py",
    "src/emergency_forecast_mlops/models/__init__.py",
    "src/emergency_forecast_mlops/pipeline/__init__.py",
    "src/emergency_forecast_mlops/api/__init__.py",
    "src/emergency_forecast_mlops/utils/__init__.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
]

print("\n📄 Creating __init__.py files...")
for init_file in init_files:
    file_path = project_root / init_file
    if not file_path.exists():
        file_path.write_text('"""Package initialization."""\n')
        print(f"   ✅ {init_file}")
    else:
        print(f"   ⏭️  {init_file} (exists)")

# =========================================================
# 3. .gitignore
# =========================================================
print("\n📄 Creating .gitignore...")
gitignore_content = """# Byte-compiled files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
env/
.env

# Data files
data/raw/*.csv
data/processed/*.csv
data/features/*.csv
data/models/*.joblib
data/models/*.pkl

# MLflow
mlruns/
mlartifacts/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Build
dist/
build/
*.egg-info/
"""

gitignore_path = project_root / ".gitignore"
if not gitignore_path.exists():
    gitignore_path.write_text(gitignore_content)
    print("   ✅ .gitignore")
else:
    print("   ⏭️  .gitignore (exists)")

# =========================================================
# 4. .env.example
# =========================================================
print("\n📄 Creating .env.example...")
env_content = """# Environment Variables

# IUDX API
IUDX_API_URL=https://rs.cos.iudx.org.in/ngsi-ld/v1/entities
IUDX_API_KEY=your_api_key_here

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# App Settings
DEBUG=True
LOG_LEVEL=INFO
"""

env_path = project_root / ".env.example"
if not env_path.exists():
    env_path.write_text(env_content)
    print("   ✅ .env.example")
else:
    print("   ⏭️  .env.example (exists)")

# =========================================================
# 5. config/config.yaml
# =========================================================
print("\n📄 Creating config/config.yaml...")
config_content = """# Project Configuration

project:
  name: Emergency Forecast MLOps
  version: 0.1.0

data:
  raw_path: data/raw
  processed_path: data/processed
  features_path: data/features

spatial:
  lat_min: 12.8
  lat_max: 13.2
  lon_min: 77.4
  lon_max: 77.8
  h3_resolution: 8

model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

api:
  host: 0.0.0.0
  port: 8000
"""

config_path = project_root / "config/config.yaml"
if not config_path.exists():
    config_path.write_text(config_content)
    print("   ✅ config/config.yaml")
else:
    print("   ⏭️  config/config.yaml (exists)")

# =========================================================
# 6. README.md
# =========================================================
print("\n📄 Creating README.md...")
readme_content = """# Emergency Vehicle Demand Forecasting

Spatiotemporal ambulance demand forecasting using H3 hexagonal grids.

## Tech Stack
- Python 3.10
- H3 (Uber Hexagonal Grids)
- XGBoost
- MLflow
- Evidently AI
- FastAPI
- Docker

## Quick Start

1. Create environment: conda create -n ambulance python=3.10 -y
2. Activate: conda activate ambulance
3. Install: pip install -r requirements.txt
4. Setup: python setup_project.py
5. Run API: uvicorn src.emergency_forecast_mlops.api.main:app --reload

## License
MIT License
"""

readme_path = project_root / "README.md"
if not readme_path.exists():
    readme_path.write_text(readme_content)
    print("   ✅ README.md")
else:
    print("   ⏭️  README.md (exists)")

# =========================================================
# 7. .gitkeep files
# =========================================================
print("\n📄 Creating .gitkeep placeholder files...")
gitkeep_dirs = ["data/raw", "data/processed", "data/features", "data/models"]

for gk_dir in gitkeep_dirs:
    gk_path = project_root / gk_dir / ".gitkeep"
    if not gk_path.exists():
        gk_path.write_text("# Placeholder to keep folder in git\n")
        print(f"   ✅ {gk_dir}/.gitkeep")
    else:
        print(f"   ⏭️  {gk_dir}/.gitkeep (exists)")

# =========================================================
# 8. SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("✅ PROJECT SETUP COMPLETE!")
print("=" * 60)
print("""
What was created:
-----------------
📁 src/emergency_forecast_mlops/  (with subfolders)
📁 data/raw, processed, features, models
📁 notebooks/, config/, docker/, tests/
📄 .gitignore, .env.example, README.md
📄 config/config.yaml

Next Steps:
-----------
1. Verify: tree /F
2. Git init: git init
3. Continue to Step 2.2: Data Generation

""")