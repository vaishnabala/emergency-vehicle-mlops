# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/models/ ./data/models/
COPY run_api.py .

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "run_api.py"]