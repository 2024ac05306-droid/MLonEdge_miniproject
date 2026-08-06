FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/model_ptq.tflite

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY dependencies first to leverage Docker caching
COPY requirements.txt .

# Install all Python packages (pip layer placed BEFORE model copy)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and utility modules
COPY config.py utils.py ./
COPY inference/ ./inference/

# Copy model files LAST so model updates only invalidate the final layer
COPY models/ ./models/

# Command to launch the inference listener/publisher service
CMD ["python", "inference/mqtt_inference_service.py"]