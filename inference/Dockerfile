FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/modmodel_pruned_ptq.tflite

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY dependencies first to leverage Docker caching
COPY requirements.txt .

# Install all Python packages (pip layer placed BEFORE model copy)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, utilities, and training statistics
COPY config.py utils.py training_stats.npy ./
COPY inference/ ./inference/

# Copy data artifacts (e.g., training_stats.npy) if required by config
COPY data/ ./data/

# Copy model files LAST so model updates only invalidate the final layer
COPY models/ ./models/

# Command to launch the inference listener/publisher service
CMD ["python", "inference/inference_service.py"]