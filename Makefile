.PHONY: help install lint test preprocess train serve docker-build docker-run k8s-deploy k8s-status k8s-delete docker-compose-up clean

help:
	@echo "Logibridge MLOps project commands - Available Commands"
	@echo "===================================="
	@echo "make install          - Install dependencies (pip)"
	@echo "make lint             - Run Ruff lint checks"
	@echo "make test             - Run unit tests"
	@echo "make eda              - Run exploratory data analysis"
	@echo "make preprocess       - Run data preprocessing"
	@echo "make train            - Run model training locally"
	@echo "make inference        - Run model inference on test data"
	@echo "make serve            - Run FastAPI model serving API locally"
	@echo "make docker-build     - Build Docker image"
	@echo "make docker-run       - Run training in Docker"
	@echo "make docker-compose-up    - Start complete stack"
	@echo "make docker-compose-down  - Stop docker services"
	@echo "make mlflow               - Start MLflow UI"
	@echo "make k8s-deploy       - Deploy API to local Kubernetes"
	@echo "make k8s-status       - Show Kubernetes deployment status"
	@echo "make k8s-delete       - Delete Kubernetes deployment"
	@echo "make clean            - Clean up generated files"
	@echo "make logs             - View training logs"

# Install dependencies
install:
	pip install -r requirements.txt

# Code quality check
lint:
	ruff check src tests

# Run unit tests
test:
	pytest

# Run EDA analysis
eda:
	python src/EDA_analysis.py

# Data preprocessing
preprocess:
	python src/preprocess_data.py

# Model training
train:
	python src/model_train.py

# Model Inference
inference:
	python src/inference.py

# Run FastAPI inference service
serve:
	uvicorn src.serve_api:app --host 0.0.0.0 --port 8000

# Build Docker image
docker-build:
	docker build -t logibridge-mlops-api:latest .

# Run Docker container
docker-run:
	docker run --rm \
		-p 8000:8000 \
		-v $$(pwd)/models:/app/models \
		logibridge-mlops-api:latest


# Docker compose deployment
docker-compose-up:
	docker-compose up --build


docker-compose-down:
	docker-compose down


k8s-deploy:
	kubectl apply -k k8s

k8s-status:
	kubectl get all -n mlops-assignment

k8s-delete:
	kubectl delete -k k8s

# Start MLflow tracking UI
mlflow:
	mlflow ui \
		--backend-store-uri sqlite:///outputs/mlflow/mlflow.db \
		--host 0.0.0.0 \
		--port 5000

# View training logs
logs:
	tail -f logs/training.log

# Clean generated files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
