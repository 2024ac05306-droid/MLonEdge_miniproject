# Project Structure

This repository is organized to support development, training, conversion, and deployment of ML models on edge devices.

Root layout (top-level files and directories you should expect):

# LogiBridge Edge ML Deployment — Project Structure & Repository Blueprint

This repository implements an end-to-end Machine Learning on Edge (MLOnEdge) deployment, monitoring, and orchestration pipeline for vehicle sensor telemetry classification.


## 📂 Root Directory Overview

```text
MLOnEdge/
├── .github/
│   └── workflows/
│       ├── docker-image.yml          # GitHub Actions CI/CD for local Docker builds & smoke tests
│       └── docker-publish.yml        # CI/CD pipeline for building, pushing, & signing images to GHCR
├── ansible/
│   ├── deploy_edge_model.yml         # Primary Ansible playbook for automated edge deployment & health checks
│   └── test_ansible_idempotency.py   # Verification script for Ansible playbooks and idempotency testing
├── data/
│   └── training_stats.npy            # Saved feature mean & standard deviation normalization parameters
├── inference/
│   ├── hardware_accelerator.py       # Micro-NPU (Ethos-U55) compiler interface and DMA async pipeline simulator
│   └── inference_service.py          # Thin edge inference runtime with MQTT event subscriber & INT8 processing
├── models/
│   ├── model_base.keras              # Baseline float32 Keras model artifact
│   └── model_ptq.tflite              # Post-Training Quantized (INT8) TensorFlow Lite model binary
├── monitoring/
│   └── reference_dist.json           # Baseline feature probability distributions for drift/OOD monitoring
├── optimisation/
│   └── build_model_variants.py       # Model architecture definition & TFLite PTQ calibration dataset generator
├── tests/
│   ├── conftest.py                   # Pytest session-wide shared fixtures and synthetic datasets
│   └── test_pipeline.py              # Unit & integration tests (normalization, quantization, & +3sigma OOD shifts)
├── Dockerfile                        # Multi-stage Docker build configuration for edge inference container
├── PROJECT_STRUCTURE.md             # Repository layout and architecture documentation (this file)
├── requirements.txt                  # Python dependencies (TensorFlow, Pytest, Paho-MQTT, MLflow, etc.)
└── utils.py                          # Utility functions for feature standardization and statistics loading
