# LogiBridge Edge ML Deployment — Project Structure & Repository Blueprint

This repository implements a complete Machine Learning on Edge (MLOnEdge) deployment, monitoring, and orchestration pipeline for vehicle sensor telemetry classification.

---

## 📂 Complete Project Directory Structure

```text
MLOnEdge/
├── .github/
│   └── workflows/
│       ├── docker-image.yml            # GitHub Actions CI for container smoke tests
│       ├── docker-publish.yml          # GitHub Actions CD for publishing signed containers to GHCR
│       └── google_GKE_Deploy.yml       # Cloud CI/CD deployment pipeline to GKE
├── data/
│   ├── critical_features.csv           # Telemetry subset for critical status conditions
│   ├── normal_features.csv             # Telemetry subset for normal operating conditions
│   ├── training_dataset.csv            # Combined raw dataset used for model training
│   ├── training_stats.npy              # Mean and std scaling statistics for feature normalization
│   └── warning_features.csv            # Telemetry subset for warning status conditions
├── data_pipeline/
│   ├── data_pipeline_cmd.md            # Execution documentation for data pipeline steps
│   ├── mqtt_architecture.md            # Architecture specs for MQTT telemetry streaming
│   ├── normalization.py                # Standalone feature normalization logic
│   ├── preprocessing.py                # Dataset clean-up and preprocessing functions
│   └── simulator.py                    # Real-time sensor telemetry MQTT publisher simulator
├── deployment/
│   └── logibridge_deploy.yml           # Ansible playbook for automated edge container deployment
├── hardware/
│   └── hardware_justification.md       # Hardware accelerator selection & constraint analysis
├── inference/
│   ├── .dockerignore                   # Docker ignore rules for inference container build
│   ├── Dockerfile                      # Container specification for edge inference runtime
│   ├── hardware_accelerator.py         # Arm Ethos-U55 NPU compiler interface & async DMA simulator
│   ├── inference_service.py            # Thin edge runtime wrapper with MQTT event subscriber
│   ├── Inference_video.md              # Documentation/link for inference demonstration video
│   └── model.tflite                    # Edge runtime model binary copy
├── logs/                               # Application execution and runtime logs directory
├── models/
│   ├── best_model.keras                # Trained base floating-point Keras model
│   ├── best_model_pruned.keras         # Pruned Keras model prior to quantization
│   ├── model_fp32.tflite               # Baseline Float32 TFLite model binary
│   ├── model_pruned_ptq.tflite         # Combined Pruned + INT8 PTQ TFLite model binary
│   └── model_ptq.tflite                # Production INT8 Post-Training Quantized TFLite model binary
├── monitoring/
│   ├── drift_monitor.py                # Real-time input feature distribution drift detection engine
│   ├── evaluate_psi_phases.py          # Script for evaluating Population Stability Index (PSI) across phases
│   ├── generate_reference_dist.py      # Generates reference baseline distribution JSON from training data
│   ├── PSI_monitor.py                  # Module for PSI metric computation on streaming data
│   ├── reference_dist.json             # Baseline feature distributions for drift monitoring
│   └── results/
│       ├── drift_metrics.json          # Recorded drift metric outputs
│       ├── drift_statistics.json       # Statistical summary of observed drift
│       ├── drift_summary.json          # Overall summary of drift evaluations
│       └── psi_chart.png               # Visualized PSI drift metrics plot
├── optimisation/
│   ├── benchmark.py                    # Inference throughput and latency benchmark script
│   ├── benchmark_recall.py             # Accuracy/Recall trade-off benchmarking across model variants
│   ├── benchmark_table.py              # Generates formatted comparison table for model variants
│   ├── build_model_variants.py         # Model architecture definitions & calibration dataset generator
│   └── results/
│       ├── benchmark_results.csv       # Benchmark raw outputs across model variants
│       └── pareto_chart.png            # Visualized Pareto frontier (Accuracy vs Latency/Size)
├── outputs/
│   ├── confusion_matrix.png            # Saved confusion matrix for model evaluation
│   ├── sensor_logs_combined.csv        # Simulated multi-phase sensor test telemetry
│   ├── sensor_logs_none.csv            # Clean sensor test logs (no drift)
│   ├── sensor_logs_temp_drift.csv      # Test logs simulating temperature sensor drift
│   ├── training_curve.png              # Loss and accuracy convergence plot
│   └── training_stats.npy              # Cached training statistics output artifact
├── scenario_architecture/
│   ├── Architecture_part1.JPG          # System architecture diagram (Part 1)
│   ├── Architecture_part2.JPG          # System architecture diagram (Part 2)
│   └── constraint_analysis.md          # Edge compute, power, and memory constraint analysis
├── scripts/
│   ├── benchmark_bandwidth.py          # Measures network and MQTT payload bandwidth usage
│   ├── run_ood_experiment.py           # Runs Out-Of-Distribution (+3sigma) model degradation test
│   ├── test_ansible_idempotency.py     # Verifies Ansible playbook idempotency across runs
│   └── test_pipeline.py                # End-to-end integration testing script
├── tests/
│   └── conftest.py                     # Shared pytest session fixtures (synthetic data & mock models)
├── training/
│   ├── convert_ptq.py                  # Converts standard Keras models to INT8 TFLite via PTQ
│   ├── Execution.md                    # Instructions for executing model training pipelines
│   ├── generate_dataset.py             # Generates synthetic/simulated sensor dataset
│   ├── prune_quantise.py               # Combined pruning and Post-Training Quantization pipeline
│   ├── run_norm_experiment.py          # Experiments on feature normalization impacts
│   └── train_model.py                  # Primary training entrypoint for baseline Keras model
├── config.py                           # Global environment variable & path configuration
├── environment.yml                     # Conda environment definition file
├── Makefile                            # Command shortcuts for setup, testing, and deployment
├── mlflow.db                           # Local SQLite database for MLflow experiment tracking
├── mlruns/                             # MLflow tracking directory for runs, parameters, and metrics
├── pip/                                # Local cached pip wheel packages
├── README.md                           # Main repository overview and setup instructions
├── requirements.txt                    # Python environment package dependencies
├── training_stats.npy                  # Root copy of dataset normalization parameters
└── utils.py                            # Core utility module for feature scaling & dataset loading
