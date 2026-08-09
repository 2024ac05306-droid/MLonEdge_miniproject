# Project Structure

This repository is organized to support development, training, conversion, and deployment of machine learning models on edge devices.

Root layout (top-level files and directories you should expect):

- .env                         - Environment variables (example/secret - not committed by default)
- .env.example                 - Example environment variables
- .github/                     - GitHub configuration (workflows, issue/PR templates)
- DEPLOYMENT_STEPS.md          - Deployment instructions for ML on edge
- Makefile                     - Build & utility tasks
- PROJECT_STRUCTURE.md         - This project structure description
- README.md                    - Project overview and quickstart
- config.py                    - Runtime/config helper (project-specific)
- environment.yml              - Conda environment specification
- requirements.txt             - Python dependencies (pip)
- pip                          - (project file; check contents)
- mlflow.db                    - Local MLflow tracking DB (large file)
- training_stats.npy           - Training statistics/artifacts
- utils.py                     - Utility functions used across the project

Top-level directories:

- __pycache__/                 - Python cache (auto-generated)
- data/                        - Datasets or dataset download scripts
- data_pipeline/               - Data preparation pipelines
- deployment/                  - Deployment manifests, scripts, or configs
- inference/                   - Inference-related code and helpers
- models/                      - Checkpoints and converted/production-ready model artifacts
- mlruns/                      - MLflow run logs
- monitoring/                  - Monitoring/metrics collectors or configs
- optimisation/                - Model optimisation scripts and artifacts
- outputs/                     - Generated outputs (reports, results)
- scenario_architecture/       - Scenario & architecture diagrams/docs
- scripts/                     - Utility scripts (model conversion, evaluation, helpers)
- tests/                       - Unit/integration tests
- training/                    - Training code and configs

Notes:
- I updated only the "Root layout" section to reflect the repository's current top-level files and directories.
- Some expected paths from a generic ML-on-edge layout (e.g., `src/`, `edge/`) are not present at the repo root; instead, this repository uses directories like `inference/`, `deployment/`, and `training/`.
- Keep models/converted/ for production-ready artifacts and add a manifest next to converted models if helpful (e.g., models/converted/manifest.json).
- If you'd like, I can add short descriptions inside any of the directories (list their contents) or create a manifest template — tell me which directory to document next.
