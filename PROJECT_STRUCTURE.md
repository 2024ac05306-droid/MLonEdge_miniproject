# Project Structure

This repository is organized to support development, training, conversion, and deployment of machine learning models on edge devices.

Root layout (top-level files and directories you should expect):

- README.md                     - Project overview and quickstart
- requirements.txt              - Python dependencies
- Dockerfile                    - Container recipe for building the app image
- scripts/                      - Utility scripts (model conversion, evaluation, helpers)
- src/                          - Source code for training, inference, and utilities
  - src/train.py                - Training entrypoint (if applicable)
  - src/inference.py            - Inference runtime used on desktop/edge
  - src/utils/                  - Helper modules
- models/                       - Checkpoints and production-ready model artifacts
  - models/ckpt/                - Checkpoints and saved_model folders
  - models/converted/           - Converted formats (TFLite, ONNX, TensorRT)
- data/                         - Datasets, sample inputs, or dataset download scripts
- notebooks/                    - Jupyter notebooks for experiments and EDA
- edge/                         - Edge-specific code, wrappers, and configs
  - edge/serve.py               - Lightweight server/launcher for the device (entrypoint)
  - edge/systemd/               - Example systemd unit files to run on boot
- tests/                        - Unit/integration tests
- docs/                         - Additional documentation

Guidelines and notes

- If some folders are missing, they may not be needed for the current assignment; this file documents an intended/typical structure for maintainers and graders.
- Use models/converted/ to keep all deployment-ready model artifacts separate from training checkpoints.
- Keep scripts idempotent and well-documented so conversion and deployment steps can be reproduced.
- Add a small manifest (e.g., models/converted/manifest.json) alongside converted models describing format, input shape, required preprocessing/postprocessing, and runtime.
- Prefer small, well-tested inference wrappers in src/inference.py and a lightweight device entrypoint at edge/serve.py so the runtime is easy to containerize and deploy.
