# Project Structure

This repository is organized to support development, training, conversion, and deployment of ML models on edge devices.

Root layout (top-level files and directories you should expect):

- README.md                     - Project overview and quickstart
- requirements.txt              - Python dependencies
- Dockerfile                    - Container recipe for building the app image
- scripts/                      - Utility scripts (model conversion, evaluation, helpers)
- src/                          - Source code for training, inference, and utilities
  - src/train.py                - Training entrypoint (if applicable)
  - src/inference.py            - Inference runtime used on desktop/edge
  - src/utils/                  - Helper modules
- models/                       - Checkpoints, raw models produced by training
  - models/ckpt/                - Checkpoints and saved_model folders
  - models/converted/           - Converted formats (TFLite, ONNX, TensorRT)
- data/                         - Datasets, sample inputs, or dataset download scripts
- notebooks/                    - Jupyter notebooks for experiments and EDA
- edge/                         - Edge-specific code, wrappers, and config
  - edge/serve.py               - Lightweight server/launcher for the device
  - edge/systemd/               - Example systemd unit files to run on boot
- tests/                        - Unit/integration tests
- docs/                         - Additional documentation

Notes:
- If some folders are missing, they may not be needed for the current assignment; this file documents an intended/typical structure for maintainers and graders.
- Use models/converted/ to keep all deployment-ready model artifacts separate from training checkpoints.
- Keep scripts idempotent and well-documented so conversion and deployment steps can be reproduced.
