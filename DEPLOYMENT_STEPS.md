# Deployment Steps — ML on Edge

This document describes repeatable steps to prepare, package, and deploy a machine learning model and its runtime to an edge device (e.g., Raspberry Pi, Jetson Nano, Coral, or other ARM/x86 devices).

## Overview
- Goal: produce a deployment-ready model artifact, containerize the runtime, and run/update reliably on the device.
- Target platforms: Linux-based edge devices (Raspberry Pi OS, Ubuntu), optional accelerators (TPU, NPU).
- Artifacts produced: converted model (TFLite/ONNX/TensorRT), Docker image (or native binary), service configuration.

## Prerequisites
- Development machine:
  - Python >=3.8, pip
  - Docker (for image build)
  - Model conversion toolchains (TensorFlow, ONNX, tflite-runtime, torch, etc.)
- Edge device:
  - SSH access, or serial/console access
  - Docker Engine (or balena/podman) or native runtime installed
  - Adequate storage & compute; optional accelerator drivers installed
- Credentials:
  - Container registry credentials if pushing images (DockerHub, GitHub Container Registry, Azure ACR, etc.)

## Prepare the model
1. Verify the trained model (checkpoint/saved_model) in `models/ckpt/`.
2. Evaluate on representative inputs to confirm expected accuracy/latency.
3. Prune/quantize if needed to reduce size and improve performance.

## Convert model to deployment format
Examples:

- TensorFlow -> TFLite (float32 or post-training quantized)
  - Convert (Python):
    ```python
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model("models/ckpt/saved_model")
    # For post-training quantization:
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open("models/converted/model.tflite", "wb").write(tflite_model)
    ```
  - CLI (if using tflite-tools): follow tool docs.

- PyTorch -> ONNX:
  ```python
  import torch
  model = torch.load("models/ckpt/model.pt")
  dummy = torch.randn(1, 3, 224, 224)
  torch.onnx.export(model, dummy, "models/converted/model.onnx", opset_version=11)
  ```

- ONNX -> TensorRT (Jetson): use TensorRT builder / trtexec.

Store final artifacts in `models/converted/` and include a small `manifest.json` describing format, input shape, and pre/postprocessing.

## Create a minimal runtime
- Keep `src/inference.py` as a thin wrapper that:
  - Loads the converted model
  - Performs required preprocessing and postprocessing
  - Exposes a simple API (HTTP/gRPC/REST) or a CLI for local inference

Example entrypoint: `edge/serve.py` that starts a Flask/FastAPI server and loads the model from `/app/models/model.tflite`.

## Containerize (recommended)
- Add a Dockerfile tailored to the device architecture and requirements (ARM vs x86).
Example Dockerfile (Raspberry Pi, TFLite):
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY edge/ ./edge/
COPY models/converted/ ./models/
EXPOSE 8080
CMD ["python", "edge/serve.py"]
```
- Build locally (match target arch):
  - For same-arch:
    docker build -t my-repo/ml-edge:latest .
  - For cross-arch or multi-arch, use `docker buildx` or build on-device.

## Push to registry (optional)
- Tag image and push:
  docker tag my-repo/ml-edge:latest ghcr.io/OWNER/ml-edge:latest
  docker push ghcr.io/OWNER/ml-edge:latest

## Deploy to the device
Options:
1. Docker run (simple):
   - Pull on the device:
     docker pull ghcr.io/OWNER/ml-edge:latest
   - Run:
     docker run -d --restart always \
       --name ml-edge \
       -p 8080:8080 \
       --device /dev/<accelerator> (if needed) \
       ghcr.io/OWNER/ml-edge:latest

2. Systemd service (auto-start):
   - Create `/etc/systemd/system/ml-edge.service`:
     ```
     [Unit]
     Description=ML Edge Service
     After=docker.service
     Requires=docker.service

     [Service]
     Restart=always
     ExecStart=/usr/bin/docker run --rm --name ml-edge -p 8080:8080 ghcr.io/OWNER/ml-edge:latest
     ExecStop=/usr/bin/docker stop ml-edge

     [Install]
     WantedBy=multi-user.target
     ```
   - Enable and start:
     sudo systemctl daemon-reload
     sudo systemctl enable ml-edge
     sudo systemctl start ml-edge

3. balena / Fleet management:
   - Use balenaCloud or Mender for fleet updates; push new images via their workflow.

4. Native (no container):
   - Install Python deps and run `python edge/serve.py` under a systemd service.

## Verify deployment
- Smoke test:
  curl -X POST http://DEVICE:8080/infer -F 'image=@sample.jpg'
- Check logs:
  - Docker: docker logs -f ml-edge
  - Systemd: sudo journalctl -u ml-edge -f
- Run unit/integration tests from `tests/` on-device if feasible.

## Monitoring & health
- Expose a /health endpoint returning status, uptime, loaded model version.
- Add basic metrics (latency, request counts) and forward to a lightweight collector (Prometheus pushgateway, Influx, or cloud).
- Keep logs accessible (rotate logs or use centralized logging).

## Updating & rollback
- Use image tags for versioning (e.g., v1.0.0). To update:
  - Build/push new image -> pull on device -> docker stop/remove -> docker run new image
  - For automated updates, use orchestrator or deployment tool.
- Provide a tested rollback tag (previous stable image) and a script to revert.

## Security & hardening
- Run containers with least privileges (non-root user inside container).
- Secure API endpoints (API key, TLS/SSH tunnel, VPN).
- Lock down device SSH, rotate keys, enable firewall.

## Reproducibility
- Record exact commands and tool versions used for conversion in `scripts/convert_model.sh` or `notebooks/`.
- Store build artifacts (Dockerfile, model manifest) alongside the code.

## Troubleshooting tips
- If model fails to load, verify correct format and expected opset/runtime.
- If slow: enable quantization, use hardware accelerators, or consider TensorRT/Edge TPU builds.
- If out-of-memory: reduce batch size, use streaming inference, optimize model.
