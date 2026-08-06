"""
Task E1: Population Stability Index (PSI) Monitoring Module.
Tracks prediction confidence distributions over rolling windows and flags data drift.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_FILE, MODEL_DIR, OUTPUT_DIR, FEATURE_COLUMNS, TARGET_COLUMN
from utils import load_training_dataset, normalize_features

# TFLite Loader
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Define 4 specific confidence score bins
BINS = [0.0, 0.25, 0.50, 0.75, 1.0]
REF_DIST_FILE = OUTPUT_DIR / "reference_dist.json"


def get_bin_percentages(confidence_scores: np.ndarray) -> np.ndarray:
    """Compute percentage of confidence scores in 4 discrete bins."""
    counts, _ = np.histogram(confidence_scores, bins=BINS)
    # Add small epsilon (1e-4) to prevent division by zero in log calculations
    percentages = (counts + 1e-4) / (len(confidence_scores) + 4e-4)
    return percentages


def compute_psi(actual_dist: np.ndarray, expected_dist: np.ndarray) -> float:
    """Calculate Population Stability Index (PSI) between two distributions."""
    psi_value = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
    return float(psi_value)


def generate_reference_distribution(model_path: Path):
    """Run inference on 300 clean Normal-class windows and save reference distribution."""
    print("[INFO] Generating PSI reference distribution from clean Normal samples...")
    X_raw, y = load_training_dataset(DATASET_FILE)
    X_norm = normalize_features(X_raw)

    # Filter for 300 clean Normal-class (class 0) samples
    normal_indices = np.where(y == 0)[0][:300]
    normal_samples = X_norm[normal_indices]

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    confidences = []
    for sample in normal_samples:
        input_data = np.expand_dims(sample, axis=0).astype(np.float32)

        # Handle INT8 Model Inputs
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Handle INT8 Model Outputs
        if output_details[0]['dtype'] == np.int8:
            scale, zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        conf = float(np.max(output))
        confidences.append(conf)

    ref_dist = get_bin_percentages(np.array(confidences)).tolist()
    
    with open(REF_DIST_FILE, "w") as f:
        json.dump({"bins": BINS, "reference_distribution": ref_dist}, f, indent=4)
        
    print(f"[SUCCESS] Saved reference distribution to: {REF_DIST_FILE}")
    print(f"[INFO] Reference Distribution Bins: {ref_dist}")
    return ref_dist


def run_psi_monitor(model_path: Path, anomaly_mode: bool = False):
    """Monitor rolling window of 100 predictions every 60s and flag drift."""
    if not REF_DIST_FILE.exists():
        expected_dist = np.array(generate_reference_distribution(model_path))
    else:
        with open(REF_DIST_FILE, "r") as f:
            expected_dist = np.array(json.load(f)["reference_distribution"])

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    X_raw, y = load_training_dataset(DATASET_FILE)
    X_norm = normalize_features(X_raw)

    print("\n[INFO] Starting Continuous PSI Monitoring Service (Interval: 60s)...")
    
    # Simulate streaming inference window
    for step in range(1, 10):
        # Sample 100 instances
        indices = np.random.choice(len(X_norm), 100, replace=False)
        window = X_norm[indices].copy()

        # Inject combined anomaly to trigger drift if specified
        if anomaly_mode and step >= 3:
            # Simulate sensor degradation / spike
            window[:, 0] += 5.0  # Temperature spike
            window[:, 3] += 2.5  # Vibration RMS spike

        confidences = []
        for sample in window:
            input_data = np.expand_dims(sample, axis=0).astype(np.float32)
            if input_details[0]['dtype'] == np.int8:
                scale, zero_point = input_details[0]['quantization']
                input_data = (input_data / scale + zero_point).astype(np.int8)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            if output_details[0]['dtype'] == np.int8:
                scale, zero_point = output_details[0]['quantization']
                output = (output.astype(np.float32) - zero_point) * scale

            confidences.append(float(np.max(output)))

        actual_dist = get_bin_percentages(np.array(confidences))
        psi = compute_psi(actual_dist, expected_dist)

        print(f"[MONITOR] Timestamp: {time.strftime('%H:%M:%S')} | Window: {step} | PSI = {psi:.4f}")

        if psi > 0.25:
            print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")
        elif psi < 0.10:
            print(f"[INFO] System Stable: Normal Distribution (PSI={psi:.3f})")

        time.sleep(2)  # Reduced delay for execution demo; set to 60 for production


if __name__ == "__main__":
    from config import PTQ_MODEL_FILE
    
    # 1. Generate Reference JSON
    generate_reference_distribution(PTQ_MODEL_FILE)
    
    # 2. Run Monitor Demo with Anomaly Injection
    run_psi_monitor(PTQ_MODEL_FILE, anomaly_mode=True)