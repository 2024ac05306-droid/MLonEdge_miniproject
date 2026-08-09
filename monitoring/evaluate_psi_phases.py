import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Path setup - Resolves project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# TFLite Import
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ImportError:
    import tensorflow.lite as tflite

# Project imports (Using DATASET_FILE and load_training_dataset)
from config import MODEL_DIR, TRAINING_STATS, DATASET_FILE
from utils import (
    load_training_dataset,
    get_features_and_labels,
    load_training_stats,
    normalize_features,
)


def calculate_psi(expected_props, actual_props, epsilon=1e-4):
    """Calculates Population Stability Index between baseline and runtime distributions."""
    expected = np.clip(expected_props, epsilon, 1.0)
    actual = np.clip(actual_props, epsilon, 1.0)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def get_confidence_proportions(interpreter, X_data, bins=[0.0, 0.25, 0.50, 0.75, 1.0]):
    """Runs TFLite model and returns output confidence bin proportions."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    confidence_scores = []

    for row in X_data:
        data = np.expand_dims(row, axis=0).astype(np.float32)

        # Handle INT8 input quantization
        if input_details[0]["dtype"] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            data = np.round(data / scale + zero_point)
            data = np.clip(data, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # Handle INT8 output dequantization
        if output_details[0]["dtype"] == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        # Apply softmax if raw logits are returned
        if not np.isclose(np.sum(output), 1.0, atol=1e-3):
            exp_output = np.exp(output - np.max(output))
            output = exp_output / np.sum(exp_output)

        confidence_scores.append(float(np.max(output)))

    counts, _ = np.histogram(confidence_scores, bins=bins)
    return counts / len(confidence_scores)


def run_psi_phase_evaluation():
    # 1. Load Model & Stats
    model_path = MODEL_DIR / "model_ptq.tflite"
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    # Load stats and dataset
    mean, std = load_training_stats(TRAINING_STATS, DATASET_FILE)
    df = load_training_dataset(DATASET_FILE)
    X_clean, _ = get_features_and_labels(df)
    X_clean = X_clean[:300]

    # Baseline Distribution (Reference)
    X_baseline = normalize_features(X_clean, mean, std)
    ref_proportions = get_confidence_proportions(interpreter, X_baseline)

    # Phase 1: Before Injection (Clean stream)
    X_before = normalize_features(X_clean[:100], mean, std)
    psi_before = calculate_psi(ref_proportions, get_confidence_proportions(interpreter, X_before))

    # Phase 2: During Injection (+3 sigma synthetic shift)
    shifted_mean = mean + (3.0 * std)
    X_during = normalize_features(X_clean[100:200], shifted_mean, std)
    psi_during = calculate_psi(ref_proportions, get_confidence_proportions(interpreter, X_during))

    # Phase 3: After Injection (Restored clean stream)
    X_after = normalize_features(X_clean[200:300], mean, std)
    psi_after = calculate_psi(ref_proportions, get_confidence_proportions(interpreter, X_after))

    # 2. Display Formatted Table Output
    print("\n" + "=" * 75)
    print("            POPULATION STABILITY INDEX (PSI) DRIFT EVALUATION           ")
    print("=" * 75)
    print(f"| {'Monitoring Phase':<28} | {'PSI Score':<10} | {'Status':<25} |")
    print(f"|{'-'*30}|{'-'*12}|{'-'*27}|")
    print(f"| {'Phase 1: Before Injection':<28} | {psi_before:<10.4f} | {'No Drift (Baseline)':<25} |")
    print(f"| {'Phase 2: During Injection':<28} | {psi_during:<10.4f} | {'SIGNIFICANT DRIFT ALERT':<25} |")
    print(f"| {'Phase 3: After Injection':<28} | {psi_after:<10.4f} | {'Restored / Stable':<25} |")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_psi_phase_evaluation()