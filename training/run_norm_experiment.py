"""
Normalization Experiment: Evaluates model performance on clean vs. 3-sigma shifted normalization stats.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Setup Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_FILE
from utils import load_training_dataset, load_training_stats, normalize_features, get_features_and_labels

# Load TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def evaluate_tflite(model_path: Path, X_norm: np.ndarray, y_true: np.ndarray):
    """Evaluates TFLite model and computes overall and per-class accuracies."""
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    for sample in X_norm:
        sample_tensor = np.expand_dims(sample, axis=0).astype(np.float32)

        # Quantization handling if INT8
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            sample_tensor = np.clip(np.round(sample_tensor / scale + zero_point), -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], sample_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]['dtype'] == np.int8:
            scale, zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        y_pred.append(np.argmax(output, axis=1)[0])

    y_pred = np.array(y_pred)

    # Calculate overall accuracy
    overall_acc = accuracy_score(y_true, y_pred) * 100.0

    # Calculate per-class accuracy (Recall per class)
    # Class 0: Normal, Class 1: Warning, Class 2: Critical/Anomaly
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0

    return overall_acc, recalls[0], recalls[1], recalls[2]


if __name__ == "__main__":
    # 1. Load Raw Dataset & Stats
    df = load_training_dataset(DATASET_FILE)
    X_raw, y = get_features_and_labels(df)
    mean, std = load_training_stats()

    # Split to extract validation set (20%)
    _, X_val_raw, _, y_val = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    # Path to trained TFLite model (FP32 baseline or PTQ)
    model_path = PROJECT_ROOT / "models" / "model_fp32.tflite"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "model_ptq.tflite"

    print(f"[INFO] Using model for experiment: {model_path.name}")

    # --- EXPERIMENT 1: BASELINE (Correct Normalization) ---
    X_val_baseline = normalize_features(X_val_raw, mean=mean, std=std)
    base_acc, base_c0, base_c1, base_c2 = evaluate_tflite(model_path, X_val_baseline, y_val)

    # --- EXPERIMENT 2: 3-SIGMA SHIFT (Mean Shifted by +3 * Std) ---
    shifted_mean = mean + (3.0 * std)
    X_val_shifted = normalize_features(X_val_raw, mean=shifted_mean, std=std)
    shift_acc, shift_c0, shift_c1, shift_c2 = evaluate_tflite(model_path, X_val_shifted, y_val)

    # --- EXPERIMENT 3: CALCULATE CHANGES (Percentage Points) ---
    diff_acc = shift_acc - base_acc
    diff_c0 = shift_c0 - base_c0
    diff_c1 = shift_c1 - base_c1
    diff_c2 = shift_c2 - base_c2

    # --- DISPLAY FORMATTED TABLE ---
    results = [
        {
            "Experiment": "Baseline",
            "Normalization statistics": "Correct training_stats.npy",
            "Validation/Test Accuracy": f"{base_acc:.2f}%",
            "Class 0 Accuracy": f"{base_c0:.2f}%",
            "Class 1 Accuracy": f"{base_c1:.2f}%",
            "Class 2 Accuracy": f"{base_c2:.2f}%",
            "Observation": "Reference performance"
        },
        {
            "Experiment": "3σ shifted",
            "Normalization statistics": "Mean/std shifted by 3σ",
            "Validation/Test Accuracy": f"{shift_acc:.2f}%",
            "Class 0 Accuracy": f"{shift_c0:.2f}%",
            "Class 1 Accuracy": f"{shift_c1:.2f}%",
            "Class 2 Accuracy": f"{shift_c2:.2f}%",
            "Observation": "Performance degradation due to distribution mismatch"
        },
        {
            "Experiment": "Change",
            "Normalization statistics": "—",
            "Validation/Test Accuracy": f"{diff_acc:+.2f} pp",
            "Class 0 Accuracy": f"{diff_c0:+.2f} pp",
            "Class 1 Accuracy": f"{diff_c1:+.2f} pp",
            "Class 2 Accuracy": f"{diff_c2:+.2f} pp",
            "Observation": "Demonstrates sensitivity to incorrect normalisation"
        }
    ]

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 120)
    print("NORMALIZATION EXPERIMENT RESULTS")
    print("=" * 120)
    print(df_res.to_string(index=False))
    print("=" * 120 + "\n")