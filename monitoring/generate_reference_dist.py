import json
from pathlib import Path
import numpy as np
import pandas as pd

# TFLite Import
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model_ptq.tflite"
NORMAL_CSV_PATH = PROJECT_ROOT / "data" / "normal_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "monitoring" / "reference_dist.json"

# Import Project Helpers
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_training_stats, normalize_features, FEATURE_COLUMNS


def generate_reference_distribution():
    """Runs TFLite inference on 300 clean Normal windows to generate reference_dist.json."""
    if not NORMAL_CSV_PATH.exists():
        raise FileNotFoundError(f"Clean normal feature dataset missing at {NORMAL_CSV_PATH}")

    df = pd.read_csv(NORMAL_CSV_PATH)
    
    # Extract first 300 clean windows
    X_normal = df[FEATURE_COLUMNS].values[:300]

    # Normalize features using saved stats
    mean, std = load_training_stats()
    X_norm = normalize_features(X_normal, mean, std)

    # Initialize TFLite Interpreter
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    confidence_scores = []

    for row in X_norm:
        data = np.expand_dims(row, axis=0).astype(np.float32)

        # Handle INT8 input quantization if required
        if input_details[0]["dtype"] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            data = np.round(data / scale + zero_point)
            data = np.clip(data, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # Handle INT8 output dequantization if required
        if output_details[0]["dtype"] == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        # Take peak confidence score
        confidence = float(np.max(output))
        confidence_scores.append(confidence)

    confidence_scores = np.array(confidence_scores)

    # Calculate frequency distribution across the 4 specified bins
    bins = [0.0, 0.25, 0.50, 0.75, 1.0]
    counts, _ = np.histogram(confidence_scores, bins=bins)
    proportions = (counts / len(confidence_scores)).tolist()

    reference_data = {
        "bins": ["[0, 0.25)", "[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.0]"],
        "counts": counts.tolist(),
        "proportions": proportions,
        "sample_size": len(confidence_scores),
    }

    # Save reference_dist.json inside monitoring/
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(reference_data, f, indent=4)

    print(f"[SUCCESS] Reference distribution saved to: {OUTPUT_PATH}")
    print(f"Proportions across 4 bins: {proportions}")


if __name__ == "__main__":
    generate_reference_distribution()