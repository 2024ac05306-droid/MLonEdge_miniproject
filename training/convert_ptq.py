"""
convert_ptq.py

Task F1
Post Training Quantization (PTQ)

Converts the trained Keras model to a fully INT8
TensorFlow Lite model using a representative dataset.
"""

import os
from pathlib import Path
import sys
import tempfile

# ---------------------------------------------------------
# Environment Setup for Legacy Keras (tf_keras)
# MUST be set before importing tensorflow or tf_keras
# ---------------------------------------------------------
os.environ["TF_USE_LEGACY_KERAS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import numpy as np
import tensorflow as tf
import tf_keras as keras

# Import correct variable names from config
try:
    from config import MODEL_FILE, PTQ_MODEL_FILE
except ImportError:
    # Fallback to default path names if PTQ_MODEL_FILE isn't in config.py yet
    from config import MODEL_FILE
    PTQ_MODEL_FILE = Path(MODEL_FILE).parent / "model_ptq.tflite"

from utils import (
    setup_mlflow,
    load_training_dataset,
    get_features_and_labels,
    load_training_stats,
    normalize_features,
    load_keras_model,
    log_artifact,
)

# ---------------------------------------------------------
# Representative Dataset
# ---------------------------------------------------------

REPRESENTATIVE_SAMPLES = 200


def representative_dataset(X):
    """
    Representative dataset used for INT8 calibration.
    """
    num_samples = min(REPRESENTATIVE_SAMPLES, len(X))

    for i in range(num_samples):
        sample = X[i].astype(np.float32)
        sample = np.expand_dims(sample, axis=0)
        yield [sample]


# ---------------------------------------------------------
# PTQ Conversion
# ---------------------------------------------------------

def convert_to_int8():

    print("=" * 60)
    print("POST TRAINING QUANTIZATION")
    print("=" * 60)

    # ---------------------------------------
    # MLflow
    # ---------------------------------------

    setup_mlflow()

    with mlflow.start_run(run_name="PostTrainingQuantization"):

        # ---------------------------------------
        # Load Dataset
        # ---------------------------------------

        df = load_training_dataset()
        X, _ = get_features_and_labels(df)

        # ---------------------------------------
        # Normalize using training statistics
        # ---------------------------------------

        mean, std = load_training_stats()
        X = normalize_features(X, mean, std)
        print(f"Calibration samples : {min(REPRESENTATIVE_SAMPLES, len(X))}")

        # ---------------------------------------
        # Load trained model
        # ---------------------------------------

        model_path = Path(MODEL_FILE)
        model = load_keras_model(model_path)

        # ---------------------------------------
        # TFLite Converter
        # Save to temporary SavedModel directory for full compatibility
        # ---------------------------------------

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_model_dir = Path(tmp_dir) / "saved_model"
            keras.models.save_model(model, saved_model_dir, save_format="tf")

            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: representative_dataset(X)

            # Enforce full INT8 quantization
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]

            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            print("\nConverting model...")
            tflite_model = converter.convert()

        # ---------------------------------------
        # Save model
        # ---------------------------------------

        output_path = Path(PTQ_MODEL_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        print("\nINT8 model saved")
        print(output_path)
        model_size = output_path.stat().st_size / 1024
        print(f"Model Size : {model_size:.2f} KB")

        # ---------------------------------------
        # MLflow Logging
        # ---------------------------------------

        mlflow.log_param("quantization", "INT8")
        mlflow.log_param("representative_samples", min(REPRESENTATIVE_SAMPLES, len(X)))
        mlflow.log_metric("model_size_kb", model_size)

        log_artifact(output_path)

    print("\nPTQ conversion completed.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    convert_to_int8()