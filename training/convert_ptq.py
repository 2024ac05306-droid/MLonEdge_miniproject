"""
convert_ptq.py

Task F1
Post Training Quantization (PTQ)

Converts the trained Keras model to a fully INT8
TensorFlow Lite model using a representative dataset.
"""

from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

from config import (
    MODEL_PATH,
    PTQ_MODEL_PATH,
)

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
    Representative dataset used for calibration.
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

        model = load_keras_model(MODEL_PATH)

        # ---------------------------------------
        # TFLite Converter
        # ---------------------------------------

        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ]

        converter.representative_dataset = lambda: representative_dataset(X)

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

        PTQ_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(PTQ_MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        print("\nINT8 model saved")
        print(PTQ_MODEL_PATH)

        model_size = PTQ_MODEL_PATH.stat().st_size / 1024

        print(f"Model Size : {model_size:.2f} KB")

        # ---------------------------------------
        # MLflow
        # ---------------------------------------

        mlflow.log_param(
            "quantization",
            "INT8"
        )

        mlflow.log_param(
            "representative_samples",
            min(REPRESENTATIVE_SAMPLES, len(X))
        )

        mlflow.log_metric(
            "model_size_kb",
            model_size
        )

        log_artifact(PTQ_MODEL_PATH)

    print("\nPTQ conversion completed.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    convert_to_int8()