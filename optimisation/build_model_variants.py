"""
Task F1: Build and export three model variants (M1, M2, M3) for edge benchmarking.
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_FILE, MODEL_DIR, FEATURE_COLUMNS, TARGET_COLUMN
from utils import load_training_dataset, load_training_stats, normalize_features

MODEL_DIR.mkdir(parents=True, exist_ok=True)

M1_PATH = MODEL_DIR / "model_fp32.tflite"
M2_PATH = MODEL_DIR / "model_ptq.tflite"
M3_PATH = MODEL_DIR / "model_pruned_ptq.tflite"


def build_base_keras_model(input_shape=(6,), num_classes=3):
    """Build baseline multi-layer perceptron architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape, name="dense_1"),
        tf.keras.layers.BatchNormalization(name="bn_1"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', name="dense_2"),
        tf.keras.layers.BatchNormalization(name="bn_2"),
        tf.keras.layers.Dense(num_classes, activation='softmax', name="output")
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_representative_dataset(X_data: np.ndarray, num_samples: int = 250):
    """Generates a calibration dataset generator for full INT8 post-training quantization."""
    # Ensure float32 dtype
    X_data = X_data.astype(np.float32)
    
    # Shuffle or randomly sample indices to represent true data distribution
    num_samples = min(num_samples, len(X_data))
    np.random.seed(42)  # Fixed seed for reproducible quantization
    indices = np.random.choice(len(X_data), size=num_samples, replace=False)
    calibration_data = X_data[indices]

    def representative_dataset_gen():
        for sample in calibration_data:
            # TFLite expects shape (1, num_features) with batch dimension
            yield [np.expand_dims(sample, axis=0)]

    return representative_dataset_gen


def export_m1_fp32(keras_model):
    """Variant M1: Convert base model to standard FP32 TFLite format."""
    print("\n--- Exporting Variant M1: FP32 Baseline ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    with open(M1_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"M1 Saved: {M1_PATH} ({len(tflite_model) / 1024:.2f} KB)")


def export_m2_ptq(keras_model, rep_gen):
    """Variant M2: Convert to Full INT8 Post-Training Quantized (PTQ) TFLite."""
    print("\n--- Exporting Variant M2: PTQ INT8 ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(M2_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"M2 Saved: {M2_PATH} ({len(tflite_model) / 1024:.2f} KB)")


def export_m3_pruned_ptq(X_train, y_train, rep_gen):
    """Variant M3: Apply 35% structured filter pruning with PolynomialDecay, then PTQ INT8."""
    print("\n--- Training & Exporting Variant M3: Pruned 35% + PTQ INT8 ---")
    
    base_model = build_base_keras_model(input_shape=(X_train.shape[1],))
    epochs = 10
    batch_size = 32
    num_train_samples = len(X_train)
    end_step = np.ceil(num_train_samples / batch_size).astype(np.int32) * epochs

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.35,  # 35% target pruning
            begin_step=0,
            end_step=end_step
        )
    }

    model_for_pruning = prune_low_magnitude(base_model, **pruning_params)
    model_for_pruning.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model_for_pruning.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # Strip pruning wrappers for export
    stripped_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    # Convert stripped model with Full INT8 PTQ
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(M3_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"M3 Saved: {M3_PATH} ({len(tflite_model) / 1024:.2f} KB)")


if __name__ == "__main__":
    # Initialize MLflow Experiment
    mlflow.set_experiment("Task_F1_Model_Export")

    with mlflow.start_run(run_name="Build_and_Export_Variants"):
        # Data loading and preprocessing
        X_raw, y = load_training_dataset(DATASET_FILE)
        mean, std = load_training_stats()
        X_norm = normalize_features(X_raw, mean=mean, std=std)

        X_train, X_val, y_train, y_val = train_test_split(
            X_norm, y, test_size=0.20, random_state=42, stratify=y
        )

        # Log parameters
        epochs = 15
        batch_size = 32
        mlflow.log_params({
            "test_split": 0.20,
            "random_state": 42,
            "batch_size": batch_size,
            "base_epochs": epochs,
            "pruning_epochs": 10,
            "pruning_target_sparsity": 0.35,
            "num_calibration_samples": 250
        })

        # Train Base Keras Model
        base_model = build_base_keras_model(input_shape=(X_train.shape[1],))
        history = base_model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=0, 
            validation_data=(X_val, y_val)
        )

        # Log training & validation metrics
        mlflow.log_metric("base_final_train_acc", history.history['accuracy'][-1])
        mlflow.log_metric("base_final_val_acc", history.history['val_accuracy'][-1])

        # Export variants and generate artifacts
        p1 = export_m1_fp32(base_model)
        p2 = export_m2_ptq(base_model, get_representative_dataset(X_train, num_samples=250))
        p3 = export_m3_pruned_ptq(X_train, y_train, get_representative_dataset(X_train, num_samples=250))

        # Log .tflite artifacts to MLflow
        mlflow.log_artifact(str(p1), artifact_path="tflite_models")
        mlflow.log_artifact(str(p2), artifact_path="tflite_models")
        mlflow.log_artifact(str(p3), artifact_path="tflite_models")

        print("\n[SUCCESS] Model building, export, and MLflow tracking complete.")