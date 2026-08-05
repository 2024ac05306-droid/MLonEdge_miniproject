"""
prune_quantise.py

Prunes the trained MLP model using TensorFlow Model Optimization Toolkit
and saves the optimized model for deployment.

"""

import os
from pathlib import Path
import sys
import numpy as np 

# Ensure TensorFlow does NOT try to use the legacy tf.keras shim
# Some local environments set TF_USE_LEGACY_KERAS=True which makes
# TensorFlow expect the external `tf_keras` package. That causes
# `tf.keras` to fail to initialize and produces the ImportError seen
# in CI/local runs. Unset it here before importing TensorFlow.
os.environ.pop('TF_USE_LEGACY_KERAS', None)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import tensorflow as tf
keras = tf.keras
import tensorflow_model_optimization as tfmot


from config import (
    MODEL_FILE,
    PRUNED_MODEL_FILE,
    TRAINING_EPOCHS,
    PRUNING_EPOCHS,
    BATCH_SIZE,
)

from utils import (
    setup_mlflow,
    load_training_dataset,
    get_features_and_labels,
    load_training_stats,
    normalize_features,
    split_dataset,
    save_keras_model,
    log_params,
    log_metrics,
    log_artifact,
)

# Avoid importing mlflow/tensorflow symbols before the runtime environment
# has finished initialising in some local shells.

# -------------------------------------------------------------------
# Load Environment Variables
# Configure MLflow
# -------------------------------------------------------------------

setup_mlflow()

# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------

print("Loading training dataset...")
df = load_training_dataset()
X, y = get_features_and_labels(df)
print(f"Dataset Shape : {df.shape}")

# -------------------------------------------------------------------
# Load Normalization Statistics
# -------------------------------------------------------------------

print("Loading normalization statistics...")
mean, std = load_training_stats()
X = normalize_features(
    X,
    mean,
    std
)

# -------------------------------------------------------------------
# Split Dataset
# -------------------------------------------------------------------

X_train, X_valid, y_train, y_valid = split_dataset(
    X,
    y
)

print(f"Training Samples   : {len(X_train)}")
print(f"Validation Samples : {len(X_valid)}")

# -------------------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------------------

print("Loading trained MLP model...")

model_path = Path(MODEL_FILE).resolve()

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found: {model_path}"
    )

try:
    model = keras.models.load_model(
        model_path,
        compile=False
    )

except Exception as exc:
    print(f"Primary model load failed: {exc}")

    if model_path.suffix != ".keras":
        fallback_path = model_path.with_suffix(".keras")
    else:
        fallback_path = model_path.with_suffix(".h5")

    if fallback_path.exists():
        print(f"Trying fallback model load: {fallback_path}")

        model = keras.models.load_model(
            fallback_path,
            compile=False
        )

    else:
        raise RuntimeError(
            f"Unable to load model from either "
            f"'{model_path}' or '{fallback_path}'."
        ) from exc

model.summary()

# -------------------------------------------------------------------
# Evaluate Original Model
# -------------------------------------------------------------------

print("\nEvaluating original model...")

original_loss, original_accuracy = model.evaluate(
    X_valid,
    y_valid,
    verbose=0
)

print(f"Original Accuracy : {original_accuracy:.4f}")
print(f"Original Loss     : {original_loss:.4f}")

# -------------------------------------------------------------------
# Apply Magnitude Pruning
# -------------------------------------------------------------------

print("\nApplying magnitude pruning...")
num_train_samples = X_train.shape[0]
end_step = int(
    np.ceil(num_train_samples / BATCH_SIZE)
) * TRAINING_EPOCHS

pruning_params = {
    "pruning_schedule":
        tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30,
            final_sparsity=0.80,
            begin_step=0,
            end_step=end_step
        )
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    **pruning_params
)

# -------------------------------------------------------------------
# Compile Pruned Model
# -------------------------------------------------------------------

pruned_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"])

# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
     keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )]

# -------------------------------------------------------------------
# Fine Tune
# -------------------------------------------------------------------

print("\nFine-tuning pruned model...")

try:
    history = pruned_model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        epochs=PRUNING_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

except Exception as e:
    raise RuntimeError(
        f"Error during pruning fine-tuning: {e}"
    ) from e
# -------------------------------------------------------------------
# Evaluate Pruned Model
# -------------------------------------------------------------------

loss, accuracy = pruned_model.evaluate(
    X_valid,
    y_valid,
    verbose=0
)

print("\nPruned Model Results")
print("----------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Loss     : {loss:.4f}")

# -------------------------------------------------------------------
# Strip Pruning Wrappers
# -------------------------------------------------------------------

print("\nRemoving pruning wrappers...")
final_model = tfmot.sparsity.keras.strip_pruning(
    pruned_model
)

# -------------------------------------------------------------------
# Save Model
# -------------------------------------------------------------------

save_keras_model(
    final_model,
    PRUNED_MODEL_FILE
)  

print(
    f"Saved : {PRUNED_MODEL_FILE}"
)
if final_model is not None:
    final_model.summary()
if not Path(PRUNED_MODEL_FILE).exists():
    raise RuntimeError("Failed to save pruned model.")

# -------------------------------------------------------------------
# Model Size Comparison
# -------------------------------------------------------------------

original_size = Path(MODEL_FILE).resolve().stat().st_size / (1024 * 1024)
pruned_size = Path(PRUNED_MODEL_FILE).resolve().stat().st_size / (1024 * 1024)

print("\nModel Size")
print("---------------------------")
print(f"Original : {original_size:.2f} MB")
print(f"Pruned   : {pruned_size:.2f} MB")

# -------------------------------------------------------------------
# MLflow Logging
# -------------------------------------------------------------------

with mlflow.start_run(run_name="MLP_Model_Pruning"):

    log_metrics({
        "original_accuracy": original_accuracy,
        "pruned_accuracy": accuracy,
        "accuracy_difference": accuracy - original_accuracy,
        "original_model_size_mb": original_size,
        "pruned_model_size_mb": pruned_size,
    })

    mlflow.keras.log_model(
        keras_model=final_model,
        artifact_path="pruned_model"
    )
    log_params({
        "pruning_initial_sparsity": 0.30,
        "pruning_final_sparsity": 0.80,
        "training_epochs": TRAINING_EPOCHS,
        "pruning_epochs": PRUNING_EPOCHS,
        "batch_size": BATCH_SIZE,
    })
    log_artifact(MODEL_FILE)
    log_artifact(PRUNED_MODEL_FILE)

print("\nModel logged to MLflow.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    print("\n===================================")
    print("Model Pruning Completed Successfully")
    print("===================================")

    print(f"\nSaved Model : {PRUNED_MODEL_FILE}")

if __name__ == "__main__":
    main()
