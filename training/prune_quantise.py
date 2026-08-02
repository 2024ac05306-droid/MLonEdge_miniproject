"""
prune_quantise.py

Prunes the trained MLP model using TensorFlow Model Optimization Toolkit
and saves the optimized model for deployment.

"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------

load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "G44_logibridge_miniproject"
)

DATASET_DIR = Path(os.getenv("DATASET_DIR", "./data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))

DATASET_FILE = DATASET_DIR / os.getenv(
    "DATASET_FILE",
    "training_dataset.csv"
)

TRAINING_STATS = DATASET_DIR / os.getenv(
    "TRAINING_STATS",
    "training_stats.npy"
)

MODEL_FILE = MODEL_DIR / os.getenv(
    "MODEL_NAME",
    "best_model.keras"
)

PRUNED_MODEL_FILE = MODEL_DIR / os.getenv(
    "PRUNED_MODEL_NAME",
    "best_model_pruned.keras"
)

TEST_SIZE = float(os.getenv("TEST_SIZE", 0.20))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 10))

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Configure MLflow
# -------------------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------

print("Loading training dataset...")

df = pd.read_csv(DATASET_FILE)

print(f"Dataset Shape : {df.shape}")


FEATURE_COLUMNS = [

    # Copy the exact feature list from train_model.py

    "temperature_mean",
    "temperature_std",
    "temperature_min",
    "temperature_max",

    "vibration_mean",
    "vibration_std",
    "vibration_min",
    "vibration_max"

]

TARGET_COLUMN = "label"

X = df[FEATURE_COLUMNS].values.astype(np.float32)
y = df[TARGET_COLUMN].values.astype(np.int32)

# -------------------------------------------------------------------
# Load Normalization Statistics
# -------------------------------------------------------------------

print("Loading normalization statistics...")

stats = np.load(
    TRAINING_STATS,
    allow_pickle=True
).item()

mean = stats["mean"]
std = stats["std"]

std = np.where(std == 0, 1.0, std)

X = (X - mean) / std

# -------------------------------------------------------------------
# Split Dataset
# -------------------------------------------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training Samples   : {len(X_train)}")
print(f"Validation Samples : {len(X_valid)}")

# -------------------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------------------

print("Loading trained MLP model...")

model = load_model(MODEL_FILE)

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
) * EPOCHS

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

    optimizer=tf.keras.optimizers.Adam(),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=["accuracy"]

)

# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------

callbacks = [

    tfmot.sparsity.keras.UpdatePruningStep(),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=5,

        restore_best_weights=True

    )

]

# -------------------------------------------------------------------
# Fine Tune
# -------------------------------------------------------------------

print("\nFine-tuning pruned model...")
PRUNING_EPOCHS = int(os.getenv("PRUNING_EPOCHS", 10))

history = pruned_model.fit(

    X_train,

    y_train,

    validation_data=(

        X_valid,

        y_valid

    ),

    batch_size=BATCH_SIZE,

    epochs=PRUNING_EPOCHS,

    callbacks=callbacks,

    verbose=1

)

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

print("\nSaving pruned model...")

final_model.save(

    PRUNED_MODEL_FILE

)

print(

    f"Saved : {PRUNED_MODEL_FILE}"

)


# -------------------------------------------------------------------
# MLflow Logging
# -------------------------------------------------------------------

with mlflow.start_run(run_name="MLP_Model_Pruning"):

    mlflow.log_param("model_type", "MLP")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("training_epochs", EPOCHS)
    mlflow.log_param("pruning_epochs", PRUNING_EPOCHS)

    mlflow.log_metric(
        "original_accuracy",
        float(original_accuracy)
    )

    mlflow.log_metric(
        "pruned_accuracy",
        float(accuracy)
    )

    mlflow.log_metric(
        "accuracy_difference",
        float(accuracy - original_accuracy)
    )


# -------------------------------------------------------------------
# Model Size Comparison
# -------------------------------------------------------------------

original_size = MODEL_FILE.stat().st_size / (1024 * 1024)
pruned_size = PRUNED_MODEL_FILE.stat().st_size / (1024 * 1024)

print("\nModel Size")

print("---------------------------")

print(f"Original : {original_size:.2f} MB")

print(f"Pruned   : {pruned_size:.2f} MB")

mlflow.log_metric(
    "original_model_size_mb",
    original_size
)

mlflow.log_metric(
    "pruned_model_size_mb",
    pruned_size
)

# -------------------------------------------------------------------
# Log Artifacts
# -------------------------------------------------------------------

mlflow.log_artifact(
    str(PRUNED_MODEL_FILE)
)

mlflow.keras.log_model(
    final_model,
    artifact_path="pruned_model"
)

print("\nModel logged to MLflow.")

mlflow.end_run()

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

