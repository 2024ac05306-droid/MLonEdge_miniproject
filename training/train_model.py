"""
Train MLP classifier for Cold Chain Monitoring.

Task D1
--------
Input:
    data/training_dataset.csv
    data/training_stats.npy

Output:
    models/best_model.keras
"""

import os
from pathlib import Path
import sys

# -------------------------------------------------------------------
# Environment Setup for Legacy Keras (tf_keras)
# MUST be set before importing tensorflow or keras
# -------------------------------------------------------------------
os.environ["TF_USE_LEGACY_KERAS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

# Explicitly import legacy Keras
import tf_keras as keras

from config import (
    MODEL_FILE,
    DATASET_FILE,
    TRAINING_EPOCHS,
    BATCH_SIZE,
    FEATURE_COLUMNS,
    CLASS_NAMES,
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
    print_dataset_info,
    print_model_results,
)

# =====================================================
# Environment
# =====================================================

setup_mlflow()


# =====================================================
# Load Dataset
# =====================================================

df = load_training_dataset(DATASET_FILE)
print_dataset_info(df)



# =====================================================
# Load Saved Statistics
# =====================================================

mean, std = load_training_stats()
X = normalize_features(X, mean, std)


# =====================================================
# Train / Validation Split
# =====================================================

X_train, X_valid, y_train, y_valid = split_dataset(X, y)

print(f"\nTraining samples   : {len(X_train)}")
print(f"Validation samples : {len(X_valid)}")
print(f"Feature columns    : {len(FEATURE_COLUMNS)}")
print(f"Class names        : {CLASS_NAMES}")


# =====================================================
# Build Model (using tf_keras)
# =====================================================

model = keras.Sequential([
    keras.layers.Input(shape=(6,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# =====================================================
# Callbacks
# =====================================================

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)


# =====================================================
# Train
# =====================================================

TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", 100))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=TRAINING_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =====================================================
# # Evaluation & Assignment Quality Gate
# =====================================================

y_prob = model.predict(X_valid, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

accuracy = accuracy_score(y_valid, y_pred)

print(f"\nValidation Accuracy : {accuracy * 100:.2f}%")
print("\nClassification Report\n")
print(classification_report(y_valid, y_pred, target_names=CLASS_NAMES))


# =====================================================
# Assignment Requirement
# Mandatory Quality Threshold Check (Task D1)
# =====================================================

if accuracy < 0.88:
    raise RuntimeError(
        "\nValidation accuracy is below the required 88%.\n"
        "Improve feature extraction or model architecture."
    )


# =====================================================
# Save Model
# =====================================================

save_keras_model(model, MODEL_FILE)
print(f"\nModel saved : {MODEL_FILE}")


# =====================================================
# Confusion Matrix
# =====================================================

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout()

CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"
TRAINING_CURVE = OUTPUT_DIR / "training_curve.png"

plt.savefig(CONFUSION_MATRIX, dpi=300)
plt.savefig(TRAINING_CURVE, dpi=300)
plt.close()


# =====================================================
# MLflow Logging
# =====================================================

with mlflow.start_run(run_name="MLP_Training"):
    log_params({
        "input_features": len(FEATURE_COLUMNS),
        "hidden_layer_1": 32,
        "hidden_layer_2": 16,
        "activation": "ReLU",
        "learning_rate": 0.001,
        "training_epochs": TRAINING_EPOCHS,
        "loss_function": "sparse_categorical_crossentropy",
        "optimizer": "Adam",
        "batch_size": BATCH_SIZE,
    })

    log_metrics({
        "validation_accuracy": accuracy,
        "best_validation_loss": min(history.history["val_loss"]),
        "best_training_loss": min(history.history["loss"]),
        "epochs_trained": len(history.history["loss"]),
    })

    log_artifact(CONFUSION_MATRIX)
    log_artifact(MODEL_FILE)

print("\nTraining completed successfully.")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")

def print_model_results(loss, accuracy):
    """
    Display evaluation results.
    """

    print("=" * 50)
    print("Evaluation")
    print("=" * 50)
    print(f"Loss     : {loss:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print("=" * 50)