"""
Train MLP classifier for Cold Chain Monitoring.

Task D1
--------
Input:
    data/training_dataset.csv
    data/training_stats.npy

Output:
    models/best_model.keras

Architecture:
    Input (6)
        ↓
    Dense(32, ReLU)
        ↓
    Dense(16, ReLU)
        ↓
    Dense(3, Softmax)
"""



import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.tensorflow
from utils import setup_mlflow
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


from config import (
    MODEL_FILE,
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

df = load_training_dataset()
print_dataset_info(df)
X, y = get_features_and_labels(df)


# =====================================================
# Load Saved Statistics
# IMPORTANT: never recompute statistics from training data.
# Always use training_stats.npy.
# =====================================================

mean, std = load_training_stats()

X = normalize_features(
    X,
    mean,
    std
)


# =====================================================
# Train / Validation Split
# =====================================================

X_train, X_valid, y_train, y_valid = split_dataset(
    X,
    y
)


print()

print("Training samples :", len(X_train))
print("Validation samples :", len(X_valid))


# =====================================================
# Build Model
# =====================================================

model = tf.keras.Sequential(

    [

        tf.keras.layers.Input(
            shape=(6,)
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            3,
            activation="softmax"
        )

    ]

)


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


model.summary()


# =====================================================
# Callbacks
# =====================================================

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)


# =====================================================
# Train
# =====================================================

TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", 100))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_valid,
        y_valid
    ),

    epochs=TRAINING_EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stop
    ],

    verbose=1

)


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =====================================================
# Validation
# =====================================================

y_prob = model.predict(
    X_valid,
    verbose=0
)

y_pred = np.argmax(
    y_prob,
    axis=1
)

accuracy = accuracy_score(
    y_valid,
    y_pred
)

print("\nValidation Accuracy : {:.2f}%".format(
    accuracy * 100
))

print("\nClassification Report\n")

print(
    classification_report(
        y_valid,
        y_pred,
        target_names=CLASS_NAMES
    )
)


# =====================================================
# Assignment Requirement
# =====================================================

if accuracy < 0.88:

    raise RuntimeError(

        "\nValidation accuracy is below the required "
        "88%.\n"
        "Improve feature extraction or model architecture."

    )


# =====================================================
# Save Model
# =====================================================

save_keras_model(
    model,
    MODEL_FILE
)

print(
    f"\nModel saved : {MODEL_FILE}"
)


# =====================================================
# Confusion Matrix
# =====================================================
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(
    y_valid,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

plt.tight_layout()


CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"
TRAINING_CURVE = OUTPUT_DIR / "training_curve.png"

plt.savefig(
    CONFUSION_MATRIX,
    dpi=300
)

plt.savefig(
    TRAINING_CURVE,
    dpi=300
)

plt.close()


# =====================================================
# MLflow Logging
# =====================================================

with mlflow.start_run(
    run_name="MLP_Training"
):

    # -------------------------
    # Parameters
    # -------------------------
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
    "training_epochs": TRAINING_EPOCHS,
     })

    # -------------------------
    # Metrics
    # -------------------------

    log_metrics({

    "validation_accuracy": accuracy,
    "best_validation_loss": min(history.history["val_loss"]),
    "best_training_loss": min(history.history["loss"]),
    "epochs_trained": len(history.history["loss"])
    })

    # -------------------------
    # Artifacts
    # -------------------------

    log_artifact(CONFUSION_MATRIX)
    log_artifact(MODEL_FILE)

print("\nTraining completed successfully.")
print("Validation Accuracy : {:.2f}%".format(
    accuracy * 100
))