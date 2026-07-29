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

from tensorflow import keras
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf


import matplotlib.pyplot as plt


from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


# =====================================================
# Environment
# =====================================================

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "./mlruns"
)

MLFLOW_EXPERIMENT_NAME = (
    "ColdChain_Model_Training"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)


# =====================================================
# Paths
# =====================================================

DATA_DIR = Path("data")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DATASET_FILE = (
    DATA_DIR /
    "training_dataset.csv"
)

STATS_FILE = (
    DATA_DIR /
    "training_stats.npy"
)

MODEL_FILE = (
    MODEL_DIR /
    "best_model.keras"
)


FEATURE_COLUMNS = [

    "temp_mean",
    "temp_std",
    "temp_rate",
    "vibration_rms",
    "vibration_peak",
    "vibration_kurtosis"

]


# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(DATASET_FILE)

X = df[FEATURE_COLUMNS].values.astype(np.float32)

y = df["label"].values.astype(np.int32)


# =====================================================
# Load Saved Statistics
# =====================================================

stats = np.load(
    STATS_FILE,
    allow_pickle=True
).item()

mean = stats["mean"]
std = stats["std"]

# IMPORTANT:
# Never recompute statistics from training data.
# Always use training_stats.npy.

X = (X - mean) / std


# =====================================================
# Train / Validation Split
# =====================================================

X_train, X_valid, y_train, y_valid = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

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

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_valid,
        y_valid
    ),

    epochs=100,

    batch_size=16,

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
        target_names=[
            "Normal",
            "Warning",
            "Critical",
        ]
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

model.save(
    MODEL_FILE
)

print(
    f"\nModel saved : {MODEL_FILE}"
)


# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(
    y_valid,
    y_pred
)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Normal",
        "Warning",
        "Critical",
    ]

)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

plt.tight_layout()

CONFUSION_MATRIX = (
    MODEL_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    CONFUSION_MATRIX,
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

    mlflow.log_param(
        "input_features",
        6
    )

    mlflow.log_param(
        "hidden_layer_1",
        32
    )

    mlflow.log_param(
        "hidden_layer_2",
        16
    )

    mlflow.log_param(
        "activation",
        "ReLU"
    )

    mlflow.log_param(
        "output_classes",
        3
    )

    mlflow.log_param(
        "optimizer",
        "Adam"
    )

    mlflow.log_param(
        "learning_rate",
        0.001
    )

    mlflow.log_param(
        "batch_size",
        16
    )

    mlflow.log_param(
        "epochs_requested",
        100
    )

    mlflow.log_param(
        "early_stopping",
        True
    )

    # -------------------------
    # Metrics
    # -------------------------

    mlflow.log_metric(
        "validation_accuracy",
        float(accuracy)
    )

    mlflow.log_metric(
        "best_validation_loss",
        float(
            min(history.history["val_loss"])
        )
    )

    mlflow.log_metric(
        "best_training_loss",
        float(
            min(history.history["loss"])
        )
    )

    mlflow.log_metric(
        "epochs_trained",
        len(history.history["loss"])
    )

    # -------------------------
    # Artifacts
    # -------------------------

    mlflow.log_artifact(
        str(CONFUSION_MATRIX)
    )

    mlflow.log_artifact(
        str(MODEL_FILE)
    )

    # TensorFlow model
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model"
    )

print("\nTraining completed successfully.")
print("Validation Accuracy : {:.2f}%".format(
    accuracy * 100
))