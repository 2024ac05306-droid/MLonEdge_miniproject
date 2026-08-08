"""
utils.py

Common utility functions for the MLOnEdge project.
"""

import json
from pathlib import Path
import os
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from config import DATASET_FILE, FEATURE_COLUMNS, TARGET_COLUMN, STATS_FILE
 

from config import (
    DATASET_FILE,
    TRAINING_STATS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)


# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------

def setup_mlflow():
    """
    Configure MLflow tracking.
    """

    # Allow skipping MLflow setup for debugging or offline runs.
    if os.getenv("SKIP_MLFLOW", "0") == "1":
        print("SKIP_MLFLOW=1 -> skipping mlflow setup")
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
    """
    Load training dataset.
    """
def load_training_dataset(dataset_path=None):
    if dataset_path is None:
        dataset_path = DATASET_FILE

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found : {dataset_path}"
        )

    df = pd.read_csv(dataset_path)
    
    # Extract feature matrix (X) and target array (y)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    return X, y


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

def get_features_and_labels(df):
    """
    Split dataframe into features and labels.
    """

    X = df[FEATURE_COLUMNS].values.astype(np.float32)

    y = df[TARGET_COLUMN].values.astype(np.int32)

    return X, y


# ---------------------------------------------------------
# load_training_stats
# ---------------------------------------------------------
# Paths setup
# Path configuration
BASE_DIR = Path(__file__).resolve().parent
STATS_FILE = BASE_DIR / "data" / "training_stats.npy"


def save_normal_training_stats(X_clean_normal, file_path=STATS_FILE):
    """
    Computes mean and std strictly from clean Normal-class data (10 mins)
    and saves them to a .npy file during offline preprocessing.
    """
    mean = np.mean(X_clean_normal, axis=0)
    std = np.std(X_clean_normal, axis=0)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(file_path, {"mean": mean, "std": std})
    print(f"[INFO] Saved clean Normal-class training stats to {file_path}")
    return mean, std


def load_training_stats(file_path=STATS_FILE):
    """Loads pre-computed training mean and std from saved .npy file."""
    path = Path(file_path)
    
    if not path.exists():
        # Fallback check for root or Docker container directory
        fallback = BASE_DIR / "training_stats.npy"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Training stats file not found at {file_path} or {fallback}. "
                "Ensure training_stats.npy was generated during preprocessing."
            )

    stats = np.load(path, allow_pickle=True).item()
    return stats["mean"], stats["std"]



# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalize_features(X, mean=None, std=None):
    """Normalizes features using provided or loaded training statistics."""
    if mean is None or std is None:
        mean, std = load_training_stats()
        
    std_adjusted = np.where(std == 0, 1e-8, std)
    return (X - mean) / std_adjusted


# ---------------------------------------------------------
# Dataset Split
# ---------------------------------------------------------

def split_dataset(X, y):
    """
    Split dataset into training and validation sets.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

from pathlib import Path
import tensorflow as tf

def load_keras_model(model_path):
    """
    Load a TensorFlow/Keras model.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_path}'. "
            "This usually indicates that the model was saved with a different "
            "TensorFlow/Keras version than the current environment.\n"
            f"Original error: {e}"
        ) from e


def save_keras_model(model, model_path):
    """
    Save TensorFlow/Keras model.
    """

    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(model_path)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(model, X, y):
    """
    Evaluate trained model.
    """

    loss, accuracy = model.evaluate(
        X,
        y,
        verbose=0
    )

    return loss, accuracy


# ---------------------------------------------------------
# Model Size
# ---------------------------------------------------------

def get_model_size(model_path):
    """
    Return model size in MB.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        return 0.0

    return model_path.stat().st_size / (1024 * 1024)

def load_training_stats():
    """Loads pre-computed training mean and std, or computes them from clean dataset."""
    X_raw, _ = load_training_dataset()
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    return mean, std

# ---------------------------------------------------------
# MLflow Logging
# ---------------------------------------------------------

def log_metrics(metrics):
    """
    Log metrics to MLflow.
    """

    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_params(params):
    """
    Log parameters to MLflow.
    """

    for key, value in params.items():
        mlflow.log_param(key, value)


def log_artifact(file_path):

    file_path = Path(file_path)

    if file_path.exists():
        mlflow.log_artifact(str(file_path))

def print_dataset_info(df):
    """
    Display dataset information.
    """

    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)
    print(f"Shape : {df.shape}")
    print(df.head())
    print()

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