"""
Normalization module for Cold Chain Monitoring.

Responsibilities:
    - Calculate training statistics from Normal class only
    - Save training_stats.npy
    - Load statistics during inference

IMPORTANT:
    Statistics are NEVER calculated from live data.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
from dotenv import load_dotenv


# =====================================================
# Environment
# =====================================================

load_dotenv()


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "./mlruns"
)


MLFLOW_EXPERIMENT_NAME = (
    "G44_logibridge_miniproject"
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

DATA_DIR = Path("..\data")


NORMAL_FEATURE_FILE = (
    DATA_DIR /
    "normal_features.csv"
)


STATS_FILE = (
    DATA_DIR /
    "training_stats.npy"
)



# =====================================================
# Feature Columns
# =====================================================

FEATURE_COLUMNS = [

    "temp_mean",

    "temp_std",

    "temp_rate",

    "vibration_rms",

    "vibration_peak",

    "vibration_kurtosis"

]



# =====================================================
# Create Training Statistics
# =====================================================

def create_training_statistics():

    """
    Calculate mean and std from Normal class only.
    """


    if not NORMAL_FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {NORMAL_FEATURE_FILE}"
        )


    df = pd.read_csv(
        NORMAL_FEATURE_FILE
    )


    X = df[
        FEATURE_COLUMNS
    ].values



    # Feature statistics

    mean = np.mean(
        X,
        axis=0
    )


    std = np.std(
        X,
        axis=0
    )


    # Avoid divide by zero

    std[
        std == 0
    ] = 1e-8



    stats = {

        "mean": mean,

        "std": std
    }



    np.save(
        STATS_FILE,
        stats
    )


    print(
        "\nTraining statistics saved:"
    )

    print(
        STATS_FILE
    )


    print(
        "\nMean:"
    )

    print(mean)


    print(
        "\nStd:"
    )

    print(std)



    # ------------------------------
    # MLflow Tracking
    # ------------------------------

    with mlflow.start_run(
        run_name="Create_Training_Stats"
    ):


        mlflow.log_param(
            "source",
            "Normal class"
        )


        mlflow.log_param(
            "features",
            6
        )


        mlflow.log_metric(
            "training_samples",
            len(X)
        )


        for i, value in enumerate(mean):

            mlflow.log_metric(
                f"mean_feature_{i}",
                float(value)
            )


        for i, value in enumerate(std):

            mlflow.log_metric(
                f"std_feature_{i}",
                float(value)
            )



        mlflow.log_artifact(
            str(STATS_FILE)
        )



# =====================================================
# Load Statistics
# =====================================================

def load_training_statistics():

    """
    Load previously generated statistics.

    Used during inference.

    NEVER recalculates statistics.
    """


    if not STATS_FILE.exists():

        raise FileNotFoundError(
            "training_stats.npy not found"
        )


    stats = np.load(
        STATS_FILE,
        allow_pickle=True
    ).item()


    return (
        stats["mean"],
        stats["std"]
    )



# =====================================================
# Normalize Features
# =====================================================

def normalize_features(
    features
):

    """
    Apply saved normalization.

    Formula:

        x_norm = (x - mean) / std

    """


    mean, std = (
        load_training_statistics()
    )


    normalized = (
        features - mean
    ) / std


    return normalized



# =====================================================
# Main
# =====================================================

if __name__ == "__main__":


    create_training_statistics()