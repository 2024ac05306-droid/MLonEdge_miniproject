"""
Dataset generation pipeline.

Task D1:
Generate labelled training dataset.

Classes:

0 -> Normal
1 -> Warning
2 -> Critical

Input:
    normal_features.csv
    warning_features.csv
    critical_features.csv

Output:
    training_dataset.csv
"""


from pathlib import Path

import pandas as pd

import mlflow
from dotenv import load_dotenv
import os



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

DATA_DIR = Path(
    "../data"
)


OUTPUT_FILE = (
    DATA_DIR /
    "training_dataset.csv"
)



NORMAL_FILE = (
    DATA_DIR /
    "normal_features.csv"
)


WARNING_FILE = (
    DATA_DIR /
    "warning_features.csv"
)


CRITICAL_FILE = (
    DATA_DIR /
    "critical_features.csv"
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
# Load Class Data
# =====================================================

def load_features(
        file_path,
        label
):

    """
    Read feature CSV and assign label.
    """


    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing file: {file_path}"
        )


    df = pd.read_csv(
        file_path
    )


    # Validate columns

    missing = set(
        FEATURE_COLUMNS
    ) - set(
        df.columns
    )


    if missing:

        raise ValueError(
            f"Missing columns {missing}"
        )



    df = df[
        FEATURE_COLUMNS
    ]


    df["label"] = label


    return df



# =====================================================
# Generate Dataset
# =====================================================

def generate_dataset():


    normal = load_features(
        NORMAL_FILE,
        label=0
    )


    warning = load_features(
        WARNING_FILE,
        label=1
    )


    critical = load_features(
        CRITICAL_FILE,
        label=2
    )



    dataset = pd.concat(
        [
            normal,
            warning,
            critical
        ],
        ignore_index=True
    )


    # Shuffle dataset

    dataset = dataset.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )



    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print(
        "\nDataset generated:"
    )

    print(
        OUTPUT_FILE
    )


    print(
        "\nClass distribution:"
    )

    print(
        dataset["label"]
        .value_counts()
        .sort_index()
    )



    return dataset



# =====================================================
# MLflow Logging
# =====================================================

def log_dataset(
        dataset
):


    with mlflow.start_run(
        run_name="Generate_Labelled_Dataset"
    ):


        mlflow.log_param(
            "classes",
            3
        )


        mlflow.log_param(
            "features",
            6
        )


        mlflow.log_metric(
            "total_samples",
            len(dataset)
        )


        for label, count in (
            dataset["label"]
            .value_counts()
            .items()
        ):

            mlflow.log_metric(
                f"class_{label}_samples",
                int(count)
            )



        mlflow.log_artifact(
            str(OUTPUT_FILE)
        )



# =====================================================
# Main
# =====================================================

if __name__ == "__main__":


    dataset = generate_dataset()


    log_dataset(
        dataset
    )


    print(
        "\nMLflow logging completed."
    )