"""
Cold Chain Preprocessing Pipeline

Responsibilities:
- Receive MQTT sensor streams
- Apply 5-sample moving average filtering
- Apply 30-second sliding window
- Extract six features
- Save extracted features by class
Feature vector:

[
 temperature_mean,
 temperature_std,
 temperature_rate_of_change,
 vibration_rms,
 vibration_peak,
 vibration_kurtosis
]
"""


import json
import os
import sys
from collections import deque
from pathlib import Path


import mlflow
import numpy as np
import pandas as pd

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from scipy.stats import kurtosis



# -----------------------------------------------------
# Project Path
# -----------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)



# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

load_dotenv()


BROKER = os.getenv(
    "MQTT_BROKER",
    "localhost"
)


PORT = int(
    os.getenv(
        "MQTT_PORT",
        1883
    )
)



# Use local MLflow by default

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file:./mlruns"
)


MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "G44_logibridge_miniproject"
)

print(os.getenv("MLFLOW_TRACKING_URI"))
print(os.getenv("MLFLOW_EXPERIMENT_NAME"))



# -----------------------------------------------------
# MQTT Topics
# -----------------------------------------------------

TEMPERATURE_TOPIC = (
    "coldchain/truck/temperature"
)

VIBRATION_TOPIC = (
    "coldchain/truck/vibration_rms"
)


DOOR_EVENT_TOPIC = (
    "coldchain/truck/door_event"
)



# -----------------------------------------------------
# Sliding Window Parameters
# -----------------------------------------------------

TEMP_SAMPLING_RATE = 1

VIB_SAMPLING_RATE = 0.5


WINDOW_SECONDS = 30

STEP_SECONDS = 10



TEMP_WINDOW = (
    WINDOW_SECONDS
    *
    TEMP_SAMPLING_RATE
)


TEMP_STEP = (
    STEP_SECONDS
    *
    TEMP_SAMPLING_RATE
)



VIB_WINDOW = int(
    WINDOW_SECONDS
    *
    VIB_SAMPLING_RATE
)


VIB_STEP = int(
    STEP_SECONDS
    *
    VIB_SAMPLING_RATE
)



# -----------------------------------------------------
# Buffers
# -----------------------------------------------------

temperature_buffer = deque(
    maxlen=TEMP_WINDOW
)


vibration_buffer = deque(
    maxlen=VIB_WINDOW
)


feature_vectors = []



# -----------------------------------------------------
# Signal Processing
# -----------------------------------------------------


def moving_average(
    signal,
    window=5
):

    signal = np.asarray(
        signal,
        dtype=np.float32
    )


    if len(signal) < window:
        return signal


    return np.convolve(
        signal,
        np.ones(window) / window,
        mode="same"
    )



def extract_features(
    temp_signal,
    vib_signal
):

    temp_signal = moving_average(
        temp_signal
    )


    vib_signal = moving_average(
        vib_signal
    )



    temp_mean = np.mean(
        temp_signal
    )


    temp_std = np.std(
        temp_signal
    )


    duration_min = (
        WINDOW_SECONDS / 60
    )


    temp_rate = (
        temp_signal[-1]
        -
        temp_signal[0]
    ) / duration_min



    vibration_rms = np.sqrt(
        np.mean(
            vib_signal ** 2
        )
    )



    vibration_peak = np.max(
        np.abs(vib_signal)
    )



    vibration_kurtosis = kurtosis(
        vib_signal,
        fisher=False,
        bias=False
    )


    if np.isnan(
        vibration_kurtosis
    ):

        vibration_kurtosis = 0.0



    return np.array(
        [
            temp_mean,
            temp_std,
            temp_rate,
            vibration_rms,
            vibration_peak,
            vibration_kurtosis
        ],
        dtype=np.float32
    )



# -----------------------------------------------------
# Save Training Statistics
# -----------------------------------------------------

def save_training_statistics():

    features = np.asarray(
        feature_vectors
    )


    mean = np.mean(
        features,
        axis=0
    )


    std = np.std(
        features,
        axis=0
    )


    std[
        std == 0
    ] = 1e-8



    Path("outputs").mkdir(
        exist_ok=True
    )



    np.save(
        "outputs/training_stats.npy",
        {
            "mean": mean,
            "std": std
        }
    )


    print(
        "Training statistics saved"
    )



# -----------------------------------------------------
# Offline Feature Extraction
# -----------------------------------------------------


def create_features(
    input_file,
    output_file
):


    df = pd.read_csv(
        input_file
    )



    temperature = (
        df[
            df["sensor"]
            ==
            "temperature"
        ]
        ["value_c"]
        .values
    )



    vibration = (
        df[
            df["sensor"]
            ==
            "vibration_rms"
        ]
        ["value_g"]
        .values
    )



    feature_rows = []



    for start in range(
        0,
        len(temperature)-TEMP_WINDOW+1,
        TEMP_STEP
    ):


        temp_window = temperature[
            start:
            start+TEMP_WINDOW
        ]



        vib_start = (
            start // 2
        )



        vib_window = vibration[
            vib_start:
            vib_start+VIB_WINDOW
        ]



        if len(vib_window) < VIB_WINDOW:
            continue



        feature_rows.append(
            extract_features(
                temp_window,
                vib_window
            )
        )



    columns = [

        "temp_mean",

        "temp_std",

        "temp_rate",

        "vibration_rms",

        "vibration_peak",

        "vibration_kurtosis"

    ]



    Path(
        "data"
    ).mkdir(
        exist_ok=True
    )



    feature_df = pd.DataFrame(
        feature_rows,
        columns=columns
    )


    feature_df.to_csv(
        output_file,
        index=False
    )



    print(
        f"Saved {output_file}"
    )


    print(
        "Shape:",
        feature_df.shape
    )



# -----------------------------------------------------
# Main
# -----------------------------------------------------

if __name__ == "__main__":


    create_features(
        "outputs/sensor_logs_none.csv",
        "data/normal_features.csv"
    )


    create_features(
        "outputs/sensor_logs_temp_drift.csv",
        "data/warning_features.csv"
    )


    create_features(
        "outputs/sensor_logs_combined.csv",
        "data/critical_features.csv"
    )


    print(
        "\nFeature extraction completed"
    )