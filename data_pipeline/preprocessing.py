"""
Cold Chain Preprocessing Pipeline

Responsibilities:
- Receive MQTT sensor streams
- Apply 5-sample moving average filtering
- Apply 30-second sliding window
- Extract six features
- Compute and save training stats strictly from 10 minutes of clean Normal-class output

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import save utility if available
try:
    from utils import save_normal_training_stats
except ImportError:
    save_normal_training_stats = None


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

load_dotenv()

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))

# Use local MLflow by default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "G44_logibridge_miniproject"
)

print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")


# -----------------------------------------------------
# MQTT Topics
# -----------------------------------------------------

TEMPERATURE_TOPIC = "coldchain/truck/temperature"
VIBRATION_TOPIC = "coldchain/truck/vibration_rms"
DOOR_EVENT_TOPIC = "coldchain/truck/door_event"


# -----------------------------------------------------
# Sliding Window Parameters
# -----------------------------------------------------

TEMP_SAMPLING_RATE = 1
VIB_SAMPLING_RATE = 0.5

WINDOW_SECONDS = 30
STEP_SECONDS = 10

TEMP_WINDOW = WINDOW_SECONDS * TEMP_SAMPLING_RATE
TEMP_STEP = STEP_SECONDS * TEMP_SAMPLING_RATE

VIB_WINDOW = int(WINDOW_SECONDS * VIB_SAMPLING_RATE)
VIB_STEP = int(STEP_SECONDS * VIB_SAMPLING_RATE)


# -----------------------------------------------------
# Buffers
# -----------------------------------------------------

temperature_buffer = deque(maxlen=TEMP_WINDOW)
vibration_buffer = deque(maxlen=VIB_WINDOW)
feature_vectors = []


# -----------------------------------------------------
# Signal Processing
# -----------------------------------------------------


def moving_average(signal, window=5):
    """Applies a 5-sample moving average filter."""
    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) < window:
        return signal

    return np.convolve(signal, np.ones(window) / window, mode="same")


def extract_features(temp_signal, vib_signal):
    """Extracts 6 joint features from filtered sliding windows."""
    temp_signal = moving_average(temp_signal)
    vib_signal = moving_average(vib_signal)

    temp_mean = np.mean(temp_signal)
    temp_std = np.std(temp_signal)

    duration_min = WINDOW_SECONDS / 60
    temp_rate = (temp_signal[-1] - temp_signal[0]) / duration_min

    vibration_rms = np.sqrt(np.mean(vib_signal**2))
    vibration_peak = np.max(np.abs(vib_signal))

    vibration_kurtosis = kurtosis(vib_signal, fisher=False, bias=False)

    if np.isnan(vibration_kurtosis):
        vibration_kurtosis = 0.0

    return np.array(
        [
            temp_mean,
            temp_std,
            temp_rate,
            vibration_rms,
            vibration_peak,
            vibration_kurtosis,
        ],
        dtype=np.float32,
    )


# -----------------------------------------------------
# Save Training Statistics (Task C2 Requirement)
# -----------------------------------------------------


def save_training_statistics_from_normal(normal_csv_path="data/normal_features.csv"):
    """
    Computes mean and std strictly from 10 minutes of clean Normal-class output
    and saves to training_stats.npy (Task C2 requirement).
    """
    normal_path = Path(normal_csv_path)
    if not normal_path.exists():
        print(f"[WARNING] Cannot compute stats: {normal_csv_path} not found.")
        return

    df = pd.read_csv(normal_path)
    feature_cols = [
        "temp_mean",
        "temp_std",
        "temp_rate",
        "vibration_rms",
        "vibration_peak",
        "vibration_kurtosis",
    ]

    features = df[feature_cols].values

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    # Handle zero-variance safeguards
    std[std == 0] = 1e-8

    # Save to multiple standard directories for safety
    target_dirs = [Path("data"), Path("outputs"), PROJECT_ROOT]
    
    stats_dict = {"mean": mean, "std": std}

    for target_dir in target_dirs:
        target_dir.mkdir(exist_ok=True, parents=True)
        save_file = target_dir / "training_stats.npy"
        np.save(save_file, stats_dict)
        print(f"[INFO] Saved clean Normal training statistics to: {save_file}")


# -----------------------------------------------------
# Offline Feature Extraction
# -----------------------------------------------------


def create_features(input_file, output_file):
    if not Path(input_file).exists():
        print(f"[WARNING] Input raw file missing: {input_file}")
        return

    df = pd.read_csv(input_file)

    temperature = df[df["sensor"] == "temperature"]["value_c"].values
    vibration = df[df["sensor"] == "vibration_rms"]["value_g"].values

    feature_rows = []

    for start in range(0, len(temperature) - TEMP_WINDOW + 1, TEMP_STEP):
        temp_window = temperature[start : start + TEMP_WINDOW]
        vib_start = start // 2
        vib_window = vibration[vib_start : vib_start + VIB_WINDOW]

        if len(vib_window) < VIB_WINDOW:
            continue

        feature_rows.append(extract_features(temp_window, vib_window))

    columns = [
        "temp_mean",
        "temp_std",
        "temp_rate",
        "vibration_rms",
        "vibration_peak",
        "vibration_kurtosis",
    ]

    Path("data").mkdir(exist_ok=True)

    feature_df = pd.DataFrame(feature_rows, columns=columns)
    feature_df.to_csv(output_file, index=False)

    print(f"Saved {output_file} | Shape: {feature_df.shape}")


# -----------------------------------------------------
# Main Execution
# -----------------------------------------------------

if __name__ == "__main__":

    print("\nExtracting features across all operational modes...")

    # 1. Class 0: Normal
    create_features(
        "outputs/sensor_logs_none.csv", "data/normal_features.csv"
    )

    # 2. Class 1: Warning
    create_features(
        "outputs/sensor_logs_temp_drift.csv", "data/warning_features.csv"
    )

    # 3. Class 2: Critical
    create_features(
        "outputs/sensor_logs_combined.csv", "data/critical_features.csv"
    )

    # 4. Compute & Save Normal-Class Training Stats (Task C2)
    print("\nComputing Normal-class training stats...")
    save_training_statistics_from_normal("data/normal_features.csv")

    print("\nFeature extraction and stats generation completed successfully.")