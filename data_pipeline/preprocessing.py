import json
import os
from collections import deque

import mlflow
import numpy as np
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from scipy.stats import kurtosis

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

load_dotenv()

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "ColdChain_Preprocessing"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

TEMPERATURE_TOPIC = "coldchain/truck/temperature"
VIBRATION_TOPIC = "coldchain/truck/vibration_rms"

# -----------------------------------------------------
# Sliding Window Parameters
# -----------------------------------------------------

TEMP_SAMPLING_RATE = 1          # Hz
VIB_SAMPLING_RATE = 0.5         # Hz

WINDOW_SECONDS = 30
STEP_SECONDS = 10

TEMP_WINDOW = WINDOW_SECONDS * TEMP_SAMPLING_RATE      # 30 samples
TEMP_STEP = STEP_SECONDS * TEMP_SAMPLING_RATE          # 10 samples

VIB_WINDOW = int(WINDOW_SECONDS * VIB_SAMPLING_RATE)   # 15 samples
VIB_STEP = int(STEP_SECONDS * VIB_SAMPLING_RATE)       # 5 samples

TRAINING_DURATION = 600         # 10 minutes

# -----------------------------------------------------
# Buffers
# -----------------------------------------------------

temperature_buffer = deque(maxlen=TEMP_WINDOW)
vibration_buffer = deque(maxlen=VIB_WINDOW)

temperature_history = []
vibration_history = []

feature_vectors = []

# -----------------------------------------------------
# Signal Processing
# -----------------------------------------------------


def moving_average(signal, window=5):
    signal = np.asarray(signal)

    if len(signal) < window:
        return signal

    return np.convolve(
        signal,
        np.ones(window) / window,
        mode="same"
    )


def extract_features(temp_signal, vib_signal):

    temp_signal = moving_average(temp_signal)
    vib_signal = moving_average(vib_signal)

    temp_mean = np.mean(temp_signal)

    temp_std = np.std(temp_signal)

    duration_min = WINDOW_SECONDS / 60

    temp_rate = (
        temp_signal[-1] - temp_signal[0]
    ) / duration_min

    vibration_rms = np.sqrt(np.mean(vib_signal ** 2))

    vibration_peak = np.max(np.abs(vib_signal))

    vibration_kurtosis = kurtosis(
        vib_signal,
        fisher=False,
        bias=False
    )

    return np.array([
        temp_mean,
        temp_std,
        temp_rate,
        vibration_rms,
        vibration_peak,
        vibration_kurtosis
    ])


# -----------------------------------------------------
# Save Training Statistics
# -----------------------------------------------------

def save_training_statistics():

    features = np.asarray(feature_vectors)

    mean = np.mean(features, axis=0)

    std = np.std(features, axis=0)

    std[std == 0] = 1e-8

    np.save(
        "training_stats.npy",
        {
            "mean": mean,
            "std": std
        }
    )

    print("\nTraining statistics saved.")
    print("File : training_stats.npy")

    with mlflow.start_run(run_name="Training_Statistics"):

        mlflow.log_param("window_seconds", WINDOW_SECONDS)
        mlflow.log_param("step_seconds", STEP_SECONDS)
        mlflow.log_param("moving_average", 5)

        mlflow.log_metric(
            "feature_vectors",
            len(features)
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

        mlflow.log_artifact("training_stats.npy")


# -----------------------------------------------------
# MQTT Callback
# -----------------------------------------------------

temp_counter = 0
vib_counter = 0


def on_message(client, userdata, msg):

    global temp_counter
    global vib_counter

    payload = json.loads(msg.payload.decode())

    if payload["sensor"] == "temperature":

        value = payload["value_c"]

        temperature_buffer.append(value)
        temperature_history.append(value)

        temp_counter += 1

    elif payload["sensor"] == "vibration_rms":

        value = payload["value_g"]

        vibration_buffer.append(value)
        vibration_history.append(value)

        vib_counter += 1

    if (
        len(temperature_buffer) == TEMP_WINDOW
        and
        len(vibration_buffer) == VIB_WINDOW
    ):

        feature = extract_features(
            np.array(temperature_buffer),
            np.array(vibration_buffer)
        )

        feature_vectors.append(feature)

        print(
            f"Feature Vector {len(feature_vectors)} :",
            feature.round(3)
        )

        for _ in range(TEMP_STEP):
            if temperature_buffer:
                temperature_buffer.popleft()

        for _ in range(VIB_STEP):
            if vibration_buffer:
                vibration_buffer.popleft()

    if len(temperature_history) >= TRAINING_DURATION:

        save_training_statistics()

        print("Training completed.")

        client.disconnect()


# -----------------------------------------------------
# MQTT
# -----------------------------------------------------

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_message = on_message

client.connect(BROKER, PORT)

client.subscribe(TEMPERATURE_TOPIC)

client.subscribe(VIBRATION_TOPIC)

print("\nWaiting for 10 minutes of NORMAL sensor data...\n")

client.loop_forever()