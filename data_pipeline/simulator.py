import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests 
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import mlflow.models
import mlflow.tracking
import mlflow.utils
import mlflow.entities
import mlflow.exceptions
import mlflow.store
import mlflow.utils.rest_utils


import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "G44_logibridge_miniproject"
)

# --------------------------------------------------
# MQTT Topics
# --------------------------------------------------
TOPICS = {
    "temperature": "coldchain/truck/temperature",
    "vibration": "coldchain/truck/vibration_rms",
    "door": "coldchain/truck/door_event",
}

# --------------------------------------------------
# MQTT Client
# --------------------------------------------------
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, PORT, 60)
client.loop_start()

# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc).isoformat()


def publish(topic: str, payload: dict):
    client.publish(topic, json.dumps(payload), qos=1)
    print(f"[{topic}] {payload}")


# --------------------------------------------------
# Sensor Generators
# --------------------------------------------------
def generate_temperature(base_temp: float, anomaly: str, reading_index: int):
    """1 Hz stream"""
    temp = random.gauss(4.0, 0.3)

    if anomaly in ["temp_drift", "combined"]:
        temp += 0.08 * reading_index

    return round(temp, 3)


def generate_vibration(anomaly: str):
    """0.5 Hz stream"""
    if anomaly in ["vibration", "combined"]:
        vib = random.gauss(1.2, 0.15)
    else:
        vib = random.gauss(0.45, 0.05)

    return round(vib, 3)


def generate_door_event():
    """Discrete event stream"""
    return random.choice(["OPEN", "CLOSE"])


# --------------------------------------------------
# Main Simulation
# --------------------------------------------------
def run_simulation(anomaly_mode: str, duration_sec: int = 300):
    sensor_logs = []
    temp_readings = 0
    vibration_counter = 0

    with mlflow.start_run(run_name=f"sim_{anomaly_mode}"):
        # Log experiment parameters
        mlflow.log_param("anomaly_mode", anomaly_mode)
        mlflow.log_param("duration_sec", duration_sec)
        mlflow.log_param("temperature_frequency_hz", 1.0)
        mlflow.log_param("vibration_frequency_hz", 0.5)
        mlflow.log_param("broker", BROKER)
        mlflow.log_param("port", PORT)

        start = time.time()

        print(f"🚚 Starting cold-chain simulation | anomaly={anomaly_mode}")

        while time.time() - start < duration_sec:
            timestamp = utc_now()

            # --------------------------------------------------
            # Temperature (1 Hz)
            # --------------------------------------------------
            temperature = generate_temperature(4.0, anomaly_mode, temp_readings)

            temp_payload = {
                "timestamp": timestamp,
                "sensor": "temperature",
                "value_c": temperature,
                "setpoint_c": 4.0,
                "anomaly_mode": anomaly_mode,
            }

            publish(TOPICS["temperature"], temp_payload)

            sensor_logs.append(temp_payload)

            # Log metric to MLflow
            mlflow.log_metric("temperature_c", temperature, step=temp_readings)

            temp_readings += 1

            # --------------------------------------------------
            # Vibration (0.5 Hz = every 2 seconds)
            # --------------------------------------------------
            if temp_readings % 2 == 0:
                vibration = generate_vibration(anomaly_mode)

                vib_payload = {
                    "timestamp": timestamp,
                    "sensor": "vibration_rms",
                    "value_g": vibration,
                    "anomaly_mode": anomaly_mode,
                }

                publish(TOPICS["vibration"], vib_payload)

                sensor_logs.append(vib_payload)

                mlflow.log_metric("vibration_rms_g", vibration, step=vibration_counter)
                vibration_counter += 1

            # --------------------------------------------------
            # Door events (discrete, random)
            # ~5% chance per second
            # --------------------------------------------------
            if random.random() < 0.05:
                event = generate_door_event()

                door_payload = {
                    "timestamp": timestamp,
                    "sensor": "door_event",
                    "event": event,
                    "anomaly_mode": anomaly_mode,
                }

                publish(TOPICS["door"], door_payload)
                sensor_logs.append(door_payload)

            # 1 Hz master clock
            time.sleep(1)

        # --------------------------------------------------
        # Save artifacts
        # --------------------------------------------------
        os.makedirs("outputs", exist_ok=True)
        csv_path = f"outputs/sensor_logs_{anomaly_mode}.csv"

        df = pd.DataFrame(sensor_logs)
        df.to_csv(csv_path, index=False)

        mlflow.log_artifact(csv_path, artifact_path="simulation_data")

        # Summary metrics
        mlflow.log_metric("total_records", len(df))
        mlflow.log_metric("temperature_records", len(df[df["sensor"] == "temperature"]))
        mlflow.log_metric("vibration_records", len(df[df["sensor"] == "vibration_rms"]))
        mlflow.log_metric("door_records", len(df[df["sensor"] == "door_event"]))

        print("\\n✅ Simulation completed")
        print(f"📁 Artifact saved: {csv_path}")
        print(f"📊 MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold-chain truck sensor simulator")

    parser.add_argument(
        "--anomaly",
        choices=["none", "temp_drift", "vibration", "combined"],
        default="none",
        help="Anomaly mode",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Simulation duration in seconds",
    )

    args = parser.parse_args()

    try:
        run_simulation(args.anomaly, args.duration)
    except KeyboardInterrupt:
        print("\\n⏹️ Simulation stopped by user")
    finally:
        client.loop_stop()
        client.disconnect()