"""
live_drift_monitor.py
Task E1 — Live PSI Drift Monitoring (previously missing)

Neither drift_monitor.py nor PSI_monitor.py in this project has a live
MQTT loop — both are offline/batch analysis scripts. This script is the
missing live component: it subscribes to real-time inference results,
maintains a rolling window of the last 100 confidence scores, computes
PSI against reference_dist.json every 60 seconds, and alerts per spec.

Reuses the SAME 4-fixed-bin PSI methodology already correctly implemented
in generate_reference_dist.py / evaluate_psi_phases.py — NOT the
10-percentile-bin approach in the older drift_monitor.py / PSI_monitor.py.

Usage:
  python live_drift_monitor.py --truck-id truck01 --ref reference_dist.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]
ROLLING_WINDOW_SIZE = 100
CHECK_INTERVAL_SECONDS = 60
PSI_ALERT_THRESHOLD = 0.25
PSI_STABLE_THRESHOLD = 0.10

parser = argparse.ArgumentParser(description="LogiBridge live PSI drift monitor")
parser.add_argument("--truck-id", default="truck01")
parser.add_argument("--broker", default="127.0.0.1")
parser.add_argument("--port", type=int, default=1883)
parser.add_argument("--ref", default=str(Path(__file__).resolve().parent / "reference_dist.json"))
args = parser.parse_args()

TOPIC_INFERENCE = f"logibridge/trucks/{args.truck_id}/inference"


def calculate_psi(expected_props, actual_props, epsilon=1e-4) -> float:
    """Same formula as evaluate_psi_phases.py / generate_reference_dist.py's
    methodology — kept identical for consistency with the offline evaluation."""
    expected = np.clip(expected_props, epsilon, 1.0)
    actual = np.clip(actual_props, epsilon, 1.0)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def bin_proportions(confidences: list[float]) -> np.ndarray:
    counts, _ = np.histogram(confidences, bins=BIN_EDGES)
    return counts / max(len(confidences), 1)


def interpret(score: float) -> str:
    if score < PSI_STABLE_THRESHOLD:
        return "STABLE"
    elif score < PSI_ALERT_THRESHOLD:
        return "WARNING"
    else:
        return "SIGNIFICANT DRIFT"


def main():
    with open(args.ref) as f:
        reference_data = json.load(f)
    reference_proportions = np.array(reference_data["proportions"])
    print(f"Loaded reference distribution from {args.ref} "
          f"({reference_data['sample_size']} samples)")
    print(f"Reference proportions: {reference_proportions}")

    rolling_confidences: deque = deque(maxlen=ROLLING_WINDOW_SIZE)
    last_message_time = [None]
    lock = threading.Lock()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        print(f"Connected (rc={reason_code}). Subscribing to {TOPIC_INFERENCE}")
        client.subscribe(TOPIC_INFERENCE, qos=1)

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        with lock:
            rolling_confidences.append(payload["confidence"])
            last_message_time[0] = time.time()

    def monitor_loop():
        """Runs on a fixed 60s interval regardless of message arrival."""
        while True:
            time.sleep(CHECK_INTERVAL_SECONDS)
            now = time.time()

            with lock:
                last_msg = last_message_time[0]
                snapshot = list(rolling_confidences)

            if last_msg is None:
                print(f"[{time.strftime('%H:%M:%S')}] No inferences received yet — "
                      f"waiting for {TOPIC_INFERENCE}")
                continue

            if len(snapshot) < 10:
                print(f"[{time.strftime('%H:%M:%S')}] Only {len(snapshot)} inferences "
                      f"collected so far — waiting for enough data to compute PSI")
                continue

            current_proportions = bin_proportions(snapshot)
            score = calculate_psi(reference_proportions, current_proportions)

            print(f"[{time.strftime('%H:%M:%S')}] PSI={score:.3f} "
                  f"(n={len(snapshot)}) — {interpret(score)}")

            if score > PSI_ALERT_THRESHOLD:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={score:.3f}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"live-drift-{args.truck_id}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, 60)

    print(f"Live drift monitor started | truck={args.truck_id} | "
          f"checking every {CHECK_INTERVAL_SECONDS}s on last {ROLLING_WINDOW_SIZE} inferences")
    print("Press Ctrl+C to stop\n")

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nDrift monitor stopped.")
        client.disconnect()


if __name__ == "__main__":
    main()
