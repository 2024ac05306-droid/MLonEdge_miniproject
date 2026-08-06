"""
Task F2: Benchmark five operational metrics across M1, M2, and M3 variants.
"""

import time
from pathlib import Path
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, recall_score

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import DATASET_FILE, MODEL_DIR
from utils import load_training_dataset, normalize_features

# Estimated CPU Thermal Design Power (TDP) for laptop host benchmarking
ESTIMATED_LAPTOP_TDP_WATTS = 15.0  # 15W typical ultrabook edge TDP


def evaluate_tflite_variant(model_path: Path, X_val: np.ndarray, y_val: np.ndarray):
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_int8 = (input_details[0]['dtype'] == np.int8)

    # Prepare inputs
    inputs = X_val.astype(np.float32)
    if is_int8:
        scale, zero_point = input_details[0]['quantization']
        inputs = (inputs / scale + zero_point).astype(np.int8)

    # 1. Warm-up (10 runs)
    for i in range(10):
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(inputs[i % len(inputs)], axis=0))
        interpreter.invoke()

    # 2. Measure Latency (200 runs) & CPU Utilization
    latencies = []
    predictions = []
    
    cpu_start = psutil.cpu_percent(interval=None)
    start_time = time.perf_counter()

    for i in range(200):
        sample = np.expand_dims(inputs[i % len(inputs)], axis=0)
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0)  # ms

    total_wall_time = time.perf_counter() - start_time
    cpu_usage = psutil.cpu_percent(interval=None) / 100.0

    # 3. Accuracy & Recall Evaluation over Full Validation Set
    for sample in inputs:
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(sample, axis=0))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])

        if is_int8:
            scale, zero_point = output_details[0]['quantization']
            out = (out.astype(np.float32) - zero_point) * scale

        predictions.append(np.argmax(out, axis=1)[0])

    acc = accuracy_score(y_val, predictions) * 100.0
    recalls = recall_score(y_val, predictions, average=None)
    critical_recall = recalls[2] * 100.0  # Class 2 (Critical)

    # Metric Computations
    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    size_kb = float(model_path.stat().st_size / 1024.0)

    # Energy Calculation: Power = TDP * CPU%; Energy = Power * latency(s) * 1000 mJ
    avg_power_watts = ESTIMATED_LAPTOP_TDP_WATTS * max(cpu_usage, 0.05)
    energy_mj = avg_power_watts * (mean_lat / 1000.0) * 1000.0

    return {
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "size_kb": size_kb,
        "accuracy_pct": acc,
        "critical_recall_pct": critical_recall,
        "energy_mj": energy_mj
    }


def run_benchmarks():
    X_raw, y = load_training_dataset(DATASET_FILE)
    X_norm = normalize_features(X_raw)

    models = {
        "M1 — FP32 Baseline": MODEL_DIR / "model_fp32.tflite",
        "M2 — PTQ INT8": MODEL_DIR / "model_ptq.tflite",
        "M3 — Pruned 35% + PTQ": MODEL_DIR / "model_pruned_ptq.tflite",
    }

    print("\n" + "="*85)
    print(f"{'Model Variant':<25} | {'Mean Lat (ms)':<13} | {'p95 Lat (ms)':<12} | {'Size (KB)':<10} | {'Acc (%)':<8} | {'Energy (mJ)':<10}")
    print("="*85)

    results = {}
    for name, path in models.items():
        if path.exists():
            res = evaluate_tflite_variant(path, X_norm, y)
            results[name] = res
            print(f"{name:<25} | {res['mean_latency_ms']:<13.3f} | {res['p95_latency_ms']:<12.3f} | {res['size_kb']:<10.2f} | {res['accuracy_pct']:<8.2f} | {res['energy_mj']:<10.4f}")

    print("="*85)
    return results


if __name__ == "__main__":
    run_benchmarks()