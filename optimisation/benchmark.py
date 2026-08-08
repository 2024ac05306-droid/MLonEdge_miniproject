import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import mlflow

# Path Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "optimisation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from config import DATASET_FILE, MODEL_DIR
from utils import load_training_dataset, load_training_stats, normalize_features, get_features_and_labels

# TFLite Loader
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# System TDP Estimate for Laptop (e.g., 15W / 15000 mW base TDP)
LAPTOP_TDP_MW = 15000.0  


def measure_power_consumption_mw() -> float:
    """Estimates active system power in mW using psutil CPU utilization and laptop TDP."""
    cpu_utilization = psutil.cpu_percent(interval=0.05) / 100.0
    # Base idle power fraction (10% TDP) + active utilization scaling
    estimated_power_mw = LAPTOP_TDP_MW * (0.10 + 0.90 * cpu_utilization)
    return estimated_power_mw


def evaluate_tflite_model(model_path: Path, X_val: np.ndarray, y_val: np.ndarray):
    """Evaluates the 5 required Task F2 metrics on a held-out validation set."""
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # --- Metric 1: Model File Size (KB) ---
    size_kb = float(os.path.getsize(model_path) / 1024.0)

    # --- Task Requirement: 10 Warm-up Runs ---
    warmup_sample = np.expand_dims(X_val[0], axis=0).astype(np.float32)
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        warmup_sample = np.clip(np.round(warmup_sample / scale + zero_point), -128, 127).astype(np.int8)

    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], warmup_sample)
        interpreter.invoke()

    # --- Task Requirement: 200 Runs Latency Benchmarking ---
    latencies = []
    # Sample 200 instances for latency timing
    num_runs = 200
    bench_indices = np.random.choice(len(X_val), size=num_runs, replace=True)

    for idx in bench_indices:
        sample = np.expand_dims(X_val[idx], axis=0).astype(np.float32)
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            sample = np.clip(np.round(sample / scale + zero_point), -128, 127).astype(np.int8)

        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0)  # Convert to ms

    latencies = np.array(latencies)
    # --- Metric 2 & 3: Mean Latency & p95 Latency ---
    mean_lat_ms = float(np.mean(latencies))
    p95_lat_ms = float(np.percentile(latencies, 95))

    # --- Metric 4: Energy per Inference (mJ) using E = P * t ---
    power_mw = measure_power_consumption_mw()
    # E (mJ) = Power (mW) * time (seconds) = Power (mW) * (latency_ms / 1000.0)
    energy_mj = float(power_mw * (mean_lat_ms / 1000.0))

    # --- Metric 5: Accuracy & Class 2 Recall on Held-Out Validation Set ---
    y_pred = []
    for sample_raw in X_val:
        sample = np.expand_dims(sample_raw, axis=0).astype(np.float32)
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            sample = np.clip(np.round(sample / scale + zero_point), -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]['dtype'] == np.int8:
            scale, zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        pred = np.argmax(output, axis=1)[0]
        y_pred.append(pred)

    acc_pct = float(accuracy_score(y_val, y_pred) * 100.0)
    c2_recall_pct = float(recall_score(y_val, y_pred, labels=[2], average=None, zero_division=0)[0] * 100.0)

    return {
        "mean_latency_ms": round(mean_lat_ms, 4),
        "p95_latency_ms": round(p95_lat_ms, 4),
        "size_kb": round(size_kb, 2),
        "accuracy_pct": round(acc_pct, 2),
        "class2_recall_pct": round(c2_recall_pct, 2),
        "energy_mj": round(energy_mj, 4)
    }


def print_evaluation_table(df: pd.DataFrame):
    """Prints a formatted evaluation table to terminal."""
    header = f"{'Model Variant':<25} | {'Mean Lat (ms)':<13} | {'p95 Lat (ms)':<12} | {'Size (KB)':<10} | {'Acc (%)':<8} | {'Class 2 Rec (%)':<15} | {'Energy (mJ)':<11}"
    divider = "=" * len(header)
    
    print("\n" + divider)
    print("TASK F2 — METRICS EVALUATION TABLE")
    print(divider)
    print(header)
    print("-" * len(header))
    
    for _, row in df.iterrows():
        print(
            f"{row['variant']:<25} | "
            f"{row['mean_latency_ms']:<13.4f} | "
            f"{row['p95_latency_ms']:<12.4f} | "
            f"{row['size_kb']:<10.2f} | "
            f"{row['accuracy_pct']:<8.2f} | "
            f"{row['class2_recall_pct']:<15.2f} | "
            f"{row['energy_mj']:<11.4f}"
        )
    print(divider + "\n")


def generate_pareto_chart(df: pd.DataFrame, output_path: Path):
    """Generates Latency vs Accuracy Pareto trade-off plot."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

    styles = {
        0: {"color": "#1f77b4", "marker": "o", "xytext": (-35, 15)},
        1: {"color": "#ff7f0e", "marker": "s", "xytext": (0, -28)},
        2: {"color": "#2ca02c", "marker": "^", "xytext": (35, 15)},
    }

    for idx, (_, row) in enumerate(df.iterrows()):
        style = styles.get(idx, {"color": "#333333", "marker": "o", "xytext": (0, 15)})
        ax.scatter(
            row["mean_latency_ms"],
            row["accuracy_pct"],
            s=160,
            color=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidth=1.2,
            zorder=5,
            label=row["variant"]
        )

        label_text = f"{row['variant']}\n({row['size_kb']:.1f} KB | {row['accuracy_pct']:.1f}%)"
        ax.annotate(
            label_text,
            (row["mean_latency_ms"], row["accuracy_pct"]),
            textcoords="offset points",
            xytext=style["xytext"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=style["color"], alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#666666", lw=1)
        )

    sorted_df = df.sort_values(by="mean_latency_ms")
    ax.plot(
        sorted_df["mean_latency_ms"],
        sorted_df["accuracy_pct"],
        linestyle="--",
        color="#555555",
        linewidth=1.5,
        alpha=0.7,
        zorder=3,
        label="Frontier Trend"
    )

    ax.set_title("Edge Optimization Trade-off: Latency vs. Accuracy", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Mean Inference Latency (ms)", fontsize=11, labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11, labelpad=8)
    ax.margins(x=0.2, y=0.2)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Add this line to explicitly set the MLflow experiment name
    mlflow.set_experiment("Task_F2_Edge_Benchmarking")
    # Load dataset and stats
    df = load_training_dataset(DATASET_FILE)
    X_raw, y = get_features_and_labels(df)
    mean, std = load_training_stats()
    X_norm = normalize_features(X_raw, mean, std)

    # Held-out 20% validation split
    _, X_val, _, y_val = train_test_split(
        X_norm, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "M1 — FP32 Baseline": MODELS_DIR / "model_fp32.tflite",
        "M2 — PTQ INT8": MODELS_DIR / "model_ptq.tflite",
        "M3 — Pruned 35% + PTQ": MODELS_DIR / "model_pruned_ptq.tflite"
    }

    results = []
# Parent MLflow Run for the full benchmark suite
    with mlflow.start_run(run_name="Edge_Benchmark_Suite"):
        for variant_name, path in models.items():
            print(f"[INFO] Benchmarking {variant_name}...")
            
            # Nested run per model variant
            with mlflow.start_run(run_name=variant_name, nested=True):
                metrics = evaluate_tflite_model(path, X_val, y_val)
                metrics["variant"] = variant_name
                results.append(metrics)

                # Log individual metrics to MLflow
                mlflow.log_metrics({
                    "mean_latency_ms": metrics["mean_latency_ms"],
                    "p95_latency_ms": metrics["p95_latency_ms"],
                    "size_kb": metrics["size_kb"],
                    "accuracy_pct": metrics["accuracy_pct"],
                    "class2_recall_pct": metrics["class2_recall_pct"],
                    "energy_mj": metrics["energy_mj"]
                })

        # Process results DataFrame
        df = pd.DataFrame(results)
        cols = ["variant", "mean_latency_ms", "p95_latency_ms", "size_kb", "accuracy_pct", "class2_recall_pct", "energy_mj"]
        df = df[cols]

        print_evaluation_table(df)

        # Save CSV and log as MLflow artifact
        csv_path = RESULTS_DIR / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path), artifact_path="benchmark_outputs")
        print(f"[SUCCESS] Saved Benchmark Results CSV to: {csv_path}")

        # Save Pareto plot and log as MLflow artifact
        chart_path = RESULTS_DIR / "pareto_chart.png"
        generate_pareto_chart(df, chart_path)
        mlflow.log_artifact(str(chart_path), artifact_path="benchmark_outputs")
        print(f"[SUCCESS] Saved Enhanced Pareto Chart to: {chart_path}")
