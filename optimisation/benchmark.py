import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score

# Path Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Output folder directed to Optimisation/results
RESULTS_DIR = PROJECT_ROOT / "optimisation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fix sys.path so Python can locate utils.py in project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_training_dataset, load_training_stats, normalize_features

# TFLite Loader
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def evaluate_tflite_model(model_path: Path, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluates latency, file size, accuracy, Class 2 recall, and estimated energy."""
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    latencies = []
    y_pred = []

    # Warmup
    dummy = np.expand_dims(X_test[0], axis=0).astype(np.float32)
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        dummy = (dummy / scale + zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()

    # Benchmark loop
    for sample in X_test:
        input_data = np.expand_dims(sample, axis=0).astype(np.float32)
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)

        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # ms

        if output_details[0]['dtype'] == np.int8:
            scale, zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        pred = np.argmax(output, axis=1)[0]
        y_pred.append(pred)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    size_kb = float(os.path.getsize(model_path) / 1024.0)
    acc = float(accuracy_score(y_test, y_pred) * 100.0)
    
    # Class 2 Recall (Critical Fault Class)
    c2_recall = float(recall_score(y_test, y_pred, labels=[2], average=None)[0] * 100.0)
    
    # Energy estimate (approx. 0.75 mW baseline power scaling with latency)
    energy_mJ = float(mean_lat * 0.75)

    return {
        "mean_latency_ms": round(mean_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "size_kb": round(size_kb, 2),
        "accuracy_pct": round(acc, 2),
        "class2_recall_pct": round(c2_recall, 2),
        "energy_mj": round(energy_mJ, 4)
    }


def print_evaluation_table(df: pd.DataFrame):
    """Prints a formatted evaluation table to the terminal."""
    header = f"{'Model Variant':<25} | {'Mean Lat (ms)':<13} | {'p95 Lat (ms)':<12} | {'Size (KB)':<10} | {'Acc (%)':<8} | {'Class 2 Rec (%)':<15} | {'Energy (mJ)':<11}"
    divider = "=" * len(header)
    
    print("\n" + divider)
    print("METRICS EVALUATION TABLE")
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
    """Generates a clear, non-overlapping Latency vs Accuracy Pareto trade-off plot."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Custom styling
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

    # Define distinct markers and colors for each model variant
    styles = {
        0: {"color": "#1f77b4", "marker": "o", "xytext": (-35, 15)},   # Top-left offset
        1: {"color": "#ff7f0e", "marker": "s", "xytext": (0, -28)},   # Bottom-center offset
        2: {"color": "#2ca02c", "marker": "^", "xytext": (35, 15)},   # Top-right offset
    }

    # Plot scatter points and non-overlapping annotations
    for idx, (_, row) in enumerate(df.iterrows()):
        style = styles.get(idx, {"color": "#333333", "marker": "o", "xytext": (0, 15)})
        
        # Plot model point
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

        # Annotate with custom offsets to eliminate overlap
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

    # Sort data for Pareto step-line
    sorted_df = df.sort_values(by="mean_latency_ms")
    
    # Draw Pareto Frontier Line (Step-wise)
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

    # Formatting axes and margins
    ax.set_title("Edge Optimization Trade-off: Latency vs. Accuracy", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Mean Inference Latency (ms)", fontsize=11, labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11, labelpad=8)

    # Expand margins so annotations do not get cropped at edges
    ax.margins(x=0.2, y=0.2)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved Enhanced Pareto Chart to: {output_path}")

if __name__ == "__main__":
    X_raw, y = load_training_dataset()
    mean, std = load_training_stats()
    X_norm = normalize_features(X_raw, mean, std)

    models = {
        "M1 — FP32 Baseline": MODELS_DIR / "model_fp32.tflite",
        "M2 — PTQ INT8": MODELS_DIR / "model_ptq.tflite",
        "M3 — Pruned 35% + PTQ": MODELS_DIR / "model_pruned_ptq.tflite"
    }

    results = []
    for variant, path in models.items():
        print(f"[INFO] Benchmarking {variant}...")
        metrics = evaluate_tflite_model(path, X_norm, y)
        metrics["variant"] = variant
        results.append(metrics)

    df = pd.DataFrame(results)
    
    cols = ["variant", "mean_latency_ms", "p95_latency_ms", "size_kb", "accuracy_pct", "class2_recall_pct", "energy_mj"]
    df = df[cols]

    # Print table to console
    print_evaluation_table(df)

    # Save outputs to optimisation/results/
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved Benchmark Results CSV to: {csv_path}")

    chart_path = RESULTS_DIR / "pareto_chart.png"
    generate_pareto_chart(df, chart_path)