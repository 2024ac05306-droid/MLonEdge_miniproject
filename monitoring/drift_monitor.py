import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONITORING_RESULTS_DIR = PROJECT_ROOT / "monitoring" / "results"
MONITORING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fix sys.path so Python can locate utils.py and config.py in project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_training_dataset, load_training_stats, normalize_features
from config import FEATURE_COLUMNS, STATS_FILE


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) between baseline (expected)
    and live/streamed (actual) feature arrays.
    """
    # Ensure minimum array length
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin edges based on expected baseline range
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Handle duplicate bin edges caused by identical values
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 1:
        return 0.0

    # Calculate bin counts
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert counts to percentages (fractions)
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Replace zero fractions with small constant to prevent division/log by zero
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    # Calculate PSI array and sum
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def plot_feature_psi(psi_dict: dict, output_path: Path):
    """
    Generates and saves a horizontal bar chart of feature-level PSI scores.
    """
    features = list(psi_dict.keys())
    psi_values = list(psi_dict.values())

    # Assign threshold colors
    colors = []
    for val in psi_values:
        if val < 0.10:
            colors.append("#2ca02c")  # Green: Normal / Low Drift
        elif val < 0.25:
            colors.append("#ff7f0e")  # Orange: Moderate Drift
        else:
            colors.append("#d62728")  # Red: High Drift / Action Required

    plt.figure(figsize=(9, 5), dpi=300)
    plt.style.use("ggplot")

    bars = plt.barh(features, psi_values, color=colors, edgecolor="black", alpha=0.85, height=0.55)

    # Threshold indicator lines
    plt.axvline(x=0.10, color="orange", linestyle="--", linewidth=1.5, label="Warning Threshold (0.10)")
    plt.axvline(x=0.25, color="red", linestyle="--", linewidth=1.5, label="Retrain Alert Threshold (0.25)")

    # Display numeric values on top of bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold"
        )

    plt.title("Feature-Level Population Stability Index (PSI) Drift Analysis", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("PSI Score", fontsize=10)
    plt.ylabel("Sensor Features", fontsize=10)
    plt.xlim(0, max(max(psi_values) + 0.06, 0.30))
    plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved PSI Drift Chart to: {output_path}")


def run_drift_monitor():
    """Main drift monitoring execution loop."""
    print("[INFO] Running Population Stability Index (PSI) Drift Monitor...")

    # Load baseline dataset
    X_baseline, _ = load_training_dataset()

    # Simulate live/incoming telemetry stream (with optional synthetic noise to simulate real edge drift)
    np.random.seed(42)
    noise = np.random.normal(loc=0.05, scale=0.15, size=X_baseline.shape)
    X_live = X_baseline + noise

    # Calculate feature-level PSI
    psi_results = {}
    num_features = X_baseline.shape[1]

    print("\n" + "=" * 65)
    print(f"{'Feature Name':<25} | {'PSI Score':<12} | {'Drift Status':<20}")
    print("=" * 65)

    for i in range(num_features):
        feat_name = FEATURE_COLUMNS[i] if i < len(FEATURE_COLUMNS) else f"feature_{i}"
        psi_score = calculate_psi(X_baseline[:, i], X_live[:, i])
        psi_results[feat_name] = round(psi_score, 4)

        if psi_score < 0.10:
            status = "Low / Stable"
        elif psi_score < 0.25:
            status = "Moderate Drift"
        else:
            status = "HIGH DRIFT (ALERT)"

        print(f"{feat_name:<25} | {psi_score:<12.4f} | {status:<20}")

    print("=" * 65 + "\n")

    # Save JSON metrics report
    json_path = MONITORING_RESULTS_DIR / "drift_metrics.json"
    with open(json_path, "w") as f:
        json.dump(psi_results, f, indent=4)
    print(f"[SUCCESS] Saved Drift Metrics JSON to: {json_path}")

    # Generate and save plot
    chart_path = MONITORING_RESULTS_DIR / "psi_chart.png"
    plot_feature_psi(psi_results, chart_path)


if __name__ == "__main__":
    run_drift_monitor()