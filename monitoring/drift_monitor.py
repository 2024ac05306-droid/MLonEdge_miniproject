import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Path Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONITORING_DIR = PROJECT_ROOT / "monitoring"
RESULTS_DIR = MONITORING_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REF_DIST_FILE = MONITORING_DIR / "reference_dist.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_FILE
from utils import get_features_and_labels, load_training_dataset, FEATURE_COLUMNS



def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between baseline and live arrays."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.unique(np.percentile(expected, percentiles))
    
    if len(bin_edges) <= 1:
        return 0.0

    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Prevent division/log by zero
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def plot_feature_psi(psi_dict: dict, output_path: Path):
    """Plots feature-level PSI scores against alert thresholds."""
    features = list(psi_dict.keys())
    psi_values = list(psi_dict.values())

    colors = []
    for v in psi_values:
        if v < 0.10:
            colors.append("#2ca02c")  # Stable
        elif v < 0.25:
            colors.append("#ff7f0e")  # Warning
        else:
            colors.append("#d62728")  # Alert

    plt.figure(figsize=(9, 5), dpi=300)
    plt.style.use("ggplot")

    bars = plt.barh(features, psi_values, color=colors, edgecolor="black", height=0.55)

    plt.axvline(x=0.10, color="orange", linestyle="--", label="Warning (0.10)")
    plt.axvline(x=0.25, color="red", linestyle="--", label="Retrain Alert (0.25)")

    for bar in bars:
        w = bar.get_width()
        plt.text(
            w + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.title("Feature-Level PSI Drift Analysis", fontsize=12, fontweight="bold")
    plt.xlabel("PSI Score")
    plt.ylabel("Sensor Features")
    plt.xlim(0, max(max(psi_values) + 0.06, 0.30))
    plt.legend(loc="lower right", frameon=True, facecolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_drift_monitor():
    """Main drift execution function."""
    print("[INFO] Running Population Stability Index (PSI) Drift Monitor...")

    # Load baseline feature dataset
    df_base = load_training_dataset(DATASET_FILE)
    X_baseline, _ = get_features_and_labels(df_base)

    # Simulate incoming live stream with mild sensor drift
    np.random.seed(42)
    noise = np.random.normal(loc=0.05, scale=0.15, size=X_baseline.shape)
    X_live = X_baseline + noise

    psi_results = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        psi_score = calculate_psi(X_baseline[:, i], X_live[:, i])
        psi_results[col] = round(psi_score, 4)

    # 1. Save JSON metrics inside monitoring/results/
    json_path = RESULTS_DIR / "drift_metrics.json"
    with open(json_path, "w") as f:
        json.dump(psi_results, f, indent=4)
    print(f"[SUCCESS] Saved Drift Metrics JSON to: {json_path}")

    # 2. Save chart inside monitoring/results/
    chart_path = RESULTS_DIR / "psi_chart.png"
    plot_feature_psi(psi_results, chart_path)
    print(f"[SUCCESS] Saved PSI Chart to: {chart_path}")


if __name__ == "__main__":
    run_drift_monitor()