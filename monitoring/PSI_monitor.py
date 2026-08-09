"""
drift_monitor.py

Feature-level Population Stability Index (PSI) drift monitoring
for the MLOnEdge vehicle sensor model.

Features monitored:
    - temp_mean
    - temp_std
    - temp_rate
    - vibration_rms
    - vibration_peak
    - vibration_kurtosis

PSI thresholds:
    < 0.10       -> Stable
    0.10 - 0.25  -> Warning
    >= 0.25      -> Retrain Alert

The script uses the training dataset as the baseline and generates
a synthetic live dataset with realistic feature-specific drift.

This is intended for demonstrating drift monitoring in the
ML-on-Edge / MLOps project.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. PATH SETUP
# ============================================================

# This file is expected to be:
#
# MLOnEdge/
# ├── config.py
# ├── utils.py
# ├── data/
# ├── models/
# └── monitoring/
#     └── drift_monitor.py
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MONITORING_DIR = PROJECT_ROOT / "monitoring"
RESULTS_DIR = MONITORING_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Make project root available for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. PROJECT IMPORTS
# ============================================================

from config import DATASET_FILE, FEATURE_COLUMNS
from utils import get_features_and_labels, load_training_dataset


# ============================================================
# 3. PSI CONFIGURATION
# ============================================================

PSI_WARNING_THRESHOLD = 0.10
PSI_RETRAIN_THRESHOLD = 0.25

NUM_BINS = 10

# Reproducibility
RANDOM_SEED = 42


# ============================================================
# 4. DRIFT SCENARIO
# ============================================================

# Available:
#
# "mild"
# "moderate"
# "severe"
#
# Recommended for your assignment:
#     moderate
#
DRIFT_MODE = "moderate"


# ============================================================
# 5. FEATURE-SPECIFIC DRIFT CONFIGURATION
# ============================================================
#
# Values represent approximately how many baseline standard
# deviations the feature mean is shifted.
#
# This is much better than adding the same absolute noise
# to every sensor feature.
#
# Example:
#
# vibration_rms = 0.8
#
# means:
#
# new_mean ≈ old_mean + 0.8 * baseline_std
#
# Different sensor features therefore receive realistic
# relative drift.
#

DRIFT_PROFILES = {

    "mild": {
        "temp_mean": 0.10,
        "temp_std": 0.15,
        "temp_rate": 0.05,
        "vibration_rms": 0.20,
        "vibration_peak": 0.20,
        "vibration_kurtosis": 0.05,
    },

    "moderate": {
        "temp_mean": 0.30,
        "temp_std": 0.60,
        "temp_rate": 0.05,
        "vibration_rms": 0.90,
        "vibration_peak": 0.80,
        "vibration_kurtosis": 0.05,
    },

    "severe": {
        "temp_mean": 0.60,
        "temp_std": 1.00,
        "temp_rate": 0.10,
        "vibration_rms": 1.50,
        "vibration_peak": 1.30,
        "vibration_kurtosis": 0.10,
    },
}


# Additional random noise expressed as a fraction of the
# baseline standard deviation.

NOISE_SCALE = {
    "mild": 0.05,
    "moderate": 0.10,
    "severe": 0.15,
}


# ============================================================
# 6. LOAD BASELINE DATA
# ============================================================

def load_baseline_data():
    """
    Load the training dataset and extract sensor features.
    """

    print("[INFO] Loading baseline training dataset...")

    # Your utils.py defines load_training_dataset()
    # without a parameter.
    df_base = load_training_dataset()

    if df_base.empty:
        raise ValueError("Training dataset is empty.")

    X_baseline, _ = get_features_and_labels(df_base)

    X_baseline = np.asarray(X_baseline, dtype=np.float64)

    if X_baseline.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix, got shape {X_baseline.shape}"
        )

    if X_baseline.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            "Feature count mismatch: "
            f"dataset has {X_baseline.shape[1]} columns but "
            f"FEATURE_COLUMNS contains {len(FEATURE_COLUMNS)} features."
        )

    print(f"[INFO] Baseline samples : {X_baseline.shape[0]}")
    print(f"[INFO] Features         : {X_baseline.shape[1]}")

    return X_baseline


# ============================================================
# 7. GENERATE SYNTHETIC LIVE DATA
# ============================================================

def generate_live_data(
    X_baseline: np.ndarray,
    mode: str = DRIFT_MODE,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Generate synthetic live sensor data with feature-specific drift.

    Drift is based on each feature's baseline standard deviation.

    This avoids the previous problem where the same absolute
    noise (0.05 mean, 0.15 std) was added to every feature.
    """

    if mode not in DRIFT_PROFILES:
        raise ValueError(
            f"Unknown drift mode '{mode}'. "
            f"Choose from: {list(DRIFT_PROFILES.keys())}"
        )

    rng = np.random.default_rng(random_seed)

    profile = DRIFT_PROFILES[mode]
    noise_fraction = NOISE_SCALE[mode]

    baseline_mean = np.mean(X_baseline, axis=0)
    baseline_std = np.std(X_baseline, axis=0)

    # Protect against zero/very-small standard deviations
    safe_std = np.where(
        baseline_std < 1e-8,
        1.0,
        baseline_std
    )

    X_live = X_baseline.copy()

    print()
    print("=" * 70)
    print(f"[INFO] Generating synthetic live data: {mode.upper()} DRIFT")
    print("=" * 70)

    for i, feature in enumerate(FEATURE_COLUMNS):

        drift_strength = profile[feature]

        # Feature-specific mean shift
        mean_shift = drift_strength * safe_std[i]

        # Feature-specific random noise
        noise_std = noise_fraction * safe_std[i]

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=X_baseline.shape[0],
        )

        X_live[:, i] = (
            X_baseline[:, i]
            + mean_shift
            + noise
        )

        print(
            f"{feature:<22} "
            f"shift={drift_strength:.2f} std, "
            f"noise={noise_fraction:.2f} std"
        )

    return X_live


# ============================================================
# 8. PSI CALCULATION
# ============================================================

def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = NUM_BINS,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    expected:
        Baseline/reference distribution.

    actual:
        Current/live distribution.

    PSI interpretation:
        < 0.10       Stable
        0.10 - 0.25  Warning
        >= 0.25      Significant drift / retraining alert
    """

    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    # Remove NaN and infinite values
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # If both distributions are effectively constant
    if (
        np.ptp(expected) < 1e-12
        and np.ptp(actual) < 1e-12
    ):
        return 0.0

    # Create percentile-based bins from baseline
    percentiles = np.linspace(
        0,
        100,
        num_bins + 1
    )

    bin_edges = np.percentile(
        expected,
        percentiles
    )

    # Remove duplicate edges
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) <= 1:
        return 0.0

    # Extend first and last bins
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts, _ = np.histogram(
        expected,
        bins=bin_edges
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=bin_edges
    )

    expected_pct = (
        expected_counts.astype(float)
        / len(expected)
    )

    actual_pct = (
        actual_counts.astype(float)
        / len(actual)
    )

    # Avoid log(0) and division by zero
    EPSILON = 1e-4

    expected_pct = np.where(
        expected_pct == 0,
        EPSILON,
        expected_pct
    )

    actual_pct = np.where(
        actual_pct == 0,
        EPSILON,
        actual_pct
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


# ============================================================
# 9. PSI INTERPRETATION
# ============================================================

def classify_psi(psi_value: float) -> str:
    """
    Convert PSI score into monitoring status.
    """

    if psi_value < PSI_WARNING_THRESHOLD:
        return "Stable"

    if psi_value < PSI_RETRAIN_THRESHOLD:
        return "Warning"

    return "Retrain Alert"


# ============================================================
# 10. FEATURE STATISTICS
# ============================================================

def calculate_feature_statistics(
    X_baseline: np.ndarray,
    X_live: np.ndarray,
):
    """
    Calculate baseline/live statistics for reporting.
    """

    statistics = {}

    for i, feature in enumerate(FEATURE_COLUMNS):

        baseline_values = X_baseline[:, i]
        live_values = X_live[:, i]

        statistics[feature] = {
            "baseline_mean": round(
                float(np.mean(baseline_values)),
                6
            ),

            "live_mean": round(
                float(np.mean(live_values)),
                6
            ),

            "baseline_std": round(
                float(np.std(baseline_values)),
                6
            ),

            "live_std": round(
                float(np.std(live_values)),
                6
            ),

            "baseline_min": round(
                float(np.min(baseline_values)),
                6
            ),

            "live_min": round(
                float(np.min(live_values)),
                6
            ),

            "baseline_max": round(
                float(np.max(baseline_values)),
                6
            ),

            "live_max": round(
                float(np.max(live_values)),
                6
            ),
        }

    return statistics


# ============================================================
# 11. PLOT PSI
# ============================================================

def plot_feature_psi(
    psi_dict: dict,
    output_path: Path,
):
    """
    Create feature-level PSI bar chart.
    """

    features = list(psi_dict.keys())
    psi_values = [
        float(psi_dict[feature]["psi"])
        for feature in features
    ]

    plt.figure(
        figsize=(10, 6),
        dpi=300
    )

    # Draw each bar separately so that the colour reflects status
    for index, value in enumerate(psi_values):

        status = classify_psi(value)

        if status == "Stable":
            bar_color = "#2ca02c"

        elif status == "Warning":
            bar_color = "#ff7f0e"

        else:
            bar_color = "#d62728"

        plt.barh(
            index,
            value,
            color=bar_color,
            edgecolor="black",
            height=0.55,
        )

        plt.text(
            value + max(0.01, value * 0.01),
            index,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.yticks(
        range(len(features)),
        features
    )

    # PSI thresholds
    plt.axvline(
        x=PSI_WARNING_THRESHOLD,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="Warning (0.10)"
    )

    plt.axvline(
        x=PSI_RETRAIN_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Retrain Alert (0.25)"
    )

    plt.title(
        "Feature-Level PSI Drift Analysis",
        fontsize=13,
        fontweight="bold"
    )

    plt.xlabel("PSI Score")
    plt.ylabel("Sensor Features")

    maximum = max(
        max(psi_values) * 1.15,
        0.30
    )

    plt.xlim(
        0,
        maximum
    )

    plt.grid(
        axis="x",
        alpha=0.25
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. PRINT PSI REPORT
# ============================================================

def print_psi_report(psi_results: dict):
    """
    Print human-readable PSI report.
    """

    print()
    print("=" * 80)
    print("FEATURE-LEVEL PSI DRIFT REPORT")
    print("=" * 80)

    print(
        f"{'Feature':<25}"
        f"{'PSI':>10}"
        f"{'Status':>20}"
    )

    print("-" * 80)

    for feature, result in psi_results.items():

        psi = result["psi"]
        status = result["status"]

        print(
            f"{feature:<25}"
            f"{psi:>10.4f}"
            f"{status:>20}"
        )

    print("=" * 80)

    alert_features = [
        feature
        for feature, result in psi_results.items()
        if result["status"] == "Retrain Alert"
    ]

    warning_features = [
        feature
        for feature, result in psi_results.items()
        if result["status"] == "Warning"
    ]

    if alert_features:

        print()
        print("[ALERT] Significant drift detected in:")

        for feature in alert_features:
            print(f"        - {feature}")

        print()
        print(
            "[ACTION] Investigate the data distribution and "
            "consider retraining if the drift is persistent."
        )

    elif warning_features:

        print()
        print("[WARNING] Moderate drift detected in:")

        for feature in warning_features:
            print(f"        - {feature}")

        print()
        print(
            "[ACTION] Continue monitoring these features."
        )

    else:

        print()
        print(
            "[OK] No significant feature-level drift detected."
        )


# ============================================================
# 13. MAIN DRIFT MONITOR
# ============================================================

def run_drift_monitor():

    print()
    print("=" * 80)
    print("MLOnEdge - FEATURE DRIFT MONITOR")
    print("=" * 80)

    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Dataset      : {DATASET_FILE}")
    print(f"[INFO] Drift mode   : {DRIFT_MODE}")
    print()

    # --------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------

    X_baseline = load_baseline_data()

    # --------------------------------------------------------
    # Generate synthetic live data
    # --------------------------------------------------------

    X_live = generate_live_data(
        X_baseline,
        mode=DRIFT_MODE,
        random_seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # Calculate PSI
    # --------------------------------------------------------

    psi_results = {}

    for i, feature in enumerate(FEATURE_COLUMNS):

        psi_score = calculate_psi(
            X_baseline[:, i],
            X_live[:, i],
            num_bins=NUM_BINS,
        )

        status = classify_psi(
            psi_score
        )

        psi_results[feature] = {
            "psi": round(
                psi_score,
                4
            ),

            "status": status,
        }

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    feature_statistics = calculate_feature_statistics(
        X_baseline,
        X_live,
    )

    # --------------------------------------------------------
    # Save PSI JSON
    # --------------------------------------------------------

    json_path = (
        RESULTS_DIR
        / "drift_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            psi_results,
            f,
            indent=4
        )

    print(
        f"\n[SUCCESS] PSI metrics saved to:\n"
        f"          {json_path}"
    )

    # --------------------------------------------------------
    # Save detailed statistics
    # --------------------------------------------------------

    stats_path = (
        RESULTS_DIR
        / "drift_statistics.json"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_statistics,
            f,
            indent=4
        )

    print(
        f"[SUCCESS] Feature statistics saved to:\n"
        f"          {stats_path}"
    )

    # --------------------------------------------------------
    # Save complete summary
    # --------------------------------------------------------

    summary = {
        "drift_mode": DRIFT_MODE,
        "random_seed": RANDOM_SEED,
        "num_bins": NUM_BINS,

        "thresholds": {
            "warning": PSI_WARNING_THRESHOLD,
            "retrain_alert": PSI_RETRAIN_THRESHOLD,
        },

        "psi_results": psi_results,

        "feature_statistics": feature_statistics,
    }

    summary_path = (
        RESULTS_DIR
        / "drift_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"[SUCCESS] Complete drift summary saved to:\n"
        f"          {summary_path}"
    )

    # --------------------------------------------------------
    # Generate chart
    # --------------------------------------------------------

    chart_path = (
        RESULTS_DIR
        / "psi_chart.png"
    )

    plot_feature_psi(
        psi_results,
        chart_path
    )

    print(
        f"[SUCCESS] PSI chart saved to:\n"
        f"          {chart_path}"
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_psi_report(
        psi_results
    )

    print()
    print("=" * 80)
    print("DRIFT MONITORING COMPLETED")
    print("=" * 80)


# ============================================================
# 14. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_drift_monitor()

    except Exception as exc:

        print()
        print("[ERROR] Drift monitoring failed.")
        print(f"[ERROR] {type(exc).__name__}: {exc}")

        raise