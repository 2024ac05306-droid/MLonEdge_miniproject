"""
Centralized configuration for the MLOnEdge project.
Loads all environment variables from .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path_value: str | os.PathLike[str], default: str) -> Path:
    """Resolve paths relative to the project root when they are not absolute."""

    candidate = Path(path_value or default)

    if candidate.is_absolute():
        return candidate

    return (PROJECT_ROOT / candidate).resolve()


# --------------------------------------------------------
# Load .env
# --------------------------------------------------------

load_dotenv()

# --------------------------------------------------------
# MQTT
# --------------------------------------------------------

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

# --------------------------------------------------------
# MLflow
# --------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "G44_logibridge_miniproject"
)

# --------------------------------------------------------
# Dataset
# --------------------------------------------------------

DATASET_DIR = resolve_path(
    os.getenv("DATASET_DIR", "./data"),
    "./data"
)

DATASET_FILE = DATASET_DIR / os.getenv(
    "DATASET_FILE",
    "training_dataset.csv"
)

TRAINING_STATS = DATASET_DIR / os.getenv(
    "TRAINING_STATS",
    "training_stats.npy"
)

# --------------------------------------------------------
# Models
# --------------------------------------------------------

MODEL_DIR = resolve_path(
    os.getenv("MODEL_DIR", "./models"),
    "./models"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / os.getenv(
    "MODEL_NAME",
    "best_model.keras"
)

PRUNED_MODEL_FILE = MODEL_DIR / os.getenv(
    "PRUNED_MODEL_NAME",
    "best_model_pruned.keras"
)

PTQ_MODEL_FILE = MODEL_DIR / os.getenv(
    "PTQ_MODEL_NAME",
    "model_ptq.tflite"
)

QAT_MODEL_FILE = MODEL_DIR / os.getenv(
    "QAT_MODEL_NAME",
    "model_qat.tflite"
)

TFLITE_MODEL_FILE = MODEL_DIR / os.getenv(
    "TFLITE_MODEL_NAME",
    "best_model_int8.tflite"
)

MODEL_FORMAT = os.getenv(
    "MODEL_FORMAT",
    "keras"
)

# --------------------------------------------------------
# Training
# --------------------------------------------------------

TRAINING_EPOCHS = int(
    os.getenv("TRAINING_EPOCHS", 100)
)

PRUNING_EPOCHS = int(
    os.getenv("PRUNING_EPOCHS", 10)
)

BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", 32)
)

TEST_SIZE = float(
    os.getenv("TEST_SIZE", 0.20)
)

RANDOM_STATE = int(
    os.getenv("RANDOM_STATE", 42)
)

# --------------------------------------------------------
# Output Directories
# --------------------------------------------------------

LOG_DIR = resolve_path(
    os.getenv("LOG_DIR", "./logs"),
    "./logs"
)

OUTPUT_DIR = resolve_path(
    os.getenv("OUTPUT_DIR", "./outputs"),
    "./outputs"
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# Dataset Columns
# --------------------------------------------------------

FEATURE_COLUMNS = [
    "temp_mean",
    "temp_std",
    "temp_rate",
    "vibration_rms",
    "vibration_peak",
    "vibration_kurtosis",
]

TARGET_COLUMN = "label"

# Classification
NUM_CLASSES = 3

CLASS_NAMES = [
    "Normal",
    "Warning",
    "Critical",
]