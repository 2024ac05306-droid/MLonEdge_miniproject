"""
Pytest configuration and global fixtures for edge deployment testing.
"""

import pytest
import numpy as np
from pathlib import Path
import tensorflow as tf

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import project modules
from utils import normalize_features

# Import project utilities
from utils import normalize_features
from optimisation.build_model_variants import build_base_keras_model, get_representative_dataset


@pytest.fixture(scope="session")
def mock_dataset():
    """Generates synthetic dataset (100 samples, 6 features, 3 classes)."""
    np.random.seed(42)
    X_raw = np.random.uniform(10.0, 100.0, size=(100, 6)).astype(np.float32)
    y = np.random.randint(0, 3, size=(100,)).astype(np.int64)
    return X_raw, y


@pytest.fixture(scope="session")
def mock_stats(mock_dataset):
    """Computes mean and std from mock dataset."""
    X_raw, _ = mock_dataset
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    return mean, std


@pytest.fixture(scope="session")
def mock_normalized_data(mock_dataset, mock_stats):
    """Returns normalized mock features and labels."""
    X_raw, y = mock_dataset
    mean, std = mock_stats
    X_norm = normalize_features(X_raw, mean, std)
    return X_norm, y


@pytest.fixture(scope="session")
def trained_base_model(mock_normalized_data):
    """Returns a trained base Keras model."""
    X_norm, y = mock_normalized_data
    model = build_base_keras_model(input_shape=(X_norm.shape[1],), num_classes=3)
    model.fit(X_norm, y, epochs=2, batch_size=16, verbose=0)
    return model