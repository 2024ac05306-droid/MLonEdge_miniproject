"""
Unit and integration tests for feature preprocessing, quantization, and OOD shifts.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from the optimisation module
from optimisation.build_model_variants import get_representative_dataset

from utils import normalize_features


class TestFeatureNormalization:
    """Tests for feature preprocessing and scaling."""

    def test_normalization_shape(self, mock_dataset, mock_stats):
        """Ensure feature shape is preserved after normalization."""
        X_raw, _ = mock_dataset
        mean, std = mock_stats
        X_norm = normalize_features(X_raw, mean, std)
        assert X_norm.shape == X_raw.shape

    def test_zero_mean_unit_variance(self, mock_dataset, mock_stats):
        """Verify that normalized data approaches 0 mean and 1 std."""
        X_raw, _ = mock_dataset
        mean, std = mock_stats
        X_norm = normalize_features(X_raw, mean, std)
        
        np.testing.assert_allclose(np.mean(X_norm, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(X_norm, axis=0), 1.0, atol=1e-5)


class TestRepresentativeDatasetGenerator:
    """Tests for post-training quantization calibration generator."""

    def test_generator_yield_shape_and_type(self, mock_normalized_data):
        """Ensure calibration dataset yields correct shape and float32 dtype for TFLite."""
        X_norm, _ = mock_normalized_data
        rep_gen_fn = get_representative_dataset(X_norm, num_samples=10)
        
        # Instantiate generator
        gen = rep_gen_fn()
        samples = list(gen)

        assert len(samples) == 10
        # First sample in batch format: [array of shape (1, num_features)]
        sample_tensor = samples[0][0]
        assert sample_tensor.shape == (1, X_norm.shape[1])
        assert sample_tensor.dtype == np.float32


class TestOutofDistributionShift:
    """Mandatory OOD Experiment Test Case: Asserts performance drop under +3sigma shift."""

    def test_3sigma_shift_degrades_accuracy(self, trained_base_model, mock_dataset, mock_stats):
        """Asserts that shifting features by +3sigma degrades prediction confidence/accuracy."""
        X_raw, y = mock_dataset
        mean, std = mock_stats

        # 1. Clean Normalized Data
        X_clean = normalize_features(X_raw, mean=mean, std=std)
        preds_clean = np.argmax(trained_base_model.predict(X_clean, verbose=0), axis=1)
        acc_clean = accuracy_score(y, preds_clean)

        # 2. Shifted Data (+3 sigma shift applied to mean)
        shifted_mean = mean + (3.0 * std)
        X_shifted = normalize_features(X_raw, mean=shifted_mean, std=std)
        preds_shifted = np.argmax(trained_base_model.predict(X_shifted, verbose=0), axis=1)
        acc_shifted = accuracy_score(y, preds_shifted)

        # Assert that shifted input produces different predictions and lower or equal accuracy
        assert not np.array_equal(preds_clean, preds_shifted), "Model output should change under +3sigma shift!"
        assert acc_shifted <= acc_clean, "Accuracy should degrade under +3sigma shift!"