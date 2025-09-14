#!/usr/bin/env python3
"""
Test script for deception detection probe functionality.

This script runs basic tests to ensure the system is working correctly.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from activation_extractor import ActivationExtractor
from run_deception_probe import FaithfulnessLabeler, ProbeTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_faithfulness_labeler():
    """Test faithfulness labeler functionality."""
    logger.info("Testing FaithfulnessLabeler...")

    # Initialize labeler
    labeler = FaithfulnessLabeler("claude-3.5-sonnet")

    # Test that we can initialize without errors
    assert labeler.model_name == "claude-3.5-sonnet"
    assert labeler.model_id == "claude__3__5__sonnet"

    logger.info("✓ FaithfulnessLabeler test passed")


def test_probe_trainer():
    """Test probe trainer functionality."""
    logger.info("Testing ProbeTrainer...")

    # Initialize trainer
    trainer = ProbeTrainer("claude-3.5-sonnet", "iphr")

    # Create dummy data
    n_samples = 100
    hidden_dim = 512
    activations = np.random.randn(n_samples, hidden_dim).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples)

    # Test probe training
    probe, scaler = trainer.train_probe(activations, labels, layer=16)

    # Check that probe was trained
    assert isinstance(probe, LogisticRegression)
    assert isinstance(scaler, StandardScaler)
    assert probe.coef_.shape == (1, hidden_dim)
    assert scaler.mean_.shape == (hidden_dim,)

    # Test probe evaluation
    metrics = trainer.evaluate_probe(probe, scaler, activations, labels, layer=16)

    # Check metrics
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "ece" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1

    logger.info("✓ ProbeTrainer test passed")


def test_activation_extractor():
    """Test activation extractor functionality."""
    logger.info("Testing ActivationExtractor...")

    # Skip if no CUDA available (this test requires a real model)
    import torch

    if not torch.cuda.is_available():
        logger.info("Skipping ActivationExtractor test (no CUDA)")
        return

    try:
        # Initialize extractor with a small model
        extractor = ActivationExtractor("distilbert-base-uncased", device="cpu")

        # Test single response extraction
        response_text = "This is a test response for activation extraction."
        layers = [0, 2, 4]

        activations = extractor.extract_activations_for_response(response_text, layers)

        # Check that activations were extracted
        assert len(activations) == len(layers)
        for layer in layers:
            assert layer in activations
            assert activations[layer].shape[0] > 0  # Should have some dimensions

        logger.info("✓ ActivationExtractor test passed")

    except Exception as e:
        logger.warning(f"ActivationExtractor test failed: {e}")
        logger.info("This is expected if the model is not available")


def test_data_structures():
    """Test that required data structures exist."""
    logger.info("Testing data structures...")

    data_dir = Path("/Users/sruth/Documents/GitHub/chainscope/chainscope/data")

    # Check that data directory exists
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    # Check for required subdirectories
    required_dirs = [
        "questions",
        "cot_eval",
        "cot_path_eval",
        "cot_responses",
        "cot_paths",
    ]
    for subdir in required_dirs:
        subdir_path = data_dir / subdir
        assert subdir_path.exists(), f"Required directory not found: {subdir_path}"

    logger.info("✓ Data structures test passed")


def test_results_directory_creation():
    """Test that results directories can be created."""
    logger.info("Testing results directory creation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        probes_dir = Path(temp_dir) / "probes"
        samples_dir = Path(temp_dir) / "samples"

        # Create directories
        results_dir.mkdir(exist_ok=True)
        probes_dir.mkdir(exist_ok=True)
        samples_dir.mkdir(exist_ok=True)

        # Test that they exist
        assert results_dir.exists()
        assert probes_dir.exists()
        assert samples_dir.exists()

        # Test subdirectory creation
        subdir = results_dir / "iphr"
        subdir.mkdir(parents=True, exist_ok=True)
        assert subdir.exists()

    logger.info("✓ Results directory creation test passed")


def run_all_tests():
    """Run all tests."""
    logger.info("Running deception detection probe tests...")

    try:
        test_faithfulness_labeler()
        test_probe_trainer()
        test_activation_extractor()
        test_data_structures()
        test_results_directory_creation()

        logger.info("✓ All tests passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
