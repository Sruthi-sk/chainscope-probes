#!/usr/bin/env python3
"""
Example usage script for deception detection probes on ChainScope data.

This script demonstrates how to:
1. Build faithfulness labels
2. Extract activations
3. Train and evaluate probes
4. Run cross-phenomenon experiments
"""

import logging

from activation_extractor import ChainScopeActivationExtractor
from cross_phenomenon_eval import CrossPhenomenonExperimentRunner
from run_deception_probe import ExperimentRunner, FaithfulnessLabeler

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_basic_probe_training():
    """Example: Train probes on IPHR data."""
    logger.info("=== Example: Basic Probe Training ===")

    # Initialize experiment runner
    runner = ExperimentRunner("claude-3.5-sonnet", "iphr")

    # Run experiment with a subset of layers for demonstration
    layers = [12, 16, 20, 24]  # Focus on mid-to-late layers
    runner.run_experiment(layers)

    logger.info("Basic probe training completed!")


def example_cross_phenomenon_evaluation():
    """Example: Cross-phenomenon evaluation."""
    logger.info("=== Example: Cross-Phenomenon Evaluation ===")

    # Initialize cross-phenomenon evaluator
    runner = CrossPhenomenonExperimentRunner("claude-3.5-sonnet")

    # Run cross-phenomenon experiments
    layers = [12, 16, 20, 24]
    results = runner.run_cross_phenomenon_experiment(layers)

    # Print results summary
    for experiment_name, experiment_results in results.items():
        logger.info(f"\n{experiment_name}:")
        for layer, metrics in experiment_results.items():
            logger.info(
                f"  Layer {layer}: Acc={metrics['accuracy']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}"
            )

    logger.info("Cross-phenomenon evaluation completed!")


def example_label_building():
    """Example: Build faithfulness labels manually."""
    logger.info("=== Example: Manual Label Building ===")

    # Initialize labeler
    labeler = FaithfulnessLabeler("claude-3.5-sonnet")

    # Build IPHR labels
    iphr_labels, iphr_ids = labeler.build_iphr_labels()
    logger.info(f"IPHR: {len(iphr_labels)} questions, {len(iphr_ids)} responses")

    # Build Restoration Errors labels
    restoration_labels, restoration_ids = labeler.build_restoration_errors_labels()
    logger.info(
        f"Restoration Errors: {len(restoration_labels)} questions, {len(restoration_ids)} responses"
    )

    # Analyze label distribution
    iphr_unfaithful_count = sum(
        sum(response_labels.values()) for response_labels in iphr_labels.values()
    )
    iphr_total_count = sum(
        len(response_labels) for response_labels in iphr_labels.values()
    )

    logger.info(
        f"IPHR unfaithful rate: {iphr_unfaithful_count}/{iphr_total_count} ({iphr_unfaithful_count/iphr_total_count:.3f})"
    )

    logger.info("Label building completed!")


def example_activation_extraction():
    """Example: Extract activations manually."""
    logger.info("=== Example: Manual Activation Extraction ===")

    # Initialize extractor
    extractor = ChainScopeActivationExtractor("claude-3.5-sonnet")

    # Extract IPHR activations
    layers = [12, 16, 20, 24]
    iphr_activations, iphr_ids = extractor.extract_iphr_activations(layers)

    logger.info("IPHR activations extracted:")
    for layer in layers:
        if layer in iphr_activations:
            logger.info(f"  Layer {layer}: {iphr_activations[layer].shape}")

    # Extract Restoration Errors activations
    restoration_activations, restoration_ids = (
        extractor.extract_restoration_errors_activations(layers)
    )

    logger.info("Restoration Errors activations extracted:")
    for layer in layers:
        if layer in restoration_activations:
            logger.info(f"  Layer {layer}: {restoration_activations[layer].shape}")

    logger.info("Activation extraction completed!")


def example_ablation_experiment():
    """Example: Run ablation experiments."""
    logger.info("=== Example: Ablation Experiments ===")

    # Initialize cross-phenomenon evaluator
    runner = CrossPhenomenonExperimentRunner("claude-3.5-sonnet")

    # Run ablation experiments on IPHR
    layers = [16, 20, 24]  # Focus on late layers
    ablation_results = runner.run_ablation_experiment("iphr", layers, n_samples=5)

    # Print ablation results
    for layer, result in ablation_results.items():
        logger.info(f"Layer {layer} ablation:")
        logger.info(f"  Samples: {result['n_samples']}")
        logger.info(f"  Projection stats: {result['projection_stats']}")

    logger.info("Ablation experiments completed!")


def main():
    """Run all examples."""
    logger.info("Starting deception detection probe examples...")

    try:
        # Example 1: Label building
        example_label_building()

        # Example 2: Activation extraction
        example_activation_extraction()

        # Example 3: Basic probe training
        example_basic_probe_training()

        # Example 4: Cross-phenomenon evaluation
        example_cross_phenomenon_evaluation()

        # Example 5: Ablation experiments
        example_ablation_experiment()

        logger.info("All examples completed successfully!")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
