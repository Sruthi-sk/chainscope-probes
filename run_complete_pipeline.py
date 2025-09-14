#!/usr/bin/env python3
"""
Complete pipeline script for deception detection probes on ChainScope data.

This script runs the entire pipeline:
1. Train probes on IPHR data
2. Train probes on Restoration Errors data
3. Run cross-phenomenon evaluation
4. Run ablation experiments
5. Generate comprehensive results

Usage:
    python run_complete_pipeline.py --model_name claude-3.5-sonnet
"""

import logging
from pathlib import Path

import click

from cross_phenomenon_eval import CrossPhenomenonExperimentRunner
from run_deception_probe import ExperimentRunner

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model_name", required=True, help="Model name (e.g., claude-3.5-sonnet)"
)
@click.option(
    "--layers",
    default="0,4,8,12,16,20,24,28",
    help="Comma-separated list of layers to sweep",
)
@click.option(
    "--skip_training", is_flag=True, help="Skip probe training (use existing probes)"
)
@click.option(
    "--skip_cross_phenomenon", is_flag=True, help="Skip cross-phenomenon evaluation"
)
@click.option("--skip_ablation", is_flag=True, help="Skip ablation experiments")
@click.option("--n_ablation_samples", default=10, help="Number of samples for ablation")
def main(
    model_name: str,
    layers: str,
    skip_training: bool,
    skip_cross_phenomenon: bool,
    skip_ablation: bool,
    n_ablation_samples: int,
):
    """Run complete deception detection pipeline."""

    # Parse layers
    layers_list = [int(l) for l in layers.split(",")]

    logger.info(f"Starting complete pipeline for {model_name}")
    logger.info(f"Layers: {layers_list}")
    logger.info(f"Skip training: {skip_training}")
    logger.info(f"Skip cross-phenomenon: {skip_cross_phenomenon}")
    logger.info(f"Skip ablation: {skip_ablation}")

    # Step 1: Train probes on individual phenomena
    if not skip_training:
        logger.info("=== Step 1: Training Probes ===")

        # Train on IPHR
        logger.info("Training probes on IPHR...")
        iphr_runner = ExperimentRunner(model_name, "iphr")
        iphr_runner.run_experiment(layers_list)

        # Train on Restoration Errors
        logger.info("Training probes on Restoration Errors...")
        restoration_runner = ExperimentRunner(model_name, "restoration_errors")
        restoration_runner.run_experiment(layers_list)

        # Train on combined data
        logger.info("Training probes on combined data...")
        combined_runner = ExperimentRunner(model_name, "all")
        combined_runner.run_experiment(layers_list)

        logger.info("Probe training completed!")
    else:
        logger.info("Skipping probe training (using existing probes)")

    # Step 2: Cross-phenomenon evaluation
    if not skip_cross_phenomenon:
        logger.info("=== Step 2: Cross-Phenomenon Evaluation ===")

        cross_runner = CrossPhenomenonExperimentRunner(model_name)
        cross_results = cross_runner.run_cross_phenomenon_experiment(layers_list)

        # Print summary
        logger.info("Cross-phenomenon results summary:")
        for experiment_name, experiment_results in cross_results.items():
            logger.info(f"\n{experiment_name}:")
            best_layer = max(
                experiment_results.keys(),
                key=lambda l: experiment_results[l]["roc_auc"],
            )
            best_metrics = experiment_results[best_layer]
            logger.info(
                f"  Best layer: {best_layer} (ROC-AUC: {best_metrics['roc_auc']:.3f})"
            )

            for layer, metrics in experiment_results.items():
                logger.info(
                    f"    Layer {layer}: Acc={metrics['accuracy']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}"
                )

        logger.info("Cross-phenomenon evaluation completed!")
    else:
        logger.info("Skipping cross-phenomenon evaluation")

    # Step 3: Ablation experiments
    if not skip_ablation:
        logger.info("=== Step 3: Ablation Experiments ===")

        ablation_runner = CrossPhenomenonExperimentRunner(model_name)

        # Run ablations on IPHR
        logger.info("Running ablations on IPHR...")
        iphr_ablation_results = ablation_runner.run_ablation_experiment(
            "iphr", layers_list, n_ablation_samples
        )

        # Run ablations on Restoration Errors
        logger.info("Running ablations on Restoration Errors...")
        restoration_ablation_results = ablation_runner.run_ablation_experiment(
            "restoration_errors", layers_list, n_ablation_samples
        )

        # Print ablation summary
        logger.info("Ablation results summary:")
        logger.info(f"IPHR ablations: {len(iphr_ablation_results)} layers")
        logger.info(
            f"Restoration Errors ablations: {len(restoration_ablation_results)} layers"
        )

        logger.info("Ablation experiments completed!")
    else:
        logger.info("Skipping ablation experiments")

    # Step 4: Generate final summary
    logger.info("=== Step 4: Final Summary ===")

    # Check results directories
    results_dir = Path("/Users/sruth/Documents/GitHub/chainscope/results")

    if results_dir.exists():
        logger.info("Results saved to:")
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                logger.info(f"  {subdir.name}/")
                for file in subdir.iterdir():
                    if file.is_file():
                        logger.info(f"    {file.name}")

    logger.info("Complete pipeline finished successfully!")

    # Print next steps
    logger.info("\n=== Next Steps ===")
    logger.info("1. Review results in the results/ directory")
    logger.info("2. Check layer sweep plots for optimal layers")
    logger.info("3. Analyze cross-phenomenon generalization")
    logger.info("4. Examine ablation samples for interpretability")
    logger.info("5. Use trained probes for inference on new data")


if __name__ == "__main__":
    main()
