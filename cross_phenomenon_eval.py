#!/usr/bin/env python3
"""
Cross-phenomenon evaluation and directional ablations for deception detection probes.

This script:
1. Trains probes on one phenomenon and tests on another
2. Runs directional ablations by projecting out learned probe vectors
3. Measures accuracy and faithfulness changes after ablation
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from activation_extractor import ChainScopeActivationExtractor
from chainscope.typing import *
from run_deception_probe import FaithfulnessLabeler

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = Path("/Users/sruth/Documents/GitHub/chainscope/results")
PROBES_DIR = Path("/Users/sruth/Documents/GitHub/chainscope/probes")
SAMPLES_DIR = Path("/Users/sruth/Documents/GitHub/chainscope/samples")


@beartype
class CrossPhenomenonEvaluator:
    """Evaluates probes trained on one phenomenon and tested on another."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.labeler = FaithfulnessLabeler(model_name)
        self.extractor = ChainScopeActivationExtractor(model_name)

    def evaluate_cross_phenomenon(
        self, train_phenomenon: str, test_phenomenon: str, layers: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """
        Train on one phenomenon, test on another.

        Args:
            train_phenomenon: Phenomenon to train on ('iphr' or 'restoration_errors')
            test_phenomenon: Phenomenon to test on ('iphr' or 'restoration_errors')
            layers: List of layers to evaluate

        Returns:
            Dict mapping layer -> evaluation metrics
        """
        logger.info(
            f"Cross-phenomenon evaluation: {train_phenomenon} -> {test_phenomenon}"
        )

        # Load training data
        train_labels, train_ids = self._get_labels_and_ids(train_phenomenon)
        train_activations, _ = self._get_activations(train_phenomenon, layers)

        # Load test data
        test_labels, test_ids = self._get_labels_and_ids(test_phenomenon)
        test_activations, _ = self._get_activations(test_phenomenon, layers)

        results = {}

        for layer in layers:
            if layer not in train_activations or layer not in test_activations:
                continue

            logger.info(f"Evaluating layer {layer}...")

            # Prepare training data
            train_X = train_activations[layer]
            train_y = np.array(
                [
                    train_labels[qid][uuid]
                    for qid, uuid in train_ids
                    if qid in train_labels and uuid in train_labels[qid]
                ]
            )

            # Prepare test data
            test_X = test_activations[layer]
            test_y = np.array(
                [
                    test_labels[qid][uuid]
                    for qid, uuid in test_ids
                    if qid in test_labels and uuid in test_labels[qid]
                ]
            )

            if len(train_X) != len(train_y) or len(test_X) != len(test_y):
                logger.warning(f"Data mismatch at layer {layer}")
                continue

            # Train probe
            scaler = StandardScaler()
            train_X_scaled = scaler.fit_transform(train_X)

            probe = LogisticRegression(
                C=0.1, penalty="l2", random_state=42, max_iter=1000
            )
            probe.fit(train_X_scaled, train_y)

            # Test probe
            test_X_scaled = scaler.transform(test_X)
            test_pred = probe.predict(test_X_scaled)
            test_prob = probe.predict_proba(test_X_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(test_y, test_pred)
            roc_auc = roc_auc_score(test_y, test_prob)

            results[layer] = {
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "train_samples": len(train_y),
                "test_samples": len(test_y),
                "train_faithful": np.sum(train_y == 0),
                "train_unfaithful": np.sum(train_y == 1),
                "test_faithful": np.sum(test_y == 0),
                "test_unfaithful": np.sum(test_y == 1),
            }

            logger.info(f"Layer {layer}: Acc={accuracy:.3f}, ROC-AUC={roc_auc:.3f}")

        return results

    def _get_labels_and_ids(
        self, phenomenon: str
    ) -> Tuple[Dict[str, Dict[str, int]], List[Tuple[str, str]]]:
        """Get labels and response IDs for a phenomenon."""
        if phenomenon == "iphr":
            return self.labeler.build_iphr_labels()
        elif phenomenon == "restoration_errors":
            return self.labeler.build_restoration_errors_labels()
        else:
            raise ValueError(f"Unknown phenomenon: {phenomenon}")

    def _get_activations(
        self, phenomenon: str, layers: List[int]
    ) -> Tuple[Dict[int, np.ndarray], List[Tuple[str, str]]]:
        """Get activations for a phenomenon."""
        if phenomenon == "iphr":
            return self.extractor.extract_iphr_activations(layers)
        elif phenomenon == "restoration_errors":
            return self.extractor.extract_restoration_errors_activations(layers)
        else:
            raise ValueError(f"Unknown phenomenon: {phenomenon}")


@beartype
class DirectionalAblator:
    """Runs directional ablations by projecting out learned probe vectors."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.extractor = ChainScopeActivationExtractor(model_name)

    def run_directional_ablation(
        self, phenomenon: str, layer: int, n_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Run directional ablation by projecting out probe vector.

        Args:
            phenomenon: Phenomenon to ablate ('iphr' or 'restoration_errors')
            layer: Layer to ablate
            n_samples: Number of samples to ablate

        Returns:
            Dictionary with ablation results
        """
        logger.info(f"Running directional ablation on {phenomenon} layer {layer}")

        # Load probe
        probe_path = PROBES_DIR / self.model_name / phenomenon / f"layer_{layer}.npz"
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe not found: {probe_path}")

        probe_data = np.load(probe_path)
        probe_weights = probe_data["weights"].flatten()
        probe_bias = probe_data["bias"]
        scaler_mean = probe_data["scaler_mean"]
        scaler_scale = probe_data["scaler_scale"]

        # Load activations
        if phenomenon == "iphr":
            activations, response_ids = self.extractor.extract_iphr_activations([layer])
        elif phenomenon == "restoration_errors":
            activations, response_ids = (
                self.extractor.extract_restoration_errors_activations([layer])
            )
        else:
            raise ValueError(f"Unknown phenomenon: {phenomenon}")

        if layer not in activations:
            raise ValueError(f"No activations found for layer {layer}")

        layer_activations = activations[layer]

        # Sample random indices
        np.random.seed(42)
        sample_indices = np.random.choice(
            len(layer_activations),
            min(n_samples, len(layer_activations)),
            replace=False,
        )

        ablation_results = []

        for idx in sample_indices:
            original_activation = layer_activations[idx].copy()
            qid, uuid = response_ids[idx]

            # Project out probe direction
            # Normalize probe weights
            probe_direction = probe_weights / np.linalg.norm(probe_weights)

            # Project activation onto probe direction
            projection = np.dot(original_activation, probe_direction)

            # Remove projection
            ablated_activation = original_activation - projection * probe_direction

            # Store results
            ablation_results.append(
                {
                    "sample_id": f"{qid}_{uuid}",
                    "qid": qid,
                    "uuid": uuid,
                    "original_activation": original_activation.tolist(),
                    "ablated_activation": ablated_activation.tolist(),
                    "projection_magnitude": float(projection),
                    "probe_direction": probe_direction.tolist(),
                }
            )

        # Save ablation samples
        samples_dir = SAMPLES_DIR / "ablation" / self.model_name / phenomenon
        samples_dir.mkdir(parents=True, exist_ok=True)

        ablation_file = samples_dir / f"layer_{layer}_ablation.jsonl"
        with open(ablation_file, "w") as f:
            for result in ablation_results:
                json.dump(result, f)
                f.write("\n")

        logger.info(
            f"Saved {len(ablation_results)} ablation samples to {ablation_file}"
        )

        return {
            "n_samples": len(ablation_results),
            "layer": layer,
            "phenomenon": phenomenon,
            "ablation_file": str(ablation_file),
            "projection_stats": {
                "mean": np.mean([r["projection_magnitude"] for r in ablation_results]),
                "std": np.std([r["projection_magnitude"] for r in ablation_results]),
                "min": np.min([r["projection_magnitude"] for r in ablation_results]),
                "max": np.max([r["projection_magnitude"] for r in ablation_results]),
            },
        }


@beartype
class CrossPhenomenonExperimentRunner:
    """Runs cross-phenomenon experiments and ablations."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.evaluator = CrossPhenomenonEvaluator(model_name)
        self.ablator = DirectionalAblator(model_name)

    def run_cross_phenomenon_experiment(self, layers: List[int]):
        """Run cross-phenomenon evaluation experiments."""
        logger.info("Running cross-phenomenon experiments...")

        # Define experiment pairs
        experiments = [("iphr", "restoration_errors"), ("restoration_errors", "iphr")]

        all_results = {}

        for train_phenomenon, test_phenomenon in experiments:
            logger.info(f"Experiment: {train_phenomenon} -> {test_phenomenon}")

            results = self.evaluator.evaluate_cross_phenomenon(
                train_phenomenon, test_phenomenon, layers
            )

            all_results[f"{train_phenomenon}_to_{test_phenomenon}"] = results

            # Save results
            self._save_cross_phenomenon_results(
                train_phenomenon, test_phenomenon, results
            )

        return all_results

    def run_ablation_experiment(
        self, phenomenon: str, layers: List[int], n_samples: int = 10
    ):
        """Run directional ablation experiments."""
        logger.info(f"Running ablation experiments on {phenomenon}...")

        ablation_results = {}

        for layer in layers:
            try:
                result = self.ablator.run_directional_ablation(
                    phenomenon, layer, n_samples
                )
                ablation_results[layer] = result
            except Exception as e:
                logger.error(f"Error running ablation on layer {layer}: {e}")
                continue

        # Save ablation results
        self._save_ablation_results(phenomenon, ablation_results)

        return ablation_results

    def _save_cross_phenomenon_results(
        self,
        train_phenomenon: str,
        test_phenomenon: str,
        results: Dict[int, Dict[str, float]],
    ):
        """Save cross-phenomenon results."""
        results_dir = RESULTS_DIR / "cross_phenomenon"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        results_file = results_dir / f"{train_phenomenon}_to_{test_phenomenon}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save as CSV
        csv_data = []
        for layer, metrics in results.items():
            csv_data.append(
                {
                    "layer": layer,
                    "accuracy": metrics["accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "train_samples": metrics["train_samples"],
                    "test_samples": metrics["test_samples"],
                    "train_faithful": metrics["train_faithful"],
                    "train_unfaithful": metrics["train_unfaithful"],
                    "test_faithful": metrics["test_faithful"],
                    "test_unfaithful": metrics["test_unfaithful"],
                }
            )

        df = pd.DataFrame(csv_data)
        csv_file = results_dir / f"{train_phenomenon}_to_{test_phenomenon}.csv"
        df.to_csv(csv_file, index=False)

        logger.info(f"Saved cross-phenomenon results to {results_dir}")

    def _save_ablation_results(
        self, phenomenon: str, results: Dict[int, Dict[str, Any]]
    ):
        """Save ablation results."""
        results_dir = RESULTS_DIR / "ablation"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / f"{phenomenon}_ablation.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved ablation results to {results_file}")


@click.command()
@click.option(
    "--model_name", required=True, help="Model name (e.g., claude-3.5-sonnet)"
)
@click.option(
    "--experiment_type",
    required=True,
    type=click.Choice(["cross_phenomenon", "ablation", "both"]),
    help="Type of experiment to run",
)
@click.option(
    "--phenomenon",
    type=click.Choice(["iphr", "restoration_errors"]),
    help="Phenomenon for ablation (required if experiment_type is ablation)",
)
@click.option(
    "--layers", default="12,16,20,24", help="Comma-separated list of layers to evaluate"
)
@click.option("--n_samples", default=10, help="Number of samples for ablation")
def main(
    model_name: str,
    experiment_type: str,
    phenomenon: Optional[str],
    layers: str,
    n_samples: int,
):
    """Run cross-phenomenon evaluation and ablation experiments."""

    # Parse layers
    layers_list = [int(l) for l in layers.split(",")]

    # Validate arguments
    if experiment_type == "ablation" and phenomenon is None:
        raise click.BadParameter("phenomenon is required for ablation experiments")

    logger.info(
        f"Model: {model_name}, Experiment: {experiment_type}, Layers: {layers_list}"
    )

    # Run experiments
    runner = CrossPhenomenonExperimentRunner(model_name)

    if experiment_type in ["cross_phenomenon", "both"]:
        logger.info("Running cross-phenomenon experiments...")
        cross_results = runner.run_cross_phenomenon_experiment(layers_list)
        logger.info("Cross-phenomenon experiments completed!")

    if experiment_type in ["ablation", "both"]:
        logger.info(f"Running ablation experiments on {phenomenon}...")
        ablation_results = runner.run_ablation_experiment(
            phenomenon, layers_list, n_samples
        )
        logger.info("Ablation experiments completed!")

    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
