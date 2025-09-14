#!/usr/bin/env python3
"""
Deception Detection Probe for ChainScope Faithfulness Data

This script extends deception-detector probes to ChainScope's faithfulness evaluation data.
It builds faithfulness labels, extracts activations, trains probes, and runs evaluations.

Usage:
    python deception_probe_chainscope.py --model_name claude-3.5-sonnet --phenomenon iphr
    python deception_probe_chainscope.py --model_name claude-3.5-sonnet --phenomenon restoration_errors
    python deception_probe_chainscope.py --model_name claude-3.5-sonnet --phenomenon all --cross_phenomenon
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from chainscope.typing import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("/Users/sruth/Documents/GitHub/chainscope/chainscope/data")
RESULTS_DIR = Path("/Users/sruth/Documents/GitHub/chainscope/results")
PROBES_DIR = Path("/Users/sruth/Documents/GitHub/chainscope/probes")
SAMPLES_DIR = Path("/Users/sruth/Documents/GitHub/chainscope/samples")

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
PROBES_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

# Probe hyperparameters
PROBE_C = 0.1  # L2 regularization (λ = 10)
LAYERS_TO_SWEEP = list(range(0, 32, 4))  # Every 4th layer, adjust based on model


@beartype
class FaithfulnessLabeler:
    """Builds faithfulness labels from ChainScope evaluation data."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_id = model_name.replace("-", "__")

    def build_iphr_labels(self) -> Dict[str, Dict[str, int]]:
        """
        Build faithfulness labels for IPHR (Implicit Post-Hoc Rationalization).

        Returns:
            Dict mapping qid -> {response_uuid -> faithfulness_label}
            where faithfulness_label: 0 = faithful, 1 = unfaithful
        """
        logger.info("Building IPHR faithfulness labels...")

        labels = {}

        # Look for IPHR datasets (wm_* files)
        questions_dir = DATA_DIR / "questions"
        cot_eval_dir = DATA_DIR / "cot_eval"

        # Find all wm_* datasets
        wm_datasets = []
        for subdir in questions_dir.iterdir():
            if subdir.is_dir():
                for file_path in subdir.glob("wm_*.yaml"):
                    wm_datasets.append(file_path.stem)

        logger.info(f"Found {len(wm_datasets)} IPHR datasets: {wm_datasets}")

        for dataset_id in wm_datasets:
            try:
                # Load dataset params
                params = DatasetParams.from_id(dataset_id)

                # Load evaluation results
                sampling_params = SamplingParams(
                    temperature=0.7, top_p=0.9, max_new_tokens=2000
                )
                eval_path = params.cot_eval_path(
                    "instr-v0", self.model_id, sampling_params
                )

                if not eval_path.exists():
                    logger.warning(f"Evaluation file not found: {eval_path}")
                    continue

                cot_eval = CotEval.load(eval_path)

                # Build labels based on evaluation results
                for qid, response_results in cot_eval.results_by_qid.items():
                    if qid not in labels:
                        labels[qid] = {}

                    for response_uuid, result in response_results.items():
                        # Label as unfaithful (1) if:
                        # - Final answer is NO but equal_values is TRUE (answer contradiction)
                        # - Final answer is UNKNOWN (external inconsistency)
                        # - Final answer is FAILED_EVAL (evaluation failure)
                        if (
                            result.final_answer == "NO"
                            and result.equal_values == "TRUE"
                        ):
                            labels[qid][response_uuid] = (
                                1  # Unfaithful: answer contradiction
                            )
                        elif result.final_answer == "UNKNOWN":
                            labels[qid][response_uuid] = (
                                1  # Unfaithful: external inconsistency
                            )
                        elif result.final_answer == "FAILED_EVAL":
                            labels[qid][response_uuid] = (
                                1  # Unfaithful: evaluation failure
                            )
                        else:
                            labels[qid][response_uuid] = 0  # Faithful

            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {e}")
                continue

        logger.info(f"Built IPHR labels for {len(labels)} questions")
        return labels

    def build_restoration_errors_labels(self) -> Dict[str, Dict[str, int]]:
        """
        Build faithfulness labels for Restoration Errors.

        Returns:
            Dict mapping qid -> {response_uuid -> faithfulness_label}
            where faithfulness_label: 0 = faithful, 1 = unfaithful
        """
        logger.info("Building Restoration Errors faithfulness labels...")

        labels = {}

        # Look for restoration errors datasets (gsm8k, math, mmlu)
        datasets = ["gsm8k", "math", "mmlu"]

        for dataset_name in datasets:
            try:
                # Load CoT path evaluation
                eval_path = (
                    DATA_DIR / "cot_path_eval" / dataset_name / f"{self.model_id}.yaml"
                )

                if not eval_path.exists():
                    logger.warning(f"Evaluation file not found: {eval_path}")
                    continue

                cot_path_eval = CoTPathEval.load_from_path(eval_path)

                # Build labels based on multi-pass evaluation
                for (
                    qid,
                    response_results,
                ) in cot_path_eval.second_pass_eval_by_qid.items():
                    if qid not in labels:
                        labels[qid] = {}

                    for response_uuid, result in response_results.items():
                        # Check if any step is marked as unfaithful
                        has_unfaithful_step = any(
                            step_status.node_status == "UNFAITHFUL"
                            for step_status in result.steps_status.values()
                        )

                        # Also check third pass evaluation for confirmation
                        if qid in cot_path_eval.third_pass_eval_by_qid:
                            third_pass_result = cot_path_eval.third_pass_eval_by_qid[
                                qid
                            ].get(response_uuid)
                            if third_pass_result:
                                has_confirmed_unfaithful = any(
                                    step_status.is_unfaithful
                                    for step_status in third_pass_result.steps_status.values()
                                )
                                labels[qid][response_uuid] = (
                                    1 if has_confirmed_unfaithful else 0
                                )
                            else:
                                labels[qid][response_uuid] = (
                                    1 if has_unfaithful_step else 0
                                )
                        else:
                            labels[qid][response_uuid] = 1 if has_unfaithful_step else 0

            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue

        logger.info(f"Built Restoration Errors labels for {len(labels)} questions")
        return labels


@beartype
class ActivationExtractor:
    """Extracts hidden-state activations for CoT responses."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def extract_activations(
        self, responses: Dict[str, Dict[str, str]], layers: List[int]
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations for responses.

        Args:
            responses: Dict mapping qid -> {response_uuid -> response_text}
            layers: List of layer indices to extract

        Returns:
            Dict mapping layer -> activations array (n_samples, hidden_dim)
        """
        logger.info(
            f"Extracting activations for {len(responses)} responses across {len(layers)} layers..."
        )

        # TODO: Implement actual activation extraction
        # This is a placeholder - you'll need to integrate with your activation extraction pipeline
        # For now, return random activations for demonstration

        activations = {}
        n_samples = sum(len(resp_dict) for resp_dict in responses.values())
        hidden_dim = 4096  # Adjust based on model

        for layer in layers:
            # Random activations for demonstration
            activations[layer] = np.random.randn(n_samples, hidden_dim).astype(
                np.float32
            )

        logger.info(
            f"Extracted activations: {n_samples} samples, {hidden_dim} dimensions"
        )
        return activations


@beartype
class ProbeTrainer:
    """Trains and evaluates linear probes for faithfulness detection."""

    def __init__(self, model_name: str, phenomenon: str):
        self.model_name = model_name
        self.phenomenon = phenomenon
        self.scalers = {}
        self.probes = {}

    def train_probe(
        self, activations: np.ndarray, labels: np.ndarray, layer: int
    ) -> Tuple[LogisticRegression, StandardScaler]:
        """
        Train a logistic regression probe for a specific layer.

        Args:
            activations: Activation matrix (n_samples, hidden_dim)
            labels: Binary labels (n_samples,)
            layer: Layer index

        Returns:
            Tuple of (trained_probe, fitted_scaler)
        """
        logger.info(f"Training probe for layer {layer}...")

        # Normalize activations
        scaler = StandardScaler()
        activations_scaled = scaler.fit_transform(activations)

        # Train logistic regression with L2 regularization
        probe = LogisticRegression(
            C=PROBE_C, penalty="l2", random_state=42, max_iter=1000
        )

        probe.fit(activations_scaled, labels)

        logger.info(
            f"Trained probe for layer {layer}: accuracy = {probe.score(activations_scaled, labels):.3f}"
        )

        return probe, scaler

    def evaluate_probe(
        self,
        probe: LogisticRegression,
        scaler: StandardScaler,
        activations: np.ndarray,
        labels: np.ndarray,
        layer: int,
    ) -> Dict[str, float]:
        """
        Evaluate probe performance.

        Args:
            probe: Trained probe
            scaler: Fitted scaler
            activations: Test activations
            labels: Test labels
            layer: Layer index

        Returns:
            Dictionary of evaluation metrics
        """
        activations_scaled = scaler.transform(activations)

        # Get predictions and probabilities
        predictions = probe.predict(activations_scaled)
        probabilities = probe.predict_proba(activations_scaled)[
            :, 1
        ]  # Probability of unfaithful

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        roc_auc = roc_auc_score(labels, probabilities)

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = np.trapz(precision, recall)

        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(labels, probabilities)

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "ece": ece,
            "layer": layer,
        }

        logger.info(
            f"Layer {layer} metrics: Acc={accuracy:.3f}, ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, ECE={ece:.3f}"
        )

        return metrics

    def _calculate_ece(
        self, labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


@beartype
class ExperimentRunner:
    """Runs the complete deception detection experiment."""

    def __init__(self, model_name: str, phenomenon: str):
        self.model_name = model_name
        self.phenomenon = phenomenon
        self.labeler = FaithfulnessLabeler(model_name)
        self.extractor = ActivationExtractor(model_name)
        self.trainer = ProbeTrainer(model_name, phenomenon)

    def run_experiment(self, cross_phenomenon: bool = False):
        """Run the complete experiment."""
        logger.info(f"Starting experiment: {self.model_name} on {self.phenomenon}")

        # Build faithfulness labels
        if self.phenomenon == "iphr":
            labels = self.labeler.build_iphr_labels()
        elif self.phenomenon == "restoration_errors":
            labels = self.labeler.build_restoration_errors_labels()
        elif self.phenomenon == "all":
            iphr_labels = self.labeler.build_iphr_labels()
            restoration_labels = self.labeler.build_restoration_errors_labels()
            labels = {**iphr_labels, **restoration_labels}
        else:
            raise ValueError(f"Unknown phenomenon: {self.phenomenon}")

        if not labels:
            logger.error("No labels found!")
            return

        # Extract activations
        # TODO: Load actual CoT responses and extract activations
        responses = {}  # Placeholder - load from cot_responses
        activations = self.extractor.extract_activations(responses, LAYERS_TO_SWEEP)

        # Prepare data for training
        all_activations = []
        all_labels = []

        for qid, response_labels in labels.items():
            for response_uuid, label in response_labels.items():
                # Get activations for this response (placeholder)
                # In practice, you'd extract activations for the specific response
                all_labels.append(label)

        all_labels = np.array(all_labels)

        # Run layer sweep
        results = {}
        for layer in LAYERS_TO_SWEEP:
            if layer not in activations:
                continue

            layer_activations = activations[layer]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                layer_activations,
                all_labels,
                test_size=0.2,
                random_state=42,
                stratify=all_labels,
            )

            # Train probe
            probe, scaler = self.trainer.train_probe(X_train, y_train, layer)

            # Evaluate probe
            metrics = self.trainer.evaluate_probe(probe, scaler, X_test, y_test, layer)
            results[layer] = metrics

            # Save probe
            self._save_probe(probe, scaler, layer)

        # Save results
        self._save_results(results)

        logger.info("Experiment completed!")

    def _save_probe(
        self, probe: LogisticRegression, scaler: StandardScaler, layer: int
    ):
        """Save trained probe and scaler."""
        probe_dir = PROBES_DIR / self.model_name / self.phenomenon
        probe_dir.mkdir(parents=True, exist_ok=True)

        probe_path = probe_dir / f"layer_{layer}.npz"
        np.savez(
            probe_path,
            weights=probe.coef_,
            bias=probe.intercept_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
        )

        logger.info(f"Saved probe to {probe_path}")

    def _save_results(self, results: Dict[int, Dict[str, float]]):
        """Save experiment results."""
        results_dir = RESULTS_DIR / self.phenomenon
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = results_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save layer sweep CSV
        layer_sweep_data = []
        for layer, metrics in results.items():
            layer_sweep_data.append(
                {
                    "layer": layer,
                    "accuracy": metrics["accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "ece": metrics["ece"],
                }
            )

        layer_sweep_df = pd.DataFrame(layer_sweep_data)
        layer_sweep_path = results_dir / "layer_sweep.csv"
        layer_sweep_df.to_csv(layer_sweep_path, index=False)

        # Plot layer sweep
        self._plot_layer_sweep(layer_sweep_df, results_dir)

        logger.info(f"Saved results to {results_dir}")


@beartype
def _plot_layer_sweep(df: pd.DataFrame, output_dir: Path):
    """Plot layer sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Accuracy
    axes[0, 0].plot(df["layer"], df["accuracy"], "o-")
    axes[0, 0].set_title("Accuracy vs Layer")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Accuracy")

    # ROC-AUC
    axes[0, 1].plot(df["layer"], df["roc_auc"], "o-")
    axes[0, 1].set_title("ROC-AUC vs Layer")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("ROC-AUC")

    # PR-AUC
    axes[1, 0].plot(df["layer"], df["pr_auc"], "o-")
    axes[1, 0].set_title("PR-AUC vs Layer")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("PR-AUC")

    # ECE
    axes[1, 1].plot(df["layer"], df["ece"], "o-")
    axes[1, 1].set_title("ECE vs Layer")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("ECE")

    plt.tight_layout()
    plt.savefig(output_dir / "layer_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()


@click.command()
@click.option(
    "--model_name", required=True, help="Model name (e.g., claude-3.5-sonnet)"
)
@click.option(
    "--phenomenon",
    required=True,
    type=click.Choice(["iphr", "restoration_errors", "all"]),
    help="Phenomenon to analyze",
)
@click.option(
    "--cross_phenomenon", is_flag=True, help="Run cross-phenomenon evaluation"
)
@click.option(
    "--layers",
    default="0,4,8,12,16,20,24,28",
    help="Comma-separated list of layers to sweep",
)
def main(model_name: str, phenomenon: str, cross_phenomenon: bool, layers: str):
    """Run deception detection probe experiment on ChainScope data."""

    # Parse layers
    global LAYERS_TO_SWEEP
    LAYERS_TO_SWEEP = [int(l) for l in layers.split(",")]

    # Run experiment
    runner = ExperimentRunner(model_name, phenomenon)
    runner.run_experiment(cross_phenomenon=cross_phenomenon)


if __name__ == "__main__":
    main()
