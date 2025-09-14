#!/usr/bin/env python3
"""
Main script to run deception detection probes on ChainScope faithfulness data.

This script:
1. Builds faithfulness labels from ChainScope eval files
2. Extracts hidden-state activations for CoT responses
3. Trains logistic regression probes with L2 regularization
4. Evaluates probes within and cross-phenomenon
5. Runs layer sweeps and optional directional ablations
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from beartype import beartype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from activation_extractor import ChainScopeActivationExtractor
from chainscope.typing import *

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("chainscope/data")
RESULTS_DIR = Path("results")
PROBES_DIR = Path("probes")
SAMPLES_DIR = Path("samples")

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
PROBES_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

# Probe hyperparameters
PROBE_C = 0.1  # L2 regularization (λ = 10)


@beartype
class FaithfulnessLabeler:
    """Builds faithfulness labels from ChainScope evaluation data."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_id = model_name.replace("-", "__")

    def build_iphr_labels(
        self,
    ) -> Tuple[Dict[str, Dict[str, int]], List[Tuple[str, str]]]:
        """
        Build faithfulness labels for IPHR (Implicit Post-Hoc Rationalization).

        Returns:
            Tuple of (labels_dict, response_ids) where:
            - labels_dict maps qid -> {response_uuid -> faithfulness_label}
            - faithfulness_label: 0 = faithful, 1 = unfaithful
            - response_ids is list of (qid, uuid) pairs
        """
        logger.info("Building IPHR faithfulness labels...")

        labels = {}
        response_ids = []

        # Look for IPHR datasets (wm_* files)
        questions_dir = DATA_DIR / "questions"

        wm_datasets = list(questions_dir.rglob("wm-*.yml")) + list(
            questions_dir.rglob("wm-*.yaml")
        )

        # # Find all wm_* datasets
        # wm_datasets = []
        # for subdir in questions_dir.iterdir():
        #     if subdir.is_dir():
        #         for file_path in subdir.glob("wm_*.yaml"):
        #             wm_datasets.append(file_path.stem)

        logger.info(f"Found {len(wm_datasets)} IPHR datasets: {wm_datasets}")

        for dataset_path in wm_datasets:
            try:
                dataset_id = (
                    dataset_path.stem
                )  # e.g. "wm-world-structure-lat_gt_NO_1_24f1b2bc"

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
                        response_ids.append((qid, response_uuid))

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

        logger.info(
            f"Built IPHR labels for {len(labels)} questions, {len(response_ids)} responses"
        )
        return labels, response_ids

    def build_restoration_errors_labels(
        self,
    ) -> Tuple[Dict[str, Dict[str, int]], List[Tuple[str, str]]]:
        """
        Build faithfulness labels for Restoration Errors.

        Returns:
            Tuple of (labels_dict, response_ids) where:
            - labels_dict maps qid -> {response_uuid -> faithfulness_label}
            - faithfulness_label: 0 = faithful, 1 = unfaithful
            - response_ids is list of (qid, uuid) pairs
        """
        logger.info("Building Restoration Errors faithfulness labels...")

        labels = {}
        response_ids = []

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
                        response_ids.append((qid, response_uuid))

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

        logger.info(
            f"Built Restoration Errors labels for {len(labels)} questions, {len(response_ids)} responses"
        )
        return labels, response_ids


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
    ) -> Dict[str, Any]:
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

        # Classification report
        report = classification_report(labels, predictions, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "ece": ece,
            "layer": layer,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "n_samples": len(labels),
            "n_faithful": np.sum(labels == 0),
            "n_unfaithful": np.sum(labels == 1),
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
        self.extractor = ChainScopeActivationExtractor(model_name)
        self.trainer = ProbeTrainer(model_name, phenomenon)

    def run_experiment(self, layers: List[int], cross_phenomenon: bool = False):
        """Run the complete experiment."""
        logger.info(f"Starting experiment: {self.model_name} on {self.phenomenon}")

        # Build faithfulness labels
        if self.phenomenon == "iphr":
            labels, response_ids = self.labeler.build_iphr_labels()
        elif self.phenomenon == "restoration_errors":
            labels, response_ids = self.labeler.build_restoration_errors_labels()
        elif self.phenomenon == "all":
            iphr_labels, iphr_ids = self.labeler.build_iphr_labels()
            restoration_labels, restoration_ids = (
                self.labeler.build_restoration_errors_labels()
            )
            labels = {**iphr_labels, **restoration_labels}
            response_ids = iphr_ids + restoration_ids
        else:
            raise ValueError(f"Unknown phenomenon: {self.phenomenon}")

        if not labels:
            logger.error("No labels found!")
            return

        # Extract activations
        if self.phenomenon == "iphr":
            activations, _ = self.extractor.extract_iphr_activations(layers)
        elif self.phenomenon == "restoration_errors":
            activations, _ = self.extractor.extract_restoration_errors_activations(
                layers
            )
        elif self.phenomenon == "all":
            iphr_activations, _ = self.extractor.extract_iphr_activations(layers)
            restoration_activations, _ = (
                self.extractor.extract_restoration_errors_activations(layers)
            )
            # Combine activations
            activations = {}
            for layer in layers:
                if layer in iphr_activations and layer in restoration_activations:
                    activations[layer] = np.vstack(
                        [iphr_activations[layer], restoration_activations[layer]]
                    )
                elif layer in iphr_activations:
                    activations[layer] = iphr_activations[layer]
                elif layer in restoration_activations:
                    activations[layer] = restoration_activations[layer]

        # Prepare data for training
        all_labels = []
        for qid, uuid in response_ids:
            if qid in labels and uuid in labels[qid]:
                all_labels.append(labels[qid][uuid])

        all_labels = np.array(all_labels)

        logger.info(
            f"Dataset: {len(all_labels)} samples, {np.sum(all_labels)} unfaithful ({np.mean(all_labels):.3f})"
        )

        # Run layer sweep
        results = {}
        for layer in layers:
            if layer not in activations:
                logger.warning(f"No activations found for layer {layer}")
                continue

            layer_activations = activations[layer]

            if len(layer_activations) != len(all_labels):
                logger.warning(
                    f"Mismatch: {len(layer_activations)} activations vs {len(all_labels)} labels"
                )
                continue

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

            # Plot confusion matrix
            self._plot_confusion_matrix(metrics["confusion_matrix"], layer)

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

    def _plot_confusion_matrix(self, cm: List[List[int]], layer: int):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Faithful", "Unfaithful"],
            yticklabels=["Faithful", "Unfaithful"],
        )
        plt.title(f"Confusion Matrix - Layer {layer}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        results_dir = RESULTS_DIR / self.phenomenon
        results_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            results_dir / f"confusion_matrix_layer_{layer}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _save_results(self, results: Dict[int, Dict[str, Any]]):
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
                    "n_samples": metrics["n_samples"],
                    "n_faithful": metrics["n_faithful"],
                    "n_unfaithful": metrics["n_unfaithful"],
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
    "--layers", default="12,20", help="Comma-separated list of layers to sweep"
)  # default='12,16,20,24,28'
@click.option("--device", default="auto", help="Device to use (auto, cpu, cuda)")
def main(
    model_name: str, phenomenon: str, cross_phenomenon: bool, layers: str, device: str
):
    """Run deception detection probe experiment on ChainScope data."""

    # Parse layers
    layers_list = [int(l) for l in layers.split(",")]

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")
    logger.info(f"Model: {model_name}, Phenomenon: {phenomenon}, Layers: {layers_list}")

    # Run experiment
    runner = ExperimentRunner(model_name, phenomenon)
    runner.run_experiment(layers_list, cross_phenomenon=cross_phenomenon)


if __name__ == "__main__":
    main()
