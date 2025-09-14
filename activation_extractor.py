#!/usr/bin/env python3
"""
Activation extraction utilities for ChainScope CoT responses.

This module provides functionality to extract hidden-state activations
from language model responses with proper tokenization and pooling.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from beartype import beartype
from transformers import AutoModel, AutoTokenizer

from chainscope.typing import *

logger = logging.getLogger(__name__)


@beartype
class ActivationExtractor:
    """Extracts hidden-state activations from language model responses."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_activations_for_response(
        self, response_text: str, layers: List[int], exclude_last_n_tokens: int = 5
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations for a single response.

        Args:
            response_text: The CoT response text
            layers: List of layer indices to extract
            exclude_last_n_tokens: Number of tokens to exclude from the end

        Returns:
            Dict mapping layer -> activation vector
        """
        # Tokenize response
        tokens = self.tokenizer(
            response_text, return_tensors="pt", padding=True, truncation=True
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)

        # Extract activations for specified layers
        activations = {}
        for layer in layers:
            if layer < len(outputs.hidden_states):
                # Get activations for this layer
                layer_activations = outputs.hidden_states[
                    layer
                ]  # (batch_size, seq_len, hidden_dim)

                # Mean pool over response tokens only, excluding last N tokens
                seq_len = layer_activations.shape[1]
                if seq_len > exclude_last_n_tokens:
                    # Exclude last N tokens and mean pool
                    pooled_activations = layer_activations[
                        :, :-exclude_last_n_tokens, :
                    ].mean(dim=1)
                else:
                    # If sequence is too short, just mean pool all tokens
                    pooled_activations = layer_activations.mean(dim=1)

                # Convert to numpy and store
                activations[layer] = pooled_activations.cpu().numpy().squeeze()

        return activations

    def extract_activations_batch(
        self,
        responses: List[str],
        layers: List[int],
        batch_size: int = 8,
        exclude_last_n_tokens: int = 5,
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations for a batch of responses.

        Args:
            responses: List of response texts
            layers: List of layer indices to extract
            batch_size: Batch size for processing
            exclude_last_n_tokens: Number of tokens to exclude from the end

        Returns:
            Dict mapping layer -> activation matrix (n_samples, hidden_dim)
        """
        logger.info(
            f"Extracting activations for {len(responses)} responses across {len(layers)} layers..."
        )

        all_activations = {layer: [] for layer in layers}

        # Process in batches
        for i in range(0, len(responses), batch_size):
            batch_responses = responses[i : i + batch_size]

            # Tokenize batch
            tokens = self.tokenizer(
                batch_responses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,  # Adjust based on model
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**tokens, output_hidden_states=True)

            # Extract activations for each layer
            for layer in layers:
                if layer < len(outputs.hidden_states):
                    layer_activations = outputs.hidden_states[
                        layer
                    ]  # (batch_size, seq_len, hidden_dim)

                    # Mean pool over response tokens only, excluding last N tokens
                    batch_size_actual = layer_activations.shape[0]
                    pooled_activations = []

                    for j in range(batch_size_actual):
                        seq_len = layer_activations.shape[1]
                        if seq_len > exclude_last_n_tokens:
                            pooled = layer_activations[
                                j, :-exclude_last_n_tokens, :
                            ].mean(dim=0)
                        else:
                            pooled = layer_activations[j, :, :].mean(dim=0)
                        pooled_activations.append(pooled.cpu().numpy())

                    all_activations[layer].extend(pooled_activations)

            if (i + batch_size) % 100 == 0:
                logger.info(
                    f"Processed {min(i + batch_size, len(responses))}/{len(responses)} responses"
                )

        # Convert to numpy arrays
        for layer in layers:
            all_activations[layer] = np.array(all_activations[layer])

        logger.info(f"Extracted activations: {len(responses)} samples")
        return all_activations


@beartype
class ChainScopeActivationExtractor:
    """Extracts activations from ChainScope CoT responses."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.extractor = ActivationExtractor(model_name, device)

    def extract_iphr_activations(
        self, layers: List[int], sampling_params: Optional[SamplingParams] = None
    ) -> Tuple[Dict[int, np.ndarray], List[Tuple[str, str]]]:
        """
        Extract activations for IPHR responses.

        Args:
            layers: List of layer indices to extract
            sampling_params: Sampling parameters to filter responses

        Returns:
            Tuple of (activations_dict, response_ids) where response_ids is list of (qid, uuid)
        """
        logger.info("Extracting IPHR activations...")

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7, top_p=0.9, max_new_tokens=2000
            )

        # Find IPHR datasets
        questions_dir = DATA_DIR / "questions"
        cot_responses_dir = DATA_DIR / "cot_responses"

        wm_datasets = []
        for subdir in questions_dir.iterdir():
            if subdir.is_dir():
                for file_path in subdir.glob("wm_*.yaml"):
                    wm_datasets.append(file_path.stem)

        logger.info(f"Found {len(wm_datasets)} IPHR datasets: {wm_datasets}")

        all_responses = []
        response_ids = []

        for dataset_id in wm_datasets:
            try:
                # Load dataset params
                params = DatasetParams.from_id(dataset_id)

                # Load CoT responses
                responses_path = params.cot_responses_path(
                    "instr-v0", self.model_name.replace("-", "__"), sampling_params
                )

                if not responses_path.exists():
                    logger.warning(f"Responses file not found: {responses_path}")
                    continue

                cot_responses = CotResponses.load(responses_path)

                # Extract responses
                for qid, response_dict in cot_responses.responses_by_qid.items():
                    for uuid, response in response_dict.items():
                        if isinstance(response, str):
                            all_responses.append(response)
                            response_ids.append((qid, uuid))

            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {e}")
                continue

        logger.info(f"Found {len(all_responses)} IPHR responses")

        # Extract activations
        activations = self.extractor.extract_activations_batch(all_responses, layers)

        return activations, response_ids

    def extract_restoration_errors_activations(
        self, layers: List[int], datasets: List[str] = None
    ) -> Tuple[Dict[int, np.ndarray], List[Tuple[str, str]]]:
        """
        Extract activations for Restoration Errors responses.

        Args:
            layers: List of layer indices to extract
            datasets: List of dataset names (default: ['gsm8k', 'math', 'mmlu'])

        Returns:
            Tuple of (activations_dict, response_ids) where response_ids is list of (qid, uuid)
        """
        logger.info("Extracting Restoration Errors activations...")

        if datasets is None:
            datasets = ["gsm8k", "math", "mmlu"]

        all_responses = []
        response_ids = []

        for dataset_name in datasets:
            try:
                # Load CoT paths
                cot_paths_path = (
                    DATA_DIR
                    / "cot_paths"
                    / dataset_name
                    / f"{self.model_name.replace('-', '__')}.yaml"
                )

                if not cot_paths_path.exists():
                    logger.warning(f"CoT paths file not found: {cot_paths_path}")
                    continue

                cot_paths = CoTPath.load_from_path(cot_paths_path)

                # Extract responses (concatenate all steps)
                for qid, response_dict in cot_paths.cot_path_by_qid.items():
                    for uuid, steps_dict in response_dict.items():
                        # Concatenate all steps to form full response
                        steps = [
                            steps_dict[step_num]
                            for step_num in sorted(steps_dict.keys())
                        ]
                        full_response = "\n".join(steps)

                        all_responses.append(full_response)
                        response_ids.append((qid, uuid))

            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue

        logger.info(f"Found {len(all_responses)} Restoration Errors responses")

        # Extract activations
        activations = self.extractor.extract_activations_batch(all_responses, layers)

        return activations, response_ids
