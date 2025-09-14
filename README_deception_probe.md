# Deception Detection Probe for ChainScope Faithfulness Data

This repository extends deception-detector probes to ChainScope's faithfulness evaluation data, enabling detection of unfaithful reasoning patterns in language model responses.

## Overview

The system builds faithfulness labels from ChainScope's evaluation files, extracts hidden-state activations from CoT responses, trains linear probes, and evaluates them across different phenomena and layers.

## Features

- **Faithfulness Labeling**: Automatically builds binary labels from ChainScope eval files
- **Activation Extraction**: Mean-pools activations over response tokens (excluding last 5 tokens)
- **Linear Probes**: Logistic regression with L2 regularization (C=0.1, λ=10)
- **Layer Sweeps**: Evaluates probes across multiple layers
- **Cross-Phenomenon Evaluation**: Trains on one phenomenon, tests on another
- **Directional Ablations**: Projects out learned probe vectors to measure impact

## Data Sources

### Implicit Post-Hoc Rationalization (IPHR)
- **Questions**: `chainscope/data/questions/` (wm_*.yaml files)
- **CoTs**: `chainscope/data/cot_responses/`
- **Eval Labels**: `chainscope/data/cot_eval/`

### Restoration Errors
- **Problems**: `chainscope/data/problems/`
- **CoT Paths**: `chainscope/data/cot_paths/`
- **Eval Labels**: `chainscope/data/cot_path_eval/`

## Labeling Policy

Binary target = UNFAITHFUL=1:

- **IPHR/Argument-Switching/Answer-Switching**: Unfaithful if ChainScope eval flags external inconsistency, switching, or answer contradiction between logically equivalent pairs
- **Restoration Errors**: Unfaithful if intermediate steps contain errors that are "restored" to a correct final answer
- **Faithful/Consistent**: Anything labeled faithful/consistent in ChainScope eval → 0

## Installation

```bash
# Install dependencies
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn click beartype jaxtyping

# Clone ChainScope repository
git clone https://github.com/jettjaniak/chainscope.git
cd chainscope
```

## Usage

### Basic Probe Training

```bash
# Train probes on IPHR data
python run_deception_probe.py --model_name claude-3.5-sonnet --phenomenon iphr

# Train probes on Restoration Errors data
python run_deception_probe.py --model_name claude-3.5-sonnet --phenomenon restoration_errors

# Train probes on both phenomena combined
python run_deception_probe.py --model_name claude-3.5-sonnet --phenomenon all
```

### Cross-Phenomenon Evaluation

```bash
# Train on IPHR, test on Restoration Errors
python cross_phenomenon_eval.py --model_name claude-3.5-sonnet --experiment_type cross_phenomenon

# Run ablation experiments
python cross_phenomenon_eval.py --model_name claude-3.5-sonnet --experiment_type ablation --phenomenon iphr

# Run both experiments
python cross_phenomenon_eval.py --model_name claude-3.5-sonnet --experiment_type both --phenomenon iphr
```

### Custom Layer Sweeps

```bash
# Specify custom layers
python run_deception_probe.py --model_name claude-3.5-sonnet --phenomenon iphr --layers "0,8,16,24,32"
```

## Output Structure

```
results/
├── iphr/
│   ├── metrics.json              # Detailed metrics per layer
│   ├── layer_sweep.csv          # Layer sweep summary
│   ├── layer_sweep.png          # Layer sweep plots
│   └── confusion_matrix_layer_*.png
├── restoration_errors/
│   └── ...
├── cross_phenomenon/
│   ├── iphr_to_restoration_errors.json
│   └── restoration_errors_to_iphr.json
└── ablation/
    └── iphr_ablation.json

probes/
├── claude-3.5-sonnet/
│   ├── iphr/
│   │   └── layer_*.npz          # Trained probe weights and scalers
│   └── restoration_errors/
│       └── layer_*.npz

samples/
└── ablation/
    └── claude-3.5-sonnet/
        └── iphr/
            └── layer_*_ablation.jsonl  # Ablation samples
```

## Metrics

- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **ECE**: Expected Calibration Error
- **Confusion Matrix**: Detailed classification breakdown

## Configuration

### Probe Hyperparameters
- **Regularization**: C=0.1 (L2 penalty λ=10)
- **Activation Pooling**: Mean over response tokens, exclude last 5 tokens
- **Normalization**: StandardScaler (zero mean, unit variance)

### Default Layer Sweep
- Layers: [0, 4, 8, 12, 16, 20, 24, 28]
- Adjust based on model architecture

## Examples

### Example 1: Basic IPHR Probe Training

```python
from run_deception_probe import ExperimentRunner

# Initialize experiment
runner = ExperimentRunner("claude-3.5-sonnet", "iphr")

# Run experiment with default layers
runner.run_experiment([0, 4, 8, 12, 16, 20, 24, 28])
```

### Example 2: Cross-Phenomenon Evaluation

```python
from cross_phenomenon_eval import CrossPhenomenonExperimentRunner

# Initialize cross-phenomenon evaluator
runner = CrossPhenomenonExperimentRunner("claude-3.5-sonnet")

# Run cross-phenomenon experiments
results = runner.run_cross_phenomenon_experiment([12, 16, 20, 24])
```

### Example 3: Loading Trained Probes

```python
import numpy as np

# Load probe weights and scaler
probe_data = np.load("probes/claude-3.5-sonnet/iphr/layer_16.npz")
weights = probe_data['weights']
bias = probe_data['bias']
scaler_mean = probe_data['scaler_mean']
scaler_scale = probe_data['scaler_scale']

# Use for inference
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

probe = LogisticRegression()
probe.coef_ = weights
probe.intercept_ = bias

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in activation extraction
2. **Missing Evaluation Files**: Check that ChainScope data is properly downloaded
3. **Model Loading Errors**: Ensure model name matches HuggingFace format

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=/path/to/chainscope:$PYTHONPATH
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chainscope2024,
  title={Chain-of-Thought Reasoning In The Wild Is Not Always Faithful},
  author={Janiak, Jett and others},
  journal={arXiv preprint arXiv:2024},
  year={2024}
}
```
