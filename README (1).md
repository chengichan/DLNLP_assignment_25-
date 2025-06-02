# Adversarial Robustness Experiments with TextFooler

This project evaluates the adversarial robustness of a variety of pre-trained HuggingFace Transformer models using the TextFooler attack. It focuses on how model architecture, size, and pretraining strategy impact vulnerability to adversarial text perturbations, while preserving semantic similarity.

## Overview

The experiments are conducted on the SST-2 sentiment classification task and report four key evaluation metrics:

- **Clean Accuracy**: Model performance on unaltered data.
- **Attack Success Rate**: Proportion of successful adversarial perturbations.
- **Semantic Similarity**: Sentence-level similarity between original and adversarial inputs.
- **Robustness Score**: An aggregate score combining accuracy and resilience.

## Project Structure

```
.
├── config.py
├── data.py
├── model.py
├── run_experiments.py
├── main.py
└── results/
    ├── bert_sizes/
    │   ├── attack_results.csv
    │   ├── attack_success_rate.png
    │   ├── avg_semantic_similarity.png
    │   └── robustness_score.png
    ├── pretraining/
    ├── distillation/
    └── architectures/
```

### File Roles

- **`config.py`**  
  Defines global configuration parameters including batch size, token limits, device settings, and a mapping of model aliases to HuggingFace model IDs.

- **`data.py`**  
  Loads and preprocesses the SST-2 dataset from the GLUE benchmark. Handles tokenisation and constructs PyTorch `DataLoader` objects for training, validation, and test splits.

- **`model.py`**  
  Builds models using HuggingFace's `AutoModelForSequenceClassification` and provides evaluation functionality (accuracy and loss).

- **`run_experiments.py`**  
  Core experiment logic. Loads a model, evaluates clean accuracy, runs TextFooler adversarial attacks, calculates metrics, and generates CSV/visual output.

- **`main.py`**  
  The script to run all experiments. Defines experiment groups (e.g., by model size or pretraining strategy) and executes `run_experiment` for each listed model.

- **`results/`**  
  Directory for all experiment outputs. Subfolders store results grouped by model type or training condition, including raw metrics and bar plots.

## Installation

Ensure you are using a Python 3.8+ environment. Install dependencies using `pip`:

```bash
pip install torch
pip install transformers
pip install datasets
pip install textattack
pip install sentence-transformers
pip install pandas
pip install matplotlib
pip install tqdm
```

We recommend running the code inside a virtual environment to avoid dependency conflicts.

