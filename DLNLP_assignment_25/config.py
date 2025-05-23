# config.py – Configuration file for adversarial robustness NLP project

import os
from pathlib import Path

# Define base project paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# General training parameters
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
EVALUATION_STEPS = 500
SAVE_STEPS = 1000
SEED = 42

# Memory thresholds for adaptive model selection (in GB)
GPU_MEMORY_THRESHOLD = 8
MIN_FREE_MEMORY = 2

# Number of test samples to use in adversarial evaluation
MAX_ATTACK_EXAMPLES = 100

# Attack modification budget (fraction of input allowed to be changed)
ATTACK_BUDGET = {
    "textfooler": 0.2,
    "bertattack": 0.2,
    "textbugger": 0.2,
    "deepwordbug": 0.15
}

# Mapping of human-readable model names to Hugging Face model IDs
MODEL_IDS = {
    "BERT-tiny": "prajjwal1/bert-tiny",
    "BERT-mini": "prajjwal1/bert-mini",
    "BERT-small": "prajjwal1/bert-small",
    "BERT-medium": "prajjwal1/bert-medium",
    "BERT-base": "bert-base-uncased",
    "BERT-large": "bert-large-uncased",
    "RoBERTa-base": "roberta-base",
    "RoBERTa-large": "roberta-large",
    "DistilBERT": "distilbert-base-uncased",
    "TinyBERT-6L": "huawei-noah/TinyBERT_General_6L_768D",
    "TinyBERT-4L": "huawei-noah/TinyBERT_General_4L_312D",
    "MobileBERT": "google/mobilebert-uncased",
    "MiniLM-L6": "microsoft/MiniLM-L6-H384-uncased",
    "ELECTRA-base": "google/electra-base-discriminator",
    "XLNet-base": "xlnet-base-cased",
    "ALBERT-base": "albert-base-v2",
    "DeBERTa-base": "microsoft/deberta-base",
    "T5-base": "t5-base"
}

# Optional: Approximate model sizes for reporting
PARAMETER_COUNTS = {
    "BERT-tiny": 4.4,
    "BERT-mini": 11.3,
    "BERT-small": 29.1,
    "BERT-medium": 41.7,
    "BERT-base": 110,
    "BERT-large": 340,
    "RoBERTa-base": 125,
    "RoBERTa-large": 355,
    "DistilBERT": 66,
    "TinyBERT-6L": 67,
    "TinyBERT-4L": 14.5,
    "MobileBERT": 25,
    "MiniLM-L6": 22,
    "ELECTRA-base": 110,
    "XLNet-base": 110,
    "ALBERT-base": 12,
    "DeBERTa-base": 140,
    "T5-base": 220,
    "LSTM": 2.5,
    "CNN": 3.2
}

# Dataset information
DATASET_NAME = "sst2"
DATASET_SPLIT = {
    "train": "train",
    "validation": "validation",
    "test": "test"
}

# Organise different experimental categories
EXPERIMENT_DIRS = {
    "bert_sizes": os.path.join(RESULTS_DIR, "bert_sizes"),
    "bert_roberta": os.path.join(RESULTS_DIR, "bert_roberta"),
    "distilled_models": os.path.join(RESULTS_DIR, "distilled_models"),
    "non_bert_models": os.path.join(RESULTS_DIR, "non_bert_models")
}

# Create directories for storing results if they don’t already exist
for dir_path in EXPERIMENT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)
