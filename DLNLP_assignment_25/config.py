import torch

# Basic configuration settings for the experiments.
BATCH_SIZE = 32  # Batch size for data loaders.
MAX_LENGTH = 128  # Maximum sequence length for tokenizer truncation and padding.
MAX_ATTACK_EXAMPLES = 100  # Maximum number of examples to use for adversarial attacks.
MODEL_NAME = "bert-base-uncased"  # Default model name for tokenizer initialization.
TOKENIZER = "bert-base-uncased"  # Default tokenizer name.

# Configuration for HuggingFace Datasets.
HF_DATA_DIR = "/content/hf_cache"  # Directory to cache HuggingFace datasets.

# Device configuration.
# Automatically detects if a CUDA-enabled GPU is available; otherwise, uses CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dictionary mapping short model keys to their full HuggingFace model identifiers.
MODEL_IDS = {
    "bert-tiny": "prajjwal1/bert-tiny",
    "bert-mini": "prajjwal1/bert-mini",
    "bert-small": "prajjwal1/bert-small",
    "bert-medium": "prajjwal1/bert-medium",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "distilbert": "distilbert-base-uncased",
    "tinybert-6l": "huawei-noah/TinyBERT_General_6L_768D",
    "tinybert-4l": "huawei-noah/TinyBERT_General_4L_312D",
    "mobilebert": "google/mobilebert-uncased",
    "electra-base": "google/electra-base-discriminator",
    "xlnet-base": "xlnet-base-cased",
    "albert-base": "albert-base-v2",
    "deberta-base": "microsoft/deberta-base"
}
