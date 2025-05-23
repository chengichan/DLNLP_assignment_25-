# data.py – Dataset loading and preprocessing module

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from datasets import load_dataset
import config

def load_and_process_sst2(model_name):
    """
    Load and preprocess the SST-2 dataset using the tokenizer associated with the model_name.

    Args:
        model_name (str): Hugging Face model name or path.

    Returns:
        dict: DataLoaders for train, validation, and test sets; tokenizer; and label count.
    """

    # Load the appropriate tokenizer for the given model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load SST-2 dataset from Hugging Face datasets
    dataset = load_dataset("glue", "sst2")

    def tokenize_function(texts, labels):
        """
        Helper function to tokenize text and convert into PyTorch TensorDataset format.
        """
        encoded = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=config.MAX_LENGTH, return_tensors="pt")
        return TensorDataset(encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels))

    # Create dataloaders for all splits
    dataloaders = {}
    for split in dataset.keys():
        texts = dataset[split]["sentence"]
        labels = dataset[split]["label"]
        tensor_data = tokenize_function(texts, labels)

        # Use random sampler for training, sequential for validation/test
        sampler = RandomSampler(tensor_data) if split == "train" else SequentialSampler(tensor_data)
        dataloaders[split] = DataLoader(tensor_data, sampler=sampler, batch_size=config.BATCH_SIZE)

    return {
        "dataloaders": dataloaders,
        "tokenizer": tokenizer,
        "num_labels": 2  # SST-2 is a binary classification task
    }
