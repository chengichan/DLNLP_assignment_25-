import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import config

def load_and_process_sst2():
    """
    Loads the SST-2 dataset from HuggingFace GLUE, processes it, and prepares
    data loaders for training, validation, and testing.

    The SST-2 (Stanford Sentiment Treebank v2) dataset is a binary
    classification dataset for sentiment analysis.

    Returns:
        dict: A dictionary containing:
            - "tokenized" (dict): Tokenized input data for each split.
            - "dataloaders" (dict): DataLoader objects for each split.
            - "texts" (dict): Original texts for each split.
            - "labels" (dict): Original labels for each split.
    """
    print("Loading SST-2 from HuggingFace GLUE...")
    # Load the SST-2 subset of the GLUE benchmark.
    # The cache_dir is specified to store downloaded datasets.
    dataset = load_dataset("glue", "sst2", cache_dir="/content/hf_cache")

    # Initialize dictionaries to store processed data for different splits (train, validation, test).
    tokenized = {}
    dataloaders = {}
    texts = {}
    labels = {}

    # Initialize the tokenizer based on the model name specified in config.
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Iterate through the dataset splits.
    for split in ["train", "validation", "test"]:
        # Filter out examples with invalid labels (e.g., -1, which might appear in test sets
        # where labels are not publicly available or are placeholders).
        # We only keep labels 0 (negative sentiment) and 1 (positive sentiment).
        filtered = [
            (s, l) for s, l in zip(dataset[split]["sentence"], dataset[split]["label"])
            if l in [0, 1]
        ]

        # If a split has no valid labels after filtering, print a warning and skip it.
        if not filtered:
            print(f"Skipping split '{split}' due to no valid labels.")
            continue  # Skip empty or test set with label -1

        # Unpack the filtered sentences and labels.
        split_texts, split_labels = zip(*filtered)

        # Tokenize the texts.
        # truncation=True: Truncates sequences longer than max_length.
        # padding=True: Pads sequences shorter than max_length to max_length.
        # max_length: The maximum length to which sequences should be padded or truncated.
        # return_tensors="pt": Returns PyTorch tensors.
        encodings = tokenizer(
            list(split_texts),
            truncation=True,
            padding=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        )

        # Extract input IDs and attention mask from the tokenized encodings.
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        # Convert labels to a PyTorch tensor.
        labels_tensor = torch.tensor(split_labels)

        # Create a TensorDataset from the input IDs, attention mask, and labels.
        # This dataset can be used with a DataLoader.
        dataset_tensor = TensorDataset(input_ids, attention_mask, labels_tensor)
        # Create a DataLoader for the current split.
        loader = DataLoader(dataset_tensor, batch_size=config.BATCH_SIZE)

        # Store the processed data in their respective dictionaries.
        tokenized[split] = encodings
        dataloaders[split] = loader
        texts[split] = split_texts
        labels[split] = split_labels

        # Print summary information for the current split.
        print(f"Split: {split} | Unique labels: {set(split_labels)}")

    # Return all processed data.
    return {
        "tokenized": tokenized,
        "dataloaders": dataloaders,
        "texts": texts,
        "labels": labels,
    }
