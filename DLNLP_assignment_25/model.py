import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config

def build_model(model_name):
    """
    Builds and loads a pre-trained sequence classification model from HuggingFace
    and moves it to the specified device (CPU or GPU).

    Args:
        model_name (str): The name or path of the pre-trained model to load
                          (e.g., "bert-base-uncased").

    Returns:
        transformers.PreTrainedModel: The loaded model configured for sequence classification.
    """
    # Load the pre-trained model for sequence classification.
    # num_labels=2 indicates a binary classification task (e.g., sentiment analysis).
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    # Move the model to the device specified in the config (CPU or CUDA).
    model.to(config.DEVICE)
    return model

def evaluate_model(model, dataloader):
    """
    Evaluates the given model's performance (accuracy and average loss) on a dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing batches
                                                  of input_ids, attention_mask, and labels.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The classification accuracy of the model.
            - avg_loss (float): The average cross-entropy loss over the dataset.
    """
    # Set the model to evaluation mode. This disables dropout and batch normalization updates.
    model.eval()
    correct = 0  # Counter for correctly predicted samples.
    total = 0    # Counter for total samples processed.
    total_loss = 0 # Accumulator for the total loss.
    # Define the loss function as CrossEntropyLoss, suitable for classification tasks.
    loss_fn = CrossEntropyLoss()

    # Disable gradient calculations during evaluation to save memory and speed up computation.
    with torch.no_grad():
        # Iterate through batches in the dataloader.
        for batch in dataloader:
            # Move each tensor in the batch to the specified device.
            input_ids, attention_mask, labels = [x.to(config.DEVICE) for x in batch]

            # Pass the inputs through the model.
            # The 'labels' are also passed to the model to compute the loss internally
            # if the model is a HuggingFace model with a classification head.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Extract logits (raw prediction scores) from the model's output.
            logits = outputs.logits
            # Calculate the loss using the logits and true labels.
            loss = loss_fn(logits, labels)
            # Accumulate the loss. .item() gets the scalar value from a tensor.
            total_loss += loss.item()
            # Get the predicted class by finding the index of the maximum logit for each sample.
            preds = torch.argmax(logits, dim=1)
            # Count correct predictions. (preds == labels) creates a boolean tensor,
            # .sum() counts True values, and .item() converts to a Python scalar.
            correct += (preds == labels).sum().item()
            # Add the number of samples in the current batch to the total count.
            total += labels.size(0)

    # Calculate overall accuracy.
    accuracy = correct / total
    # Calculate average loss per batch.
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss
