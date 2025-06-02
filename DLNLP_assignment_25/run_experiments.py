import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from model import build_model, evaluate_model
from data import load_and_process_sst2
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack import Attacker
from textattack.datasets import Dataset
from textattack.attack_args import AttackArgs
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import config

# Initialize a SentenceTransformer model for computing semantic similarity between sentences.
# "all-MiniLM-L6-v2" is a pre-trained model optimized for semantic similarity tasks.
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_semantic_similarity(originals, adversarials):
    """
    Computes the cosine semantic similarity between original and adversarial texts.

    Args:
        originals (list[str]): A list of original sentences.
        adversarials (list[str]): A list of adversarial (perturbed) sentences.

    Returns:
        list[float]: A list of cosine similarity scores, one for each pair of
                     (original, adversarial) texts.
    """
    # Encode the original sentences into embeddings.
    # convert_to_tensor=True ensures the embeddings are PyTorch tensors.
    embeddings1 = similarity_model.encode(originals, convert_to_tensor=True)
    # Encode the adversarial sentences into embeddings.
    embeddings2 = similarity_model.encode(adversarials, convert_to_tensor=True)
    # Compute cosine similarity between all pairs of embeddings.
    # .diagonal() extracts the similarity scores for corresponding pairs (original[i] vs adversarial[i]).
    scores = util.cos_sim(embeddings1, embeddings2).diagonal()
    # Convert the scores from a PyTorch tensor to a list of Python floats.
    return scores.cpu().numpy().tolist()

def run_experiment(experiment_dir, model_key, model_name_or_path):
    """
    Runs a TextFooler adversarial attack experiment on a specified model
    and evaluates its robustness. Results are saved to a CSV and plots are generated.

    Args:
        experiment_dir (str): The directory where experiment results (CSV, plots) will be saved.
        model_key (str): A short identifier for the model (e.g., "bert-base").
        model_name_or_path (str): The full HuggingFace model name or path to load.
    """
    # Ensure the experiment directory exists.
    os.makedirs(experiment_dir, exist_ok=True)

    # Load and process the SST-2 dataset.
    data = load_and_process_sst2()

    # Determine which data split to use for testing.
    # Prioritize "test" split; if not available, fall back to "validation".
    test_loader = data["dataloaders"].get("test") or data["dataloaders"]["validation"]
    test_texts = data["texts"].get("test") or data["texts"]["validation"]
    test_labels = data["labels"].get("test") or data["labels"]["validation"]

    # Inform the user if the test split is not available.
    if "test" not in data["dataloaders"]:
        print("Test split not usable, falling back to validation")

    # Build the classification model using the specified model name/path.
    clf_model = build_model(model_name_or_path)
    # Load the tokenizer corresponding to the model.
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Wrap the classifier model and tokenizer for TextAttack compatibility.
    clf_wrapper = HuggingFaceModelWrapper(clf_model, tokenizer)

    # Evaluate the base accuracy of the model before any attacks.
    base_acc, _ = evaluate_model(clf_model, test_loader)
    print(f"Base accuracy: {base_acc:.4f}")

    # Build the TextFooler attack recipe.
    # TextFoolerJin2019 is an adversarial attack that generates perturbations
    # by replacing words with synonyms to fool the model.
    attacker_instance = TextFoolerJin2019.build(model_wrapper=clf_wrapper)

    # Prepare the dataset for the attack.
    # TextAttack's Dataset expects a list of (text, label) tuples.
    # We limit the number of examples to attack based on config.MAX_ATTACK_EXAMPLES.
    attack_dataset = Dataset(list(zip(test_texts, test_labels))[:config.MAX_ATTACK_EXAMPLES])

    # Configure attack arguments.
    attack_args = AttackArgs(
        num_examples=config.MAX_ATTACK_EXAMPLES, # Number of examples to attack.
        disable_stdout=True, # Suppress TextAttack's verbose stdout during attack.
        random_seed=42, # Set a random seed for reproducibility.
    )

    # Create an Attacker instance with the attack recipe, dataset, and arguments.
    attacker = Attacker(attacker_instance, attack_dataset, attack_args)
    # Run the attack on the dataset. This returns an iterator of attack results.
    results_iter = attacker.attack_dataset()

    # Initialize counters for attack results.
    success, fail, skip = 0, 0, 0
    # Lists to store original and perturbed texts for semantic similarity calculation.
    original_texts, perturbed_texts = [], []

    # Iterate through the attack results. tqdm provides a progress bar.
    for result in tqdm(results_iter):
        # Check the type of attack result.
        if isinstance(result, SuccessfulAttackResult):
            success += 1 # Increment success counter if the attack was successful.
            # Store the original and perturbed texts.
            original_texts.append(result.original_text())
            perturbed_texts.append(result.perturbed_text())
        elif isinstance(result, FailedAttackResult):
            fail += 1 # Increment fail counter if the attack failed.
        else:
            skip += 1 # Increment skip counter if the attack was skipped for some reason.

    # Calculate total examples processed.
    total = success + fail + skip
    # Calculate the attack success rate. Avoid division by zero.
    attack_success_rate = 100 * success / (success + fail) if (success + fail) else 0.0
    # Adversarial accuracy is 100% minus the attack success rate.
    adv_accuracy = 100.0 - attack_success_rate

    # Calculate average semantic similarity.
    avg_similarity = 0.0
    if original_texts: # Only compute if there are successful attacks.
        scores = compute_semantic_similarity(original_texts, perturbed_texts)
        avg_similarity = sum(scores) / len(scores)

    # Calculate the robustness score. This is a common metric that combines
    # base accuracy and attack success rate to give an overall measure of robustness.
    # A higher score indicates better robustness.
    robustness_score = base_acc * (1.0 - attack_success_rate / 100.0)

    # Print a summary of the experiment results for the current model.
    print(f"\nSummary for {model_key}")
    print(f"Base Accuracy: {base_acc:.4f}")
    print(f"Attack Success Rate: {attack_success_rate:.2f}%")
    print(f"Avg Semantic Similarity: {avg_similarity:.4f}")
    print(f"Robustness Score: {robustness_score:.4f}")

    # Create a Pandas DataFrame to store the results.
    df = pd.DataFrame([{
        "model": model_key,
        "attack": "textfooler",
        "base_accuracy": base_acc * 100, # Convert to percentage for consistency.
        "success": success,
        "fail": fail,
        "skip": skip,
        "total": total,
        "attack_success_rate": attack_success_rate,
        "adv_accuracy": adv_accuracy,
        "avg_semantic_similarity": avg_similarity,
        "robustness_score": robustness_score,
    }])

    # Define the output path for the CSV file.
    out_path = os.path.join(experiment_dir, "attack_results.csv")
    # If the CSV file already exists, load it and concatenate the new results.
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        df = pd.concat([prev, df], ignore_index=True)

    # Save the DataFrame to a CSV file. index=False prevents writing the DataFrame index as a column.
    df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")

    # Generate plots for key metrics.
    for metric in ["attack_success_rate", "avg_semantic_similarity", "robustness_score"]:
        plt.figure() # Create a new figure for each plot.
        # Sort the DataFrame by model name for consistent plotting order.
        plot_df = df.sort_values("model")
        # Create a bar plot for the current metric.
        plt.bar(plot_df["model"], plot_df[metric])
        # Set y-axis label (e.g., "Attack Success Rate").
        plt.ylabel(metric.replace("_", " ").title())
        # Set x-axis label.
        plt.xlabel("Model")
        # Set plot title.
        plt.title(metric.replace("_", " ").title())
        # Rotate x-axis labels for better readability if model names are long.
        plt.xticks(rotation=45)
        # Adjust plot layout to prevent labels from overlapping.
        plt.tight_layout()
        # Define the file path for saving the plot.
        fig_path = os.path.join(experiment_dir, f"{metric}.png")
        # Save the plot as a PNG image.
        plt.savefig(fig_path)
        print(f"Plot saved: {fig_path}")
