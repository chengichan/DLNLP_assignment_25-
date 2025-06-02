from run_experiments import run_experiment
import config
import os

# Define experiment groups. Each key represents a group name, and its value is
# a list of model keys (as defined in config.MODEL_IDS) to be tested within that group.
EXPERIMENTS = {
    # Experiment group focusing on different BERT model sizes.
    "bert_sizes": [
        "bert-tiny", "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large"
    ],
    # Experiment group comparing models with different pretraining objectives.
    "pretraining": [
        "bert-base", "roberta-base", "bert-large", "roberta-large"
    ],
    # Experiment group focusing on distilled models.
    "distillation": [
        "distilbert", "tinybert-6l", "tinybert-4l", "mobilebert"
    ],
    # Experiment group comparing different transformer architectures.
    "architectures": [
        "bert-base", "electra-base", "xlnet-base", "albert-base", "deberta-base"
    ]
}

# Base directory where all experiment results (CSV files and plots) will be saved.
BASE_DIR = "results"

print("Starting grouped experiments using TextFooler...\n")

# Iterate through each experiment group defined in the EXPERIMENTS dictionary.
for group_name, model_keys in EXPERIMENTS.items():
    print(f"\nRunning Experiment Group: {group_name}")
    # Create a subdirectory for the current experiment group within the BASE_DIR.
    # os.path.join ensures correct path construction across different operating systems.
    group_dir = os.path.join(BASE_DIR, group_name)
    # Create the directory if it doesn't already exist. exist_ok=True prevents an error
    # if the directory already exists.
    os.makedirs(group_dir, exist_ok=True)

    # Iterate through each model key within the current experiment group.
    for model_key in model_keys:
        # Check if the current model_key is defined in config.MODEL_IDS.
        # This prevents errors if an unknown model key is accidentally listed.
        if model_key not in config.MODEL_IDS:
            print(f"Skipping unknown model: {model_key}")
            continue # Skip to the next model if the current one is not found.

        # Get the full HuggingFace model name or path from the config.
        model_name_or_path = config.MODEL_IDS[model_key]
        print(f"\nRunning {model_key} ({model_name_or_path})")
        # Call the run_experiment function to execute the adversarial attack
        # and evaluation for the current model.
        # group_dir: The directory to save results for this specific group.
        # model_key: The short identifier for the model.
        # model_name_or_path: The full HuggingFace identifier for the model.
        run_experiment(group_dir, model_key, model_name_or_path)
