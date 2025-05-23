# main.py
import config
from data import load_and_process_sst2
from model import (
    train_model,
    evaluate_model,
    train_lstm_model,
    evaluate_lstm_model
)
from attack_utils import run_adversarial_attacks
import pandas as pd
import matplotlib.pyplot as plt
import os
from sentence_transformers import SentenceTransformer, util
import torch

EXPERIMENTS = {
    "bert_sizes": ["BERT-tiny", "BERT-mini", "BERT-small", "BERT-medium", "BERT-base"],
    "bert_roberta": ["BERT-base", "RoBERTa-base"],
    "distilled_models": ["DistilBERT", "TinyBERT-6L", "MobileBERT", "MiniLM-L6"],
    "non_bert_models": ["BERT-base", "ELECTRA-base", "XLNet-base", "ALBERT-base", "DeBERTa-base", "LSTM"]
}

def compute_semantic_similarity(original_texts, adversarial_texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    original_embeddings = model.encode(original_texts, convert_to_tensor=True)
    adv_embeddings = model.encode(adversarial_texts, convert_to_tensor=True)
    similarities = util.cos_sim(original_embeddings, adv_embeddings)
    return float(torch.diag(similarities).mean())

def run_experiment(model_names, exp_name):
    results = []

    for name in model_names:
        print(f"\n=== Running {exp_name} | Model: {name} ===")
        if name == "LSTM":
            dataloaders, tokenizer = load_and_process_sst2("bert-base-uncased", for_lstm=True)
            model = train_lstm_model(dataloaders["train"], dataloaders["validation"], f"{exp_name}_{name}")
            metrics = evaluate_lstm_model(model, dataloaders["test"])
            adv_results = {}
            similarity = None
        else:
            model_id = config.MODEL_IDS[name]
            dataloaders, tokenizer = load_and_process_sst2(model_id)
            model = train_model(dataloaders["train"], dataloaders["validation"], model_id, f"{exp_name}_{name}")
            metrics = evaluate_model(model, dataloaders["test"])

            original_texts = tokenizer.batch_decode(dataloaders["test"].dataset.tensors[0], skip_special_tokens=True)
            labels = dataloaders["test"].dataset.tensors[2].tolist()

            adv_results, adv_texts, adv_preds = run_adversarial_attacks(
                model, tokenizer, original_texts, labels, return_texts=True, return_preds=True
            )

            similarity = compute_semantic_similarity(original_texts[:len(adv_texts)], adv_texts)
            adv_results["SemanticSimilarity"] = similarity

            # Clean accuracy
            clean_accuracy = metrics["accuracy"]
            # Robustness score: proportion of adversarial examples classified correctly
            robust_correct = sum([p == l for p, l in zip(adv_preds, labels[:len(adv_preds)])])
            robustness_score = robust_correct / len(adv_preds) if adv_preds else 0.0
            adv_results["RobustnessScore"] = robustness_score
            adv_results["CleanAccuracy"] = clean_accuracy

        print(f"[{name}] Clean Metrics: {metrics}")
        print(f"[{name}] Adversarial Results: {adv_results}")

        all_results = {"Experiment": exp_name, "Model": name}
        all_results.update(metrics)
        all_results.update(adv_results)
        results.append(all_results)

    df = pd.DataFrame(results)
    result_path = os.path.join(config.RESULTS_DIR, f"{exp_name}_results.csv")
    df.to_csv(result_path, index=False)
    print(f"Results saved to: {result_path}")

    # Plot Clean Metrics
    metrics_to_plot = ["accuracy", "f1", "precision", "recall", "SemanticSimilarity", "CleanAccuracy", "RobustnessScore"]
    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(10, 6))
        plt.bar(df["Model"], df[metric], color='mediumseagreen')
        plt.title(f"{metric} - {exp_name}")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, f"{exp_name}_{metric}_plot.png"))

    # Plot Adversarial Success Rates
    for col in df.columns:
        if col.endswith("_SuccessRate"):
            plt.figure(figsize=(10, 6))
            plt.bar(df["Model"], df[col], color='tomato')
            plt.title(f"{col} - {exp_name}")
            plt.ylabel("Success Rate")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(config.RESULTS_DIR, f"{exp_name}_{col}_plot.png"))

def main():
    for exp_name, model_list in EXPERIMENTS.items():
        run_experiment(model_list, exp_name)

if __name__ == "__main__":
    main()
