from textattack.attack_recipes import TextFoolerJin2019, BERTAttackLi2020, DeepWordBugGao2018
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs
import torch

def run_adversarial_attacks(model, tokenizer, texts, labels, max_samples=50, return_texts=False, return_preds=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    wrapper = HuggingFaceModelWrapper(model, tokenizer)
    dataset_list = [(text, label) for text, label in zip(texts[:max_samples], labels[:max_samples])]

    attacks = {
        "TextFooler": TextFoolerJin2019.build(wrapper),
        "BERTAttack": BERTAttackLi2020.build(wrapper),
        "DeepWordBug": DeepWordBugGao2018.build(wrapper)
    }

    results = {}
    adv_texts = []
    adv_preds = []
    for name, attack in attacks.items():
        print(f"Running {name}...")
        attack_args = AttackArgs(num_examples=len(dataset_list), disable_stdout=True)
        attacker = Attacker(attack, Dataset(dataset_list), attack_args)
        attack_results = attacker.attack_dataset()
        success_count = sum(1 for r in attack_results if r.goal_status == "Succeeded")
        results[f"{name}_SuccessRate"] = success_count / len(dataset_list)
        if return_texts or return_preds:
            for r in attack_results:
                adv_text = r.perturbed_text() if r.goal_status == "Succeeded" else r.original_text()
                adv_texts.append(adv_text)
                if return_preds:
                    inputs = tokenizer(adv_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        pred = torch.argmax(outputs.logits, dim=1).cpu().item()
                    adv_preds.append(pred)
    if return_texts or return_preds:
        return results, adv_texts, adv_preds
    return results
