# Adversarial Robustness in NLP

This project evaluates the robustness of various NLP models against adversarial attacks using the SST-2 sentiment classification dataset. It compares different Transformer architectures, distilled models, and a traditional LSTM baseline using standard adversarial attack methods.

## 📦 Project Structure

```text
├── config.py              # Global configuration, model list, paths, hyperparameters
├── data.py                # Dataset loading, tokenisation, and preprocessing
├── model.py               # Training and evaluation for Transformer and LSTM models
├── attack_utils.py        # Applies adversarial attacks using TextAttack
├── main.py                # Runs the full pipeline: train, evaluate, attack, log results
├── plot_results.py        # Optional: Generates bar charts from result summary
├── requirements.txt       # Required Python packages
└── results/               # Output metrics, scores, and plots
```

## ⚙️ Prerequisites

Install dependencies using:

```bash
pip install -r requirements.txt
```

Requirements include:
- transformers
- datasets
- textattack
- sentence-transformers
- scikit-learn
- matplotlib
- seaborn

## 📁 Role of Each Python File

### `config.py`
Contains paths, training hyperparameters, model names, attack settings, and dataset splits. Central to controlling experiment behaviour.

### `data.py`
Loads the SST-2 dataset from Hugging Face. Tokenises text using the appropriate model tokenizer and returns `DataLoader` objects for training, validation, and test sets.

### `model.py`
Defines model training and evaluation logic. Supports both:
- Hugging Face Transformer models
- A custom PyTorch-based LSTM model

### `attack_utils.py`
Runs adversarial attacks (TextFooler, BERTAttack, DeepWordBug) using the TextAttack library. Calculates the attack success rate and returns attack results.

### `main.py`
Main pipeline script. Trains each model, evaluates clean performance, performs attacks, computes semantic similarity, and logs results (accuracy, ASR, composite score) to `summary.csv`.

### `plot_results.py`
Optional utility to generate bar plots of:
- Composite Robustness Score
- Attack Success Rate (ASR)

Plots are saved in the `results/` directory.

## 🚀 How to Run

### 🖥️ Local Execution

1. Clone or extract the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```

4. (Optional) Generate plots:
   ```bash
   python plot_results.py
   ```

### 💻 Google Colab

1. Upload the zipped project and unzip it.
2. Run:
   ```python
   !pip install -r requirements.txt
   !python main.py
   !python plot_results.py
   ```

## 📊 Output

Results are saved to `results/summary.csv` with the following columns:
- Model
- Attack type
- Accuracy (clean test)
- Attack Success Rate (ASR)
- Semantic Similarity
- Composite Robustness Score

Plots of ASR and robustness are also saved as PNG files in the results folder.


