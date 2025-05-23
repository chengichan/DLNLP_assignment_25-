# model.py – Model training and evaluation module

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import config

# Define a custom LSTM-based classifier as a baseline (non-transformer)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=128, hidden_dim=128, output_dim=2):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, (h_n, _) = self.lstm(x)
        logits = self.fc(h_n.squeeze(0))
        return logits

# Train a transformer model using standard Hugging Face pipeline
def train_model(model, dataloader, val_dataloader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * config.NUM_EPOCHS)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model

# Evaluate a transformer model on test/val set
def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, axis=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1

# Train the custom LSTM model on tokenized text
def train_lstm_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for batch in tqdm(dataloader, desc=f"LSTM Epoch {epoch+1}"):
            input_ids, _, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    return model

# Evaluate the LSTM model on test/val set
def evaluate_lstm_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, _, labels = [b.to(device) for b in batch]
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1
