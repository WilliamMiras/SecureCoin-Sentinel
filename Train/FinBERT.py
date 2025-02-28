# Works with Data Collection team for sentiment analysis - taking the scraped data and analyzing it using sentiment anaylis
# Creator: Palladen

# Code generic from model author site + assistance from Perplexity, might need retrofitting / re-labeling
# This model classifies text into three categories: positive, neutral, and negative sentiment.

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load pre-trained FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Load cryptocurrency dataset
df = pd.read_csv('github_data.csv')
# Prepare the dataset
texts = df['description'].tolist()
labels = df['is_scam'].tolist()

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)


class CryptoDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label)
        }


# Create datasets and dataloaders
train_dataset = CryptoDataset(train_texts, train_labels)
val_dataset = CryptoDataset(val_texts, val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

num_epochs = 3
for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()

    val_accuracy = correct / len(val_dataset)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Accuracy: {val_accuracy:.4f}")

# Save the fine-tuned model
model.save_pretrained("./finbert_crypto_scam_detector")
tokenizer.save_pretrained("./finbert_crypto_scam_detector")
