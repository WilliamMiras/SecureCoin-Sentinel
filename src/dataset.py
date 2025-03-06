

import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer


# CodeBERT Dataset
class CodeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_and_tokenize_code_data(xls_path, tokenizer, max_length=512):
    df = pd.read_excel(xls_path)
    df["source_code"] = df["source_code"].str.strip()
    df["label"] = df["label"].str.lower()
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])  # "good" -> 0, "bad" -> 1
    encodings = tokenizer(
        df["source_code"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = df["label"].tolist()
    return CodeDataset(encodings, labels), label_encoder


# FNN Dataset
def load_and_preprocess_fnn_data(csv_path):
    df = pd.read_csv(csv_path)

    # Encode categorical variables
    le = LabelEncoder()
    df['Chain'] = le.fit_transform(df['Chain'])
    df['Type'] = le.fit_transform(df['Type'])
    df['Root Causes'] = le.fit_transform(df['Root Causes'])

    # Preprocess features
    X = df[['Chain', 'Losses', 'Type', 'Root Causes']]
    X['Losses'] = X['Losses'].str.replace('$', '').str.replace(',', '').astype(float)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Assuming all are scams for now (adjust if labels vary)
    y = torch.ones(len(df))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test), scaler