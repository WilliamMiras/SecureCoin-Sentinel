import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer


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


def load_and_tokenize_data(xls_path, tokenizer, max_length=512):
    # Load Excel file
    df = pd.read_excel(xls_path)

    # Clean and preprocess
    df["source_code"] = df["source_code"].str.strip()
    df["label"] = df["label"].str.lower()  # Normalize labels

    # Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])  # "good" -> 0, "bad" -> 1

    # Tokenize source code
    encodings = tokenizer(
        df["source_code"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = df["label"].tolist()
    return CodeDataset(encodings, labels), label_encoder