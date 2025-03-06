import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer
from dataset import load_and_tokenize_code_data, load_and_preprocess_fnn_data
from models.hybrid_model import HybridModel

# Load data (assuming some overlap or preprocessing to align datasets)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
code_dataset, _ = load_and_tokenize_code_data("../data/code_labels.xls", tokenizer)
_, fnn_dataset, _ = load_and_preprocess_fnn_data("../data/github_data.csv")  # Using test set for demo

# Split code dataset
train_size = int(0.8 * len(code_dataset))
val_size = len(code_dataset) - train_size
train_code_dataset, val_code_dataset = random_split(code_dataset, [train_size, val_size])
train_loader = DataLoader(train_code_dataset, batch_size=8, shuffle=True)

# Initialize model
fnn_input_dim = fnn_dataset.tensors[0].shape[1]
model = HybridModel(fnn_input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop (simplified, assumes fnn_input aligns with code)
model.train()
for epoch in range(3):
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        fnn_input = torch.rand(8, fnn_input_dim).to(device)  # Placeholder; adjust to real data
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, fnn_input)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# Save model
model.save("../models/hybrid_ScamDetection")