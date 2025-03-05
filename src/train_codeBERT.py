import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer, AdamW
from tqdm import tqdm
from dataset import load_and_tokenize_data
from models.codebert import CodeBERTClassifier #codeBert

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = CodeBERTClassifier(num_labels=2)

# Load dataset
dataset, _ = load_and_tokenize_data("../data/code_labels.xls", tokenizer)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(3):
    loop = tqdm(train_loader)
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# Save model
model.save("../models/my_codebert_goodbad_model")
tokenizer.save_pretrained("../models/my_codebert_goodbad_model")