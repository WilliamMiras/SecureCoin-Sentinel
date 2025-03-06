import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import load_and_preprocess_fnn_data
from models.fnn import ScamDetectionFNN

# Load data
train_dataset, test_dataset, scaler = load_and_preprocess_fnn_data("../data/github_data.csv")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model
input_dim = train_dataset.tensors[0].shape[1]  # Number of features
model = ScamDetectionFNN(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Training loop
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    total_test_loss = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        test_loss = loss_fn(outputs.squeeze(), y_batch)
        total_test_loss += test_loss.item()
    print(f'Test Loss: {total_test_loss / len(test_loader):.4f}')

# Save model
model.save("../models/fnn_scam_detection")