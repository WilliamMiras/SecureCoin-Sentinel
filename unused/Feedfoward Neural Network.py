#Liquidity, Tokenomics, Transaction Analysis
#Creator: Evan

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Defining the model structure
class ScamDetectionFNN(nn.Module):
    def __init__(self, input_dim):
        super(ScamDetectionFNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.output_dim = 1 # For hybrid model compatibility

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


# Load and preprocess data
df = pd.read_csv('github_data.csv')

le = LabelEncoder()
df['Chain'] = le.fit_transform(df['Chain'])
df['Type'] = le.fit_transform(df['Type'])
df['Root Causes'] = le.fit_transform(df['Root Causes'])

X = df[['Chain', 'Losses', 'Type', 'Root Causes']]
X['Losses'] = X['Losses'].str.replace('$', '').str.replace(',', '').astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = torch.ones(len(df))  # Assuming all entries are scams

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Initialize the model
input_dim = X_train.shape[1]
model = ScamDetectionFNN(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = loss_fn(test_outputs.squeeze(), y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), "FNN_ScamDetection.pth")
