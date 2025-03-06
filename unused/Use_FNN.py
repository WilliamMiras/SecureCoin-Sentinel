import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the same model structure
class ScamDetectionFNN(nn.Module):
    def __init__(self, input_dim):
        super(ScamDetectionFNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Load trained model
model = ScamDetectionFNN(input_dim=4)  # Update input_dim if needed
model.load_state_dict(torch.load("FNN_ScamDetection.pth"))
model.eval()

# Function to predict scam probability
def fnn_scam_prob(data_path, model):
    df = pd.read_csv(data_path)

    # Preprocess data (same steps as training)
    le = LabelEncoder()
    df['Chain'] = le.fit_transform(df['Chain'])
    df['Type'] = le.fit_transform(df['Type'])
    df['Root Causes'] = le.fit_transform(df['Root Causes'])

    X = df[['Chain', 'Losses', 'Type', 'Root Causes']]
    X['Losses'] = X['Losses'].str.replace(r'[$,]', '', regex=True).astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X_scaled)

    # Get predictions
    with torch.no_grad():
        scam_probs = model(X_tensor).squeeze().numpy()  # Convert to NumPy for readability

    df['Scam Probability'] = scam_probs  # Append probability to the dataframe
    return scam_probs
