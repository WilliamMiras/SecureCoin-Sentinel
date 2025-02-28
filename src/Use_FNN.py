import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Defining the model structure (same as in the training script)
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


# Loading the trained model
input_dim = 4  # Assuming 4 features: Chain, Losses, Type, Root Causes
model = ScamDetectionFNN(input_dim)
model.load_state_dict(torch.load("FNN_ScamDetection.pth"))
model.eval()

# Initialize preprocessing tools
le = LabelEncoder()
scaler = StandardScaler()

# Load the original dataset to fit the LabelEncoder and StandardScaler
df = pd.read_csv('scam_dataset.csv')
le.fit(df['Chain'])
le.fit(df['Type'])
le.fit(df['Root Causes'])
X = df[['Chain', 'Losses', 'Type', 'Root Causes']]
X['Losses'] = X['Losses'].str.replace('$', '').str.replace(',', '').astype(float)
scaler.fit(X)


def fnn_scam_prob(coin_data, fnn_model):
    """
    :param coin_data: (dict) Dictionary containing coin data
    :param fnn_model: (torch.nn.Module) Trained FNN model
    :return: (float) Probability of being a scam
    """
    # Preprocess the input data
    processed_data = {
        'Chain': le.transform([coin_data['Chain']])[0],
        'Losses': float(coin_data['Losses']),
        'Type': le.transform([coin_data['Type']])[0],
        'Root Causes': le.transform([coin_data['Root Causes']])[0]
    }

    input_data = pd.DataFrame([processed_data])
    input_scaled = scaler.transform(input_data)

    # Convert to PyTorch tensor
    input_tensor = torch.FloatTensor(input_scaled)

    # Get prediction from trained model
    with torch.no_grad():
        prediction = fnn_model(input_tensor)

    return float(prediction[0][0])


# Example usage
example_coin = {
    'Chain': 'ETH',
    'Losses': 100000,
    'Type': 'Combination',
    'Root Causes': 'Combination'
}

scam_probability = fnn_scam_prob(example_coin, model)
print(f"Probability of scam: {scam_probability:.4f}")
