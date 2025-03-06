#Liquidity, Tokenomics, Transaction Analysis
#Creator: Evan

import torch
import torch.nn as nn



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

    def save(self, path):
        torch.save(self.state_dict(), f"{path}/fnn_state.pt")

    @classmethod
    def load(cls, input_dim, path):
        model = cls(input_dim)
        model.load_state_dict(torch.load(f"{path}/fnn_state.pt"))
        return model