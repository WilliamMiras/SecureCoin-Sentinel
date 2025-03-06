# Detects suspicious wallet clusters and fund movements
# Nodes -> Wallets; Edges -> Transactions
# Creator: Will

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import networkx as nx

# Defining and loading the CSV File
df = pd.read_csv('GNN_train.csv')

# Expected CSV Columns: [sender_wallet, receiver_wallet, scam_label, transaction_amount]

wallets = list(set(df["sender_wallet"]).union(set(df["receiver_wallet"])))
wallet_to_index = {wallet: i for i, wallet in enumerate(wallets)}

# Creating Edge List
edge_index = torch.tensor([
    [wallet_to_index[sender], wallet_to_index[receiver]]
    for sender, receiver in zip(df["sender_wallet"], df["receiver_wallet"])
], dtype=torch.long).t() # Transpose to match PyTorch Geometric format

transaction_counts = df["sender_wallet"].value_counts().reindex(wallets, fill_value=0).values
transaction_amounts = df.groupby("sender_wallet")["transaction_amount"].sum().reindex(wallets, fill_value=0).values

node_features = torch.tensor(list(zip(transaction_counts, transaction_amounts)), dtype=torch.float)

labels = torch.tensor(df.set_index("sender_wallet")["scam_label"].reindex(wallets, fill_value=0).values, dtype=torch.long)

graph_data = Data(x = node_features, edge_index = edge_index, y = labels)

G = nx.Graph()
for wallet in wallets:
    G.add_node(wallet)
for sender, receiver in zip(df["sender_wallet"], df["receiver_wallet"]):
    G.add_edge(sender, receiver)

# Defining the model structure
class ScamDetectionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ScamDetectionGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv1d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Initializing the Model
model = ScamDetectionGNN(in_channels = 2, hidden_channels = 4, out_channels = 2)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
loss_fn = nn.CrossEntropyLoss()

# labels = <input real labels here> TODO Include data 

# Training the model
for epoch in range(100):
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(out, labels)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "./GNN_ScamDetection.pth")