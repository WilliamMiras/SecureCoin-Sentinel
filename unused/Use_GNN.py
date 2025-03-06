import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

# Defining model struct (same as GNN_Train.py definition)
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

# Loading the trained model
model = ScamDetectionGNN(in_channels = 2, hidden_channels = 4, out_channels = 2)
model.load_state_dict(torch.load("../Train/GNN_ScamDetection.pth"))
model.eval()

def gnn_scam_prob(wallet_address, gnn_model, graph_data, G):
    """
    :param wallet_address: (str)Address of wallet to check
    :param gnn_model: (torch.nn.Module) Trained GNN model
    :param graph_data: (torch_geometric.data.Data) Transaction graph data
    :param G: (networkx.Graph) Original graph mapping wallet addresses to indices
    :return: (0 to 1) Probability of being a scam. (None) if wallet is not found
    """

    # Wallet not found in transaction data
    if wallet_address not in G.nodes():
        return

    # Get predictions from trained model
    with torch.no_grad():
        predictions = gnn_model(graph_data.x, graph_data.edge_index)
        scam_probabilities = F.softmax(predictions, dim=1)[:, 1] # Extracting scam class probability

    scam_prob_dict = {wallet: float(prob) for wallet, prob in zip(G.nodes(), scam_probabilities)}

    return scam_prob_dict.get(wallet_address, None)