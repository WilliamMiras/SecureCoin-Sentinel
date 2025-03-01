# Detects suspicious wallet clusters and fund movements
# Nodes -> Wallets; Edges -> Transactions
# Creator: Will

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn

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