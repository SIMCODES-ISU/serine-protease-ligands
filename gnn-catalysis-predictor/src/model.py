# gnn-catalysis-predictor/src/model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class LigandEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch=batch)

class EfficiencyRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, emb):
        return self.fc(emb)