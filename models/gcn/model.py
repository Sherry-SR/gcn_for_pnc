import importlib

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv

class GcnNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, **kwargs):
        super(GcnNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 1)
        self.fc1 = torch.nn.Linear(num_nodes, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = F.dropout(x, p = 0.2, training = self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x)

        x = x.view(data.num_graphs, -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p = 0.2, training = self.training)
        x = self.fc2(x)
        x = self.relu(x)

        x = F.softmax(x, dim=1)

        return x