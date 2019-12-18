import importlib

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GraphConv
import pdb
class GcnNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, **kwargs):
        super(GcnNet, self).__init__()

        self.conv1 = GraphConv(in_channels, 20)
        self.bn1 = torch.nn.BatchNorm1d(num_nodes)
        self.conv2 = GraphConv(20, 20)
        self.bn2 = torch.nn.BatchNorm1d(num_nodes)
        self.conv3 = GraphConv(20, 1)

        self.fc1 = torch.nn.Linear(num_nodes, 50)
        self.fc2 = torch.nn.Linear(50, out_channels)
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr[:, 2]
        epsilon = 1e-10
        h = self.conv1(x, edge_index, edge_attr)
        h = F.leaky_relu(h)

        h = h.view(data.num_graphs,h.shape[0]//data.num_graphs,h.shape[-1])
        h = self.bn1(h)
        h = h.view(data.num_graphs*h.shape[1],h.shape[-1])

        h = self.conv2(h, edge_index, edge_attr)
        h = F.leaky_relu(h)

        h = h.view(data.num_graphs,h.shape[0]//data.num_graphs,h.shape[-1])
        h = self.bn2(h)
        h = h.view(data.num_graphs*h.shape[1],h.shape[-1])

        h = self.conv3(h, edge_index, edge_attr)
        h = F.leaky_relu(h)
        h = h.view(data.num_graphs, -1)

        out = (h - torch.mean(h, dim = 1, keepdim = True)) / (torch.var(h, dim = 1, keepdim = True) + epsilon)
        out = self.fc1(out)
        out = self.relu(out)
        out = F.dropout(out, p = 0.5, training = self.training)
        out = self.fc2(out)

        return out, h