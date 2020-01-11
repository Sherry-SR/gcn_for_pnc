import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_mean
from torch_geometric.nn import GraphConv, GATConv, TopKPooling
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import pdb
class WeightedGATCov(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(WeightedGATCov, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_weight):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_weight is None:
            x_j = x_j * alpha.view(-1, self.heads, 1)
        else:
            x_j = x_j * alpha.view(-1, self.heads, 1) * edge_weight.view(-1, 1, 1)
        return x_j

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
class GcnNet2Channel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, **kwargs):
        super(GcnNet2Channel, self).__init__()

        self.conv11 = GraphConv(in_channels, 40)
        self.bn11 = torch.nn.BatchNorm1d(num_nodes[0])
        self.conv12 = GraphConv(40, 40)
        self.bn12 = torch.nn.BatchNorm1d(num_nodes[0])
        self.conv13 = GraphConv(40, 1)

        self.conv21 = GraphConv(in_channels, 40)
        self.bn21 = torch.nn.BatchNorm1d(num_nodes[1])
        self.conv22 = GraphConv(40, 40)
        self.bn22 = torch.nn.BatchNorm1d(num_nodes[1])
        self.conv23 = GraphConv(40, 1)

        self.fc11 = torch.nn.Linear(num_nodes[0], 50)
        self.fc21 = torch.nn.Linear(num_nodes[1], 50)
        self.fc2 = torch.nn.Linear(100, out_channels)
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, data):
        x1, edge_index1, edge_attr1, x2, edge_index2, edge_attr2 = data.x, data.edge_index, data.edge_attr, data.feature_node, data.feature_edge_index, data.features
        epsilon = 1e-10

        h1 = self.conv11(x1, edge_index1, edge_attr1)
        h1 = F.leaky_relu(h1)
        h1 = h1.view(data.num_graphs,h1.shape[0]//data.num_graphs,h1.shape[-1])
        h1 = self.bn11(h1)
        h1 = h1.view(data.num_graphs*h1.shape[1],h1.shape[-1])
        h1 = self.conv12(h1, edge_index1, edge_attr1)
        h1 = F.leaky_relu(h1)
        h1 = h1.view(data.num_graphs,h1.shape[0]//data.num_graphs,h1.shape[-1])
        h1 = self.bn12(h1)
        h1 = h1.view(data.num_graphs*h1.shape[1],h1.shape[-1])
        h1 = self.conv13(h1, edge_index1, edge_attr1)
        h1 = F.leaky_relu(h1)
        h1 = h1.view(data.num_graphs, -1)
        h1 = (h1 - torch.mean(h1, dim = 1, keepdim = True)) / (torch.var(h1, dim = 1, keepdim = True) + epsilon)
        out1 = F.dropout(h1, p = 0.5, training = self.training)
        out1 = self.fc11(h1)

        h2 = self.conv21(x2, edge_index2, edge_attr2)
        h2 = F.leaky_relu(h2)
        h2 = h2.view(data.num_graphs,h2.shape[0]//data.num_graphs,h2.shape[-1])
        h2 = self.bn21(h2)
        h2 = h2.view(data.num_graphs*h2.shape[1],h2.shape[-1])
        h2 = self.conv22(h2, edge_index2, edge_attr2)
        h2 = F.leaky_relu(h2)
        h2 = h2.view(data.num_graphs,h2.shape[0]//data.num_graphs,h2.shape[-1])
        h2 = self.bn22(h2)
        h2 = h2.view(data.num_graphs*h2.shape[1],h2.shape[-1])
        h2 = self.conv23(h2, edge_index2, edge_attr2)
        h2 = F.leaky_relu(h2)
        h2 = h2.view(data.num_graphs, -1)
        h2 = (h2 - torch.mean(h2, dim = 1, keepdim = True)) / (torch.var(h2, dim = 1, keepdim = True) + epsilon)
        out2 = F.dropout(h2, p = 0.5, training = self.training)
        out2 = self.fc21(h2)

        out = torch.cat([out1, out2], dim = -1)
        out = self.relu(out)
        out = F.dropout(out, p = 0.5, training = self.training)
        out = self.fc2(out)

        return out

class GatNet_SF(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, **kwargs):
        super(GatNet_SF, self).__init__()

        self.conv1 = WeightedGATCov(in_channels, 10, heads=5)
        self.bn1 = torch.nn.BatchNorm1d(num_nodes)
        self.pool1 = TopKPooling(10*5, ratio=0.5)
        num_nodes = int(np.ceil(num_nodes * 0.5))

        self.conv2 = WeightedGATCov(10*5, 30, heads=1)
        self.bn2 = torch.nn.BatchNorm1d(num_nodes)
        self.pool2 = TopKPooling(30, ratio=0.5)
        num_nodes = int(np.ceil(num_nodes * 0.5))

        self.conv3 = WeightedGATCov(30, 30, heads=1)
        self.bn3 = torch.nn.BatchNorm1d(num_nodes)
        self.pool3 = TopKPooling(30, ratio=0.5)
        num_nodes = int(np.ceil(num_nodes * 0.5))

        self.fc1 = torch.nn.Linear(num_nodes * 30, 32)
        self.fc2 = torch.nn.Linear(32, out_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, data):
        x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
        epsilon = 1e-10

        h1 = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        h1 = self.bn1(h1.view(data.num_graphs,h1.shape[0]//data.num_graphs,h1.shape[-1])).view(h1.shape[0],h1.shape[-1])
        h1, edge_index, edge_attr, batch, _, _ = self.pool1(h1, edge_index, edge_attr, batch = batch)

        h1 = F.leaky_relu(self.conv2(h1, edge_index))
        h1 = self.bn2(h1.view(data.num_graphs,h1.shape[0]//data.num_graphs,h1.shape[-1])).view(h1.shape[0],h1.shape[-1])
        h1, edge_index, edge_attr, batch, _, _ = self.pool2(h1, edge_index, edge_attr, batch = batch)

        h1 = F.leaky_relu(self.conv3(h1, edge_index))
        h1 = self.bn3(h1.view(data.num_graphs,h1.shape[0]//data.num_graphs,h1.shape[-1])).view(h1.shape[0],h1.shape[-1])
        h1, edge_index, edge_attr, batch, _, _ = self.pool3(h1, edge_index, edge_attr, batch = batch)

        h1 = h1.view(data.num_graphs, -1)
        h1 = (h1 - torch.mean(h1, dim = 1, keepdim = True)) / (torch.var(h1, dim = 1, keepdim = True) + epsilon)

        out = self.relu(self.fc1(h1))
        out = F.dropout(out, p = 0.5, training = self.training)
        out = self.fc2(out)

        return out