import importlib

import numpy as np
import torch

class Ndist:
    def __init__(self, inputs):
        """
        Args:
        inputs : The graph embedding with shape (num_node, num_feature)
        """
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        self.inputs = inputs
    def cal_seqdist(self, delta = 2):
        n = self.inputs.shape[0]
        seqdist = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                x = np.abs(self.inputs[i] - self.inputs[j])
                seqdist[i, j] = np.sum(x ** delta)
        seqdist = seqdist + np.transpose(np.triu(seqdist, 1))
        return seqdist    
    def cal_dist(self, delta = 2):
        return self.cal_seqdist(delta) ** (1/delta)
    def cal_dist_potensial(self, delta, epsilon):
        dist_potensial = epsilon / (self.cal_seqdist(delta) + epsilon) - np.eye(self.inputs.shape[0])
        dist_potensial = dist_potensial / np.sum(dist_potensial, axis = 0)
        return dist_potensial

class newman_watts_strogatz_modified:
    def __init__(self, K, p, seed=None, **kwargs):
        """
        Args:
        K : Each node is connected to K nearest neighbors in ring topology
        p : The probability of adding a new edge for each edge
        seed : Seed for random number generator (default=None)
        """
        self.K = K
        self.p = p
        self.seed = seed
        
    def __call__(self, p_matrix, con_matrix = None):
        """Return a Watts-Strogatz small-world graph from embedding
        Args:
        p_matrix: potential matrix of existence of an edge between two node, with shape (num_node, num_node)
        con_matrix: optional, original connectivity matrix
        Ruturns:
        G: generated graph
        """
        if con_matrix is None:
            con_matrix = np.ones(p_matrix.shape)
        num_node = p_matrix.shape[0]
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.K >= num_node:
            raise ValueError("K>=num_node, choose smaller K or larger num_node")

        ind = np.argsort(-p_matrix, axis = 1)
        G = np.zeros((num_node, num_node))

        # create a ring over n nodes
        in_ring = {}
        i = np.random.randint(num_node)
        in_ring[i] = len(in_ring)
        while len(in_ring) < num_node:
            j = 0
            while (in_ring.get(ind[i, j]) is not None) or (ind[i, j] == i):
                j += 1
                if j == num_node - 1:
                    break
            G[i, ind[i, j]] = con_matrix[i, ind[i, j]]
            i = ind[i, j]
            in_ring[i] = len(in_ring)

        keys = np.array(list(in_ring.keys()))
        values = np.array(list(in_ring.values()))
        fromv = keys[values]
        G[i, fromv[0]] = con_matrix[i, fromv[0]]
        
        # add K - 2 nearest nodes
        for j in range(2, int(self.K / 2) + 1):
            tov = np.concatenate((fromv[j:], fromv[0:j])) # the first j are now last
            for i in range(len(fromv)):
                G[fromv[i], tov[i]] = con_matrix[fromv[i], tov[i]]

        # for each edge i-w, randomly select existing node j, and add new edge i-j with probability p_matrix[i, j]
        edges = np.argwhere(G > 0)
        for edge in edges:
            np.random.shuffle(edge)
            i = edge[0]
            j = np.random.choice(num_node)
            find_candidate = True
            # no self-loops and reject if edge u-w exists
            while j == i or (G[i, j] + G[j, i]) > 0:
                if np.sum(G[i, :]==0) <= 1:
                    find_candidate = False
                    break
                j = np.random.choice(num_node)
            if find_candidate and np.random.random_sample() <  p_matrix[i, j]:
                G[i, j] = con_matrix[i , j]
        G = G + np.transpose(G)
        return G

""" for test
embedding = np.random.randn(10)
probmodel = Ndist(embedding)
prob = probmodel.cal_dist_potensial(2, 0.01)
graphmodel = newman_watts_strogatz_modified(4, 0.1)
G = graphmodel(prob)
print('done')
"""