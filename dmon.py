'''
Has classes of all the models investigated. 
They include, GCN, GAT, MinCut, GraphSAGE, DiffPool, DMon

'''


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,dense_mincut_pool, dense_diff_pool
from torch_geometric.nn import DenseGCNConv, DMoNPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch.nn import Linear





class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads))
            
        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
                 dropout=0.5, aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
    
class MinCut(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        
        in_channels = data.x.shape[1]
        hidden_channels = 32
        out_channels = len(set(data.y))
        n_clusters= 10
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.pool = Linear(hidden_channels, n_clusters)
        
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Cluster assignments
        s = self.pool(x)
        
        adj = to_dense_adj(edge_index)
        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)
        
        # Final classification
        out = self.classifier(x)
        
        return F.log_softmax(out, dim=-1), mc_loss, o_loss
    

class DiffPool(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        
        in_channels = data.x.shape[1]
        hidden_channels = 32
        out_channels = len(set(data.y))
        n_clusters= 10
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.pool = Linear(hidden_channels, n_clusters)
        
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # clsuterings
        s = self.pool(x)
        
        # losses
        adj = to_dense_adj(edge_index)
        _, _, mc_loss, o_loss = dense_diff_pool(x, adj, s)
        
        # Final classification
        out = self.classifier(x)
        
        return F.log_softmax(out, dim=-1), mc_loss, o_loss
    



# The DMoN model for unsupervised clustering
# GCN with no self loops but with skip connections
# and a pooling layer that uses the DMoN pooling method
    

class GCNWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)  # No bias, just X * W
        self.skip_connection = skip_connection
        if skip_connection:
            self.skip_weight = nn.Parameter(torch.ones(out_channels) * 0.1)
        else:
            self.skip_weight = 0

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = scatter_add(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x_lin = self.linear(x)  # X * W
        messages = norm.view(-1, 1) * x_lin[col]
        out = scatter_add(messages, row, dim=0, dim_size=x.size(0))

        if self.skip_connection:
            out = x_lin * self.skip_weight + out

        return out




class DMoN(torch.nn.Module):
    def __init__(self, in_channels, num_clusters, hidden_channels=64, gcn_skip = False):
        super().__init__()
        if gcn_skip:
            self.gcn = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        else:
            self.gcn = GCNWithSkip(in_channels, hidden_channels)
        self.pool = DMoNPooling([hidden_channels, hidden_channels], num_clusters, dropout=0.5)
        # paper uses 1 layer only
        # self.conv2 = DenseGCNConv(hidden_channels, hidden_channels) 
        # self.pool2 = DMoNPooling([hidden_channels, hidden_channels], num_clusters)
        
    def forward(self, x, edge_index):
        x = F.selu(self.gcn(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        # Convert sparse to dense representation (for pooling)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x, mask = to_dense_batch(x, batch=batch) # x is now [1, N, F]
        adj = to_dense_adj(edge_index, batch=batch) # adj is now [1, N, N]
        # print("x shape after conv1:", x.shape)
        # print("adj shape:", adj.shape)
        # print("mask shape:", mask.shape)
        # Apply pooling
        clusters_assigned, x, adj, spectral_loss, ortho_loss, collapse_loss = self.pool(x, adj, mask) 
        # print("ca1:", ca.shape)  # should be [1, N, K]
        # print("x after pool1:", x.shape)
        # print("adj after pool1:", adj.shape)
        # Apply second convolution
        # x = F.selu(self.conv2(x, adj))
        # ca, x, adj, sl2, ol2, cl2 = self.pool2(x, adj)
        # print("ca2:", ca.shape)  # should still be [1, N', K]
        # print("x after pool2:", x.shape)
        # print("adj after pool2:", adj.shape)
        
        # Return cluster assignments and combined losses
        # return ca.squeeze(0), sl1 + sl2 + ol1 + ol2 + cl1 + cl2
        return clusters_assigned.squeeze(0), spectral_loss + ortho_loss + collapse_loss

    

