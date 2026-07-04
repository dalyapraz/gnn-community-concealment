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





# class GCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
#         super(GCN, self).__init__()
#         self.num_layers = num_layers
#         self.convs = torch.nn.ModuleList()
#         self.dropout = dropout

#         self.convs.append(GCNConv(in_channels, hidden_channels))
        
#         for _ in range(num_layers - 2):
#             self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
#         self.convs.append(GCNConv(hidden_channels, out_channels))

#     def forward(self, x, edge_index):
#         for i in range(self.num_layers - 1):
#             x = self.convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return F.log_softmax(x, dim=1)


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



# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
#                  heads=8, dropout=0.5):
#         super(GAT, self).__init__()
#         self.num_layers = num_layers
#         self.convs = torch.nn.ModuleList()
#         self.dropout = dropout

#         self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
#         for _ in range(num_layers - 2):
#             self.convs.append(
#                 GATConv(hidden_channels * heads, hidden_channels, heads=heads))
            
#         self.convs.append(
#             GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

#     def forward(self, x, edge_index):
#         for i in range(self.num_layers - 1):
#             x = self.convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return F.log_softmax(x, dim=1)

# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, 
#                  dropout=0.5, aggregator='mean'):
#         super(GraphSAGE, self).__init__()
#         self.num_layers = num_layers
#         self.convs = torch.nn.ModuleList()
#         self.dropout = dropout

#         # Input layer
#         self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        
#         # Hidden layers
#         for _ in range(num_layers - 2):
#             self.convs.append(
#                 SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            
#         # Output layer
#         self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))

#     def forward(self, x, edge_index):
#         for i in range(self.num_layers - 1):
#             x = self.convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return F.log_softmax(x, dim=1)
    
    
class MinCut(torch.nn.Module):
    """
    Unsupervised MinCutPool clustering
    Returns:
        cluster_assignments: [num_nodes, num_clusters] soft node-cluster assignments
        loss: mincut loss + orthogonality loss
    """
    def __init__(self, in_channels, num_clusters, hidden_channels=32, num_layers=1, gcn_skip=False,  dropout=0.5, ortho_regularization=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.gcn_skip = gcn_skip
        self.dropout = dropout
        self.ortho_regularization = ortho_regularization

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            if gcn_skip:
                self.convs.append(GCNWithSkip(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(in_channels, hidden_channels))
        else:
            if gcn_skip:
                self.convs.append(GCNWithSkip(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(in_channels, hidden_channels))

            for _ in range(num_layers - 1):
                if gcn_skip:
                    self.convs.append(GCNWithSkip(hidden_channels, hidden_channels))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Learns node-to-cluster assignment logits S with shape [num_nodes, num_clusters]
        self.assignment = Linear(hidden_channels, num_clusters)

    def forward(self, x, edge_index):
        # Produce GNN embeddings
        for conv in self.convs:
            x = F.selu(conv(x, edge_index)) # if we fairly compare to DMoN should use SeLU as in DMoN pipeline
            x = F.dropout(x, p=self.dropout, training=self.training) # and dropput as in DMoN
        # maps node embeddings to cluster-assignment logits
        s_logits = self.assignment(x)

        # Convert sparse to dense
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_dense, mask = to_dense_batch(x, batch=batch)
        adj_dense = to_dense_adj(edge_index, batch=batch)
        s_dense, _ = to_dense_batch(s_logits, batch=batch)

        # Apply pooling
        _, _, mincut_loss, ortho_loss = dense_mincut_pool(
            x_dense,
            adj_dense,
            s_dense,
            mask=mask
        )

        cluster_assignments = torch.softmax(s_logits, dim=-1)
        total_loss = mincut_loss + self.ortho_regularization * ortho_loss

        return cluster_assignments, total_loss


class DiffPool(torch.nn.Module):
    """
    Unsupervised DiffPool clustering
    Returns:
        cluster_assignments: [num_nodes, num_clusters] soft node-cluster assignments
        loss: link prediction loss + entropy regularization loss
    """
    def __init__(self, in_channels, num_clusters, hidden_channels=32, num_layers=1, gcn_skip=False,  dropout=0.5, link_pred_regularization=1.0, entropy_regularization=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.gcn_skip = gcn_skip
        self.dropout = dropout
        self.link_pred_regularization = link_pred_regularization
        self.entropy_regularization = entropy_regularization

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            if gcn_skip:
                self.convs.append(GCNWithSkip(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(in_channels, hidden_channels))
        else:
            if gcn_skip:
                self.convs.append(GCNWithSkip(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(in_channels, hidden_channels))

            for _ in range(num_layers - 1):
                if gcn_skip:
                    self.convs.append(GCNWithSkip(hidden_channels, hidden_channels))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Learns node-to-cluster assignment logits S with shape [num_nodes, num_clusters]
        self.assignment = Linear(hidden_channels, num_clusters)

    def forward(self, x, edge_index):
        # Produce GNN embeddings
        for conv in self.convs:
            x = F.selu(conv(x, edge_index)) # if we fairly compare to DMoN should use SeLU as in DMoN pipeline
            x = F.dropout(x, p=self.dropout, training=self.training) # and dropput as in DMoN
        # maps node embeddings to cluster-assignment logits
        s_logits = self.assignment(x)

        # Convert sparse to dense
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_dense, mask = to_dense_batch(x, batch=batch)
        adj_dense = to_dense_adj(edge_index, batch=batch)
        s_dense, _ = to_dense_batch(s_logits, batch=batch)

        # Apply pooling
        _, _, link_prediction_loss, entropy_loss = dense_diff_pool(
            x_dense,
            adj_dense,
            s_dense,
            mask=mask
        )

        cluster_assignments = torch.softmax(s_logits, dim=-1)
        total_loss = self.link_pred_regularization * link_prediction_loss + self.entropy_regularization * entropy_loss

        return cluster_assignments, total_loss




# The DMoN model for unsupervised clustering
# GCN with no self loops but with skip connections
# and a pooling layer that uses the DMoN pooling method


class DMoN(torch.nn.Module):
    """
    Deep Modularity Network (DMoN) for unsupervised community detection.
    Returns:
        clusters_assigned (torch.Tensor): Soft assignment matrix of shape
            [num_nodes, num_clusters].
        total_loss (torch.Tensor): Spectral modularity loss + orthogonality
            loss + weighted collapse regularization loss.
    """
    def __init__(self, in_channels, num_clusters, hidden_channels=32, num_layers=1, gcn_skip=True, 
                 collapse_regularization=1.0, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.gcn_skip = gcn_skip
        self.collapse_regularization = collapse_regularization
        
        # Build multiple GCN layers
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            if gcn_skip:
                # DMoN paper: GCN without self-loops + learnable skip connection
                self.convs.append(GCNWithSkip(in_channels, hidden_channels))
            else:
                # Alternative: standard GCN without self-loops
                self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
        else:
            # First layer
            if gcn_skip:
                self.convs.append(GCNWithSkip(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
            
            # Additional hidden layers
            for _ in range(num_layers - 1):
                if gcn_skip:
                    self.convs.append(GCNWithSkip(hidden_channels, hidden_channels))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
        
        self.pool = DMoNPooling([hidden_channels, hidden_channels], num_clusters, dropout=dropout)
        # paper uses 1 layer only
        # self.conv2 = DenseGCNConv(hidden_channels, hidden_channels) 
        # self.pool2 = DMoNPooling([hidden_channels, hidden_channels], num_clusters)
        
    def forward(self, x, edge_index):
        # Apply all GCN layers
        for i, conv in enumerate(self.convs):
            x = F.selu(conv(x, edge_index))
        
        # Convert sparse to dense representation (for pooling)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x, mask = to_dense_batch(x, batch=batch) # x is now [1, N, F]
        adj = to_dense_adj(edge_index, batch=batch) # adj is now [1, N, N]
        
        # Apply pooling
        cluster_assignments, x, adj, spectral_loss, ortho_loss, collapse_loss = self.pool(x, adj, mask) 
        
        # Return cluster assignments and combined losses with weighted collapse regularization
        # print(f'Spectral Loss:{spectral_loss}')
        # print(f'Orthogonal Loss: {ortho_loss}')
        # print(f'Collapse Loss: {collapse_loss}')
        total_loss = spectral_loss + ortho_loss + self.collapse_regularization * collapse_loss
        return cluster_assignments.squeeze(0), total_loss

    

